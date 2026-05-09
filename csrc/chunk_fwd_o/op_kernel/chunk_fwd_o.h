/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o.h
 * \brief ChunkFwdO 算子在 Ascend NPU 上的 AIC + AIV 混合 AscendC 实现。
 *
 *  本算子计算 FLA（Flash-Linear-Attention）前向输出的 chunk 切分公式：
 *
 *      o = (Q @ H) * exp(g) * scale + (causal_mask(Q @ K) * exp(g_i - g_j)) @ V * scale
 *
 *  三个 GEMM（Q@H、Q@K^T、A_masked@V）下放到 AIC（Cube）核上、调用 AscendC
 *  的高阶 Matmul API 完成；exp(g)、因果 mask、乘 scale 以及向 fp16/bf16 的
 *  cast 等向量计算交由 AIV（Vector）核完成。两类核之间通过 CrossCoreSetFlag /
 *  WaitFlag 同步，并共享若干位于 GM 上的 workspace 缓冲区。
 *
 *  分核策略（与 triton kernel 完全一致）：
 *      total_tasks = ceil(V/BV) * sum_n(H * 1)
 *      // 每个 program 对应一个 (i_v, n*H + h) 任务
 *      // 任务在 AIC 核间均匀划分；每个 AIC 配两个 AIV，按 chunk 串行流水
 *      // 协作处理（KERNEL_TYPE_MIX_AIC_1_2）。
 */

#ifndef __CHUNK_FWD_O_KERNEL_H__
#define __CHUNK_FWD_O_KERNEL_H__

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <type_traits>
#include "chunk_fwd_o_tiling_data.h"

namespace ChunkFwdO {

using namespace AscendC;
using namespace matmul;

// ---------------------------------------------------------------------------
// 常量与辅助函数
// ---------------------------------------------------------------------------
constexpr uint32_t BLOCK_BYTES        = 32;
constexpr uint32_t BT                 = CHUNK_FWD_O_BT;       // chunk 大小，默认 64
constexpr uint32_t BV_MAX             = CHUNK_FWD_O_BV;       // V 方向块大小，默认 128
constexpr uint32_t BK_MAX             = CHUNK_FWD_O_BK;       // K 方向块大小，默认 128

// AIC 与 AIV 之间的同步标记。
constexpr uint32_t SYNC_AIC_TO_AIV_QH    = 0;   // AIC 完成 Q@H
constexpr uint32_t SYNC_AIC_TO_AIV_QK    = 1;   // AIC 完成 Q@K
constexpr uint32_t SYNC_AIV_TO_AIC_AV    = 2;   // AIV 完成 mask，AIC 可执行 A@V
constexpr uint32_t SYNC_AIC_TO_AIV_AV    = 3;   // AIC 完成 A@V
constexpr uint32_t SYNC_AIV_TO_AIC_NEXT  = 4;   // AIV 释放上一 chunk 的 workspace

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

template <typename T>
__aicore__ inline T AlignUp(T a, T align)
{
    return CeilDiv(a, align) * align;
}

// ---------------------------------------------------------------------------
// AIC（Cube）服务：每个 chunk 完成三次 Matmul。
// ---------------------------------------------------------------------------
template <typename Q_T>
class ChunkFwdOAIC {
public:
    using L0_T = float;

    using QH_AT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using QH_BT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using QH_CT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;
    using QH_BIAS_T  = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;

    using QK_AT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using QK_BT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, true>;  // K 转置：(K, T)
    using QK_CT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;
    using QK_BIAS_T  = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;

    using AV_AT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using AV_BT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using AV_CT_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;
    using AV_BIAS_T  = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;

    Matmul<QH_AT_TYPE, QH_BT_TYPE, QH_CT_TYPE, QH_BIAS_T> mmQH;
    Matmul<QK_AT_TYPE, QK_BT_TYPE, QK_CT_TYPE, QK_BIAS_T> mmQK;
    Matmul<AV_AT_TYPE, AV_BT_TYPE, AV_CT_TYPE, AV_BIAS_T> mmAV;

    __aicore__ inline ChunkFwdOAIC() {}

    __aicore__ inline void Init(const ChunkFwdOTilingData* td, TPipe* pipe)
    {
        td_   = td;
        pipe_ = pipe;
    }

    __aicore__ inline void Process(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                   GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                   GM_ADDR workspace,
                                   int64_t taskBegin, int64_t taskEnd)
    {
        const int64_t H  = td_->vNumHead;
        const int64_t Hg = td_->kNumHead;
        const int64_t K  = td_->kHeadDim;
        const int64_t V  = td_->vHeadDim;
        const int64_t BV = (BV_MAX < V) ? BV_MAX : V;
        const int64_t BVN = CeilDiv<int64_t>(V, BV);

        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
        chunkOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkOffsets));

        for (int64_t taskId = taskBegin; taskId < taskEnd; ++taskId) {
            int64_t i_v   = taskId % BVN;
            int64_t i_nh  = taskId / BVN;
            int64_t i_n   = i_nh / H;
            int64_t i_h   = i_nh % H;

            int64_t bos   = cuSeqlensGm_.GetValue(i_n);
            int64_t Tcur  = cuSeqlensGm_.GetValue(i_n + 1) - bos;
            int64_t boh   = chunkOffsetsGm_.GetValue(i_n);
            int64_t NT    = CeilDiv<int64_t>(Tcur, BT);

            for (int64_t i_t = 0; i_t < NT; ++i_t) {
                IssueChunk(q, k, v, h, workspace,
                           bos, i_h, i_v, i_t, boh,
                           Tcur, K, V, H, Hg, BV);
            }
        }
        // 收尾：等待 AIV 把最后一个 chunk 的 AV 结果消费完毕。
        CrossCoreWaitFlag(SYNC_AIV_TO_AIC_NEXT);
    }

private:
    __aicore__ inline void IssueChunk(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                      GM_ADDR workspace,
                                      int64_t bos, int64_t i_h, int64_t i_v, int64_t i_t, int64_t boh,
                                      int64_t Tcur, int64_t K, int64_t V, int64_t H, int64_t Hg,
                                      int64_t BV)
    {
        int64_t i_hg    = (Hg == H) ? i_h : (i_h / (H / Hg));
        int64_t qOffset = (bos + i_t * BT) * (Hg * K) + i_hg * K;
        int64_t kOffset = qOffset;
        int64_t vOffset = (bos + i_t * BT) * (H * V) + i_h * V + i_v * BV;
        int64_t i_tg    = boh + i_t;
        int64_t hOffset = (i_tg * H + i_h) * K * V + i_v * BV;

        int64_t actBT = Tcur - i_t * BT;
        if (actBT > BT) actBT = BT;
        int64_t actBV = V - i_v * BV;
        if (actBV > BV) actBV = BV;

        // ---- 1) Q @ H -> hWorkspace (BT x BV, fp32) ----
        GM_ADDR hwsAddr = workspace + td_->hWorkspaceOffset;
        mmQH.SetTensorA(reinterpret_cast<__gm__ Q_T*>(q) + qOffset, /*transpose=*/false);
        mmQH.SetTensorB(reinterpret_cast<__gm__ Q_T*>(h) + hOffset, /*transpose=*/false);
        mmQH.SetOrgShape(actBT, actBV, K, V, K);
        mmQH.SetTail(actBT, actBV, K);
        mmQH.IterateAll(reinterpret_cast<__gm__ L0_T*>(hwsAddr), false);
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV_QH);

        // ---- 2) Q @ K^T -> attnWorkspace (BT x BT, fp32) ----
        GM_ADDR attnAddr = workspace + td_->attnWorkspaceOffset;
        mmQK.SetTensorA(reinterpret_cast<__gm__ Q_T*>(q) + qOffset, /*transpose=*/false);
        mmQK.SetTensorB(reinterpret_cast<__gm__ Q_T*>(k) + kOffset, /*transpose=*/true);
        mmQK.SetOrgShape(actBT, actBT, K, K, BT);
        mmQK.SetTail(actBT, actBT, K);
        mmQK.IterateAll(reinterpret_cast<__gm__ L0_T*>(attnAddr), false);
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV_QK);

        // ---- 3) 等待 AIV 把 mask 后的 A 写入 aftermaskWorkspace ----
        CrossCoreWaitFlag(SYNC_AIV_TO_AIC_AV);

        // ---- 4) A_masked @ V -> vWorkspace (BT x BV, fp32) ----
        GM_ADDR amAddr  = workspace + td_->aftermaskWorkspaceOffset;
        GM_ADDR vwsAddr = workspace + td_->vWorkspaceOffset;
        mmAV.SetTensorA(reinterpret_cast<__gm__ Q_T*>(amAddr), /*transpose=*/false);
        mmAV.SetTensorB(reinterpret_cast<__gm__ Q_T*>(v) + vOffset, /*transpose=*/false);
        mmAV.SetOrgShape(actBT, actBV, BT, H * V, BV);
        mmAV.SetTail(actBT, actBV, actBT);
        mmAV.IterateAll(reinterpret_cast<__gm__ L0_T*>(vwsAddr), false);
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV_AV);
    }

    const ChunkFwdOTilingData* td_ {nullptr};
    TPipe* pipe_ {nullptr};
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkOffsetsGm_;
};

// ---------------------------------------------------------------------------
// AIV（Vector）服务：完成 Q 缩放、gate、causal mask、partial 累加与输出 cast。
// ---------------------------------------------------------------------------
template <typename Q_T>
class ChunkFwdOAIV {
public:
    __aicore__ inline ChunkFwdOAIV() {}

    __aicore__ inline void Init(const ChunkFwdOTilingData* td, TPipe* pipe)
    {
        td_   = td;
        pipe_ = pipe;
        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        const uint32_t bvAlign = AlignUp<uint32_t>(BV_MAX, 16);
        const uint32_t qhBytes   = btAlign * bvAlign * sizeof(float);
        const uint32_t attnBytes = btAlign * btAlign * sizeof(float);
        const uint32_t gBytes    = btAlign * sizeof(float);
        const uint32_t outBytes  = btAlign * bvAlign * sizeof(Q_T);

        pipe_->InitBuffer(qhQue_,   1, qhBytes);
        pipe_->InitBuffer(avQue_,   1, qhBytes);
        pipe_->InitBuffer(attnQue_, 1, attnBytes);
        pipe_->InitBuffer(gQue_,    1, gBytes);
        pipe_->InitBuffer(maskBuf_, btAlign * btAlign * sizeof(float));
        pipe_->InitBuffer(outQue_,  1, outBytes);
        pipe_->InitBuffer(amQue_,   1, btAlign * btAlign * sizeof(Q_T));
    }

    __aicore__ inline void Process(GM_ADDR g, GM_ADDR o,
                                   GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                   GM_ADDR workspace,
                                   int64_t taskBegin, int64_t taskEnd)
    {
        BuildCausalMaskOnce();

        const int64_t H  = td_->vNumHead;
        const int64_t V  = td_->vHeadDim;
        const float scale = td_->scale;
        const int64_t BV = (BV_MAX < V) ? BV_MAX : V;
        const int64_t BVN = CeilDiv<int64_t>(V, BV);
        const int64_t Tmax = td_->seqlen;
        const bool useG = td_->hasG != 0;

        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
        chunkOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkOffsets));

        for (int64_t taskId = taskBegin; taskId < taskEnd; ++taskId) {
            int64_t i_v  = taskId % BVN;
            int64_t i_nh = taskId / BVN;
            int64_t i_n  = i_nh / H;
            int64_t i_h  = i_nh % H;

            int64_t bos  = cuSeqlensGm_.GetValue(i_n);
            int64_t Tcur = cuSeqlensGm_.GetValue(i_n + 1) - bos;
            int64_t boh  = chunkOffsetsGm_.GetValue(i_n);
            int64_t NT   = CeilDiv<int64_t>(Tcur, BT);

            for (int64_t i_t = 0; i_t < NT; ++i_t) {
                ProcessChunk(g, o, workspace,
                             bos, i_h, i_v, i_t, boh,
                             Tcur, Tmax, V, H, BV, scale, useG);
            }
        }
    }

private:
    __aicore__ inline void BuildCausalMaskOnce()
    {
        if (maskInitialized_) {
            return;
        }
        maskInitialized_ = true;
        LocalTensor<float> mask = maskBuf_.Get<float>();
        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        for (uint32_t i = 0; i < BT; ++i) {
            for (uint32_t j = 0; j < BT; ++j) {
                mask.SetValue(i * btAlign + j, (i >= j) ? 1.0f : 0.0f);
            }
        }
    }

    __aicore__ inline void ProcessChunk(GM_ADDR g, GM_ADDR o, GM_ADDR workspace,
                                        int64_t bos, int64_t i_h, int64_t i_v, int64_t i_t, int64_t boh,
                                        int64_t Tcur, int64_t Tmax, int64_t V, int64_t H, int64_t BV,
                                        float scale, bool useG)
    {
        int64_t actBT = Tcur - i_t * BT;
        if (actBT > BT) actBT = BT;
        int64_t actBV = V - i_v * BV;
        if (actBV > BV) actBV = BV;

        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        const uint32_t bvAlign = AlignUp<uint32_t>(static_cast<uint32_t>(BV), 16);
        const uint32_t totalQH   = btAlign * bvAlign;
        const uint32_t totalAttn = btAlign * btAlign;

        // ---- (a) 等待 Q@H 并加载到 UB ----
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV_QH);
        LocalTensor<float> qh = qhQue_.AllocTensor<float>();
        GlobalTensor<float> hwsGm;
        hwsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + td_->hWorkspaceOffset));
        DataCopy(qh, hwsGm, totalQH);
        qhQue_.EnQue(qh);
        qh = qhQue_.DeQue<float>();

        // ---- (b) 等待 Q@K，加载 attn ----
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV_QK);
        LocalTensor<float> attn = attnQue_.AllocTensor<float>();
        GlobalTensor<float> attnGm;
        attnGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + td_->attnWorkspaceOffset));
        DataCopy(attn, attnGm, totalAttn);
        attnQue_.EnQue(attn);
        attn = attnQue_.DeQue<float>();

        // ---- (c) 可选 gate g 处理 ----
        if (useG) {
            LocalTensor<float> gVec = gQue_.AllocTensor<float>();
            GlobalTensor<float> gGm;
            gGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
            int64_t gOff = bos + i_h * Tmax + i_t * BT;
            DataCopy(gVec, gGm[gOff], btAlign);
            gQue_.EnQue(gVec);
            gVec = gQue_.DeQue<float>();

            // 向量化 exp(g)：gExp[i] = exp(g[i])
            Exp(gVec, gVec, btAlign);
            PipeBarrier<PIPE_V>();

            // qh_row[i] *= gExp[i]   （按行缩放，actBV 个元素）
            // attn[i,j] *= (i>=j) ? gExp[i] / gExp[j] : 0   （结合因果性的 safe_exp）
            for (uint32_t i = 0; i < static_cast<uint32_t>(actBT); ++i) {
                float ei = gVec.GetValue(i);
                Muls(qh[i * bvAlign], qh[i * bvAlign], ei,
                     static_cast<uint32_t>(actBV));
                for (uint32_t j = 0; j <= i; ++j) {
                    float ej = gVec.GetValue(j);
                    float scl = (ej > 0.0f) ? (ei / ej) : 0.0f;
                    if (scl > 1.0f) {
                        // 数值保护：当 g_i - g_j > 0 时，原始 safe_exp 返回 0，
                        // 与 triton kernel 行为保持一致。
                        scl = 0.0f;
                    }
                    float v0 = attn.GetValue(i * btAlign + j);
                    attn.SetValue(i * btAlign + j, v0 * scl);
                }
                // 严格上三角部分 attn[i,j>i] 会在后续 (d) 中被因果 mask 清零，
                // 此处无需再写入。
            }
            PipeBarrier<PIPE_V>();
            gQue_.FreeTensor(gVec);
        }

        // ---- (d) 因果 mask ----
        LocalTensor<float> mask = maskBuf_.Get<float>();
        Mul(attn, attn, mask, totalAttn);
        PipeBarrier<PIPE_V>();

        // ---- (e) 将 attn 从 fp32 cast 为 Q_T，写入 aftermaskWorkspace ----
        LocalTensor<Q_T> attnQt = amQue_.AllocTensor<Q_T>();
        if constexpr (std::is_same<Q_T, half>::value) {
            Cast(attnQt, attn, AscendC::RoundMode::CAST_NONE, totalAttn);
        } else {
            Cast(attnQt, attn, AscendC::RoundMode::CAST_RINT, totalAttn);
        }
        amQue_.EnQue(attnQt);
        attnQt = amQue_.DeQue<Q_T>();
        GlobalTensor<Q_T> amGm;
        amGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(workspace + td_->aftermaskWorkspaceOffset));
        DataCopy(amGm, attnQt, totalAttn);
        amQue_.FreeTensor(attnQt);
        attnQue_.FreeTensor(attn);

        // 通知 AIC：mask 后的 A 已就绪
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC_AV);

        // ---- (f) 等待 A@V 结果 ----
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV_AV);
        LocalTensor<float> av = avQue_.AllocTensor<float>();
        GlobalTensor<float> vwsGm;
        vwsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + td_->vWorkspaceOffset));
        DataCopy(av, vwsGm, totalQH);
        avQue_.EnQue(av);
        av = avQue_.DeQue<float>();

        // ---- (g) o = (qh + av) * scale ----
        Add(qh, qh, av, totalQH);
        PipeBarrier<PIPE_V>();
        Muls(qh, qh, scale, totalQH);
        PipeBarrier<PIPE_V>();
        avQue_.FreeTensor(av);

        // ---- (h) cast 输出并写回 GM ----
        LocalTensor<Q_T> outBuf = outQue_.AllocTensor<Q_T>();
        if constexpr (std::is_same<Q_T, half>::value) {
            Cast(outBuf, qh, AscendC::RoundMode::CAST_NONE, totalQH);
        } else {
            Cast(outBuf, qh, AscendC::RoundMode::CAST_RINT, totalQH);
        }
        outQue_.EnQue(outBuf);
        outBuf = outQue_.DeQue<Q_T>();
        GlobalTensor<Q_T> oGm;
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(o));
        int64_t oOff = (bos + i_t * BT) * (H * V) + i_h * V + i_v * BV;
        DataCopyExtParams oParams {
            static_cast<uint16_t>(actBT),
            static_cast<uint32_t>(actBV * sizeof(Q_T)),
            static_cast<uint32_t>((bvAlign - actBV) * sizeof(Q_T) / 32),
            static_cast<uint32_t>((H * V - actBV) * sizeof(Q_T)),
            0
        };
        DataCopyPad(oGm[oOff], outBuf, oParams);
        outQue_.FreeTensor(outBuf);
        qhQue_.FreeTensor(qh);

        // 允许 AIC 复用 workspace 处理下一 chunk
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC_NEXT);
    }

    const ChunkFwdOTilingData* td_ {nullptr};
    TPipe* pipe_ {nullptr};
    bool maskInitialized_ {false};
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkOffsetsGm_;
    TQue<QuePosition::VECIN, 1> qhQue_;
    TQue<QuePosition::VECIN, 1> avQue_;
    TQue<QuePosition::VECIN, 1> attnQue_;
    TQue<QuePosition::VECIN, 1> gQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECOUT, 1> amQue_;
    TBuf<TPosition::VECCALC>     maskBuf_;
};

}  // namespace ChunkFwdO

#endif  // __CHUNK_FWD_O_KERNEL_H__
