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
 * \brief ChunkFwdO 算子在 Ascend NPU 上的 AscendC CV 融合实现。
 *
 *  数据形状（与对外参数表一致）：
 *      q : [B, T, Hg, D]  bf16
 *      k : [B, T, Hg, D]  bf16
 *      v : [B, H, T, D]   bf16
 *      h : [B, H, NT,D,D] bf16
 *      g : [B, H, T]      fp32
 *      o : [B, H, T, D]   bf16
 *      cu_seqlens    : [N+1] int64
 *      chunk_indices : [NT, 2] int64  // [seq_id, chunk_in_seq_id]
 *
 *  设计要点：
 *      - 1 AIC + 2 AIV 的 MIX 核类型（KERNEL_TYPE_MIX_AIC_1_2），三个高
 *        阶 Matmul（Q@K^T、Q@H、A_masked@V）跑在 AIC 上，向量后处理跑
 *        在 AIV 上。
 *      - Ping-Pong 双缓冲 PING_PONG_STAGES = 2，offsets 数组分时使用：
 *          Cube1 / Vec1 用 (currStage-1) % 2
 *          Cube23 / Vec2 用 (currStage-2) % 2
 *      - 软件流水：每个主循环迭代同时启动新任务的 Phase A（Cube1+Vec1）
 *        与上一任务的 Phase B（Cube23+Vec2），不同 buf 互不冲突。
 *      - 任务划分：
 *          taskNum = vLoops × shapeBatch × numChunks × vNumHead
 *        AIC 用 GetBlockIdx() 取任务区间；AIV 用 GetBlockIdx() *
 *        GetSubBlockNum() + GetSubBlockIdx() 合并索引（成对的两个 AIV
 *        共享同一任务、按行拆分 BT）。
 *
 *  参考实现：本算子的 ping-pong 调度器、tiling 基类与 CV 融合骨架借鉴
 *  了 https://gitcode.com/cann/cannbot-skills（AscendC 算子开发 skills）、
 *  https://gitcode.com/cann/ops-transformer（lightning_indexer / sparse
 *  flash attention 的 Matmul + AIV 后处理流水）以及
 *  https://gitcode.com/cann/ops-sparse 中的双缓冲与跨核同步范式。
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

// 算法常量
constexpr uint32_t BT          = CHUNK_FWD_O_BT;      // chunk 大小，固定 64
constexpr uint32_t BV_MAX      = CHUNK_FWD_O_BV;      // V 方向块大小上限，128
constexpr uint32_t MASK_FP32   = 64;                  // 向量指令每次 mask 元素数

// CV 融合的跨核同步 flag。基地址 + buf_idx 区分 ping/pong。
constexpr uint32_t FLAG_CUBE1_DONE_BASE  = 3;
constexpr uint32_t FLAG_VEC1_DONE_BASE   = 5;
constexpr uint32_t FLAG_CUBE23_DONE_BASE = 7;
constexpr uint32_t FLAG_VEC2_DONE_BASE   = 9;

__aicore__ inline uint32_t MakeFlag(uint32_t base, int64_t buf) {
    return base + static_cast<uint32_t>(buf);
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b) { return (b == 0) ? 0 : (a + b - 1) / b; }

template <typename T>
__aicore__ inline T AlignUp(T a, T align) { return CeilDiv(a, align) * align; }

// 单个 ping-pong 槽位所携带的"任务上下文"
struct GDNFwdOOffsets {
    int64_t qOffset;
    int64_t kOffset;
    int64_t vOffset;
    int64_t hOffset;
    int64_t gOffset;
    int64_t oOffset;
    int64_t actBT;
    int64_t actBV;
    int64_t i_v;
    int64_t i_n;
    int64_t i_h;
    int64_t i_t;
    bool    valid;
};

// 把全局 task_id 解码成所有偏移
template <typename CuT, typename CiT>
__aicore__ inline void ComputeOffsetsFromTaskId(
    int64_t taskId,
    const ChunkFwdOTilingData& td,
    const GlobalTensor<CuT>& cuSeqlensGm,
    const GlobalTensor<CiT>& chunkIndicesGm,
    GDNFwdOOffsets& off)
{
    int64_t H      = td.vNumHead;
    int64_t Hg     = td.kNumHead;
    int64_t K      = td.kHeadDim;
    int64_t V      = td.vHeadDim;
    int64_t T      = td.seqlen;
    int64_t NT     = td.totalChunks;
    int64_t vLoops = td.vLoops;

    // 任务分解顺序：i_v 最内层，i_tg 中层，i_h 最外层
    int64_t i_v  = taskId % vLoops;
    int64_t rest = taskId / vLoops;
    int64_t i_tg = rest % NT;        // 全局 chunk 行号（chunk_indices 行）
    int64_t i_h  = rest / NT;

    // chunk_indices[i_tg] = [i_n, i_t_in_seq]
    int64_t i_n = chunkIndicesGm.GetValue(i_tg * 2);
    int64_t i_t = chunkIndicesGm.GetValue(i_tg * 2 + 1);

    int64_t bos  = cuSeqlensGm.GetValue(i_n);
    int64_t eos  = cuSeqlensGm.GetValue(i_n + 1);
    int64_t Tcur = eos - bos;

    int64_t i_hg = (Hg == H) ? i_h : (i_h / (H / Hg));

    // 各张量的元素偏移（详见文件头形状注释）
    off.qOffset = (bos + i_t * BT) * Hg * K + i_hg * K;
    off.kOffset = off.qOffset;
    off.vOffset = i_h * T * V + (bos + i_t * BT) * V + i_v * BV_MAX;
    off.hOffset = i_h * NT * K * V + i_tg * K * V + i_v * BV_MAX;
    off.gOffset = i_h * T + bos + i_t * BT;
    off.oOffset = off.vOffset;

    int64_t actBT = Tcur - i_t * BT;
    off.actBT = (actBT > static_cast<int64_t>(BT)) ? BT : (actBT < 0 ? 0 : actBT);
    int64_t actBV = V - i_v * BV_MAX;
    off.actBV = (actBV > static_cast<int64_t>(BV_MAX)) ? BV_MAX : actBV;

    off.i_v = i_v;
    off.i_n = i_n;
    off.i_h = i_h;
    off.i_t = i_t;
    off.valid = (off.actBT > 0);
}

// =============================================================================
// Cube 调度器（AIC）：根据 GetBlockIdx() 直接取任务区间，三次 Matmul / chunk。
// =============================================================================
template <typename Q_T, typename L0_T = float>
class ChunkFwdOAIC {
public:
    using AT_QH = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using BT_QH = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using CT_QH = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;

    using AT_QK = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using BT_QK = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, true>;   // B 转置
    using CT_QK = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;

    using AT_AV = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using BT_AV = MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using CT_AV = MatmulType<TPosition::GM, CubeFormat::ND, L0_T>;

    Matmul<AT_QH, BT_QH, CT_QH> mmQH;
    Matmul<AT_QK, BT_QK, CT_QK> mmQK;
    Matmul<AT_AV, BT_AV, CT_AV> mmAV;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkFwdOTilingData* td, TPipe* pipe)
    {
        td_   = td;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(v));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(h));
        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
        chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkIndices));

        InitWorkspaceTensors(workspace);
        InitMatmul();

        int64_t blockId = GetBlockIdx();
        int64_t numCube = td_->numCubeCore;
        if (numCube <= 0) numCube = 1;
        taskBegin_ = (td_->taskNum * blockId) / numCube;
        taskEnd_   = (td_->taskNum * (blockId + 1)) / numCube;
    }

    /*
     * Cube 主循环（软件流水）：
     *   for currStage = 1..taskCount+1:
     *       Phase A on task (currStage-1)：Cube1 (Q@K^T) → SetFlag(cube1Done)
     *       Phase B on task (currStage-2)：WaitFlag(vec1Done) →
     *                                     Cube2/Cube3 → SetFlag(cube23Done)
     *   两阶段使用不同 buf，可在 cube 上交错执行。
     */
    __aicore__ inline void Process()
    {
        int64_t taskCount = taskEnd_ - taskBegin_;
        for (int64_t s = 1; s <= taskCount + 1; ++s) {
            int64_t taskNew = s - 1;
            int64_t taskOld = s - 2;
            int64_t bufNew  = taskNew % PING_PONG_STAGES;
            int64_t bufOld  = (taskOld >= 0) ? (taskOld % PING_PONG_STAGES) : 0;

            // ------ Phase A：Cube1 on task taskNew，写 attnWs[bufNew] ------
            if (taskNew >= 0 && taskNew < taskCount) {
                ComputeOffsetsFromTaskId<int64_t, int64_t>(
                    taskBegin_ + taskNew, *td_, cuSeqlensGm_, chunkIndicesGm_, offsets_[bufNew]);

                // buf 复用：等待两轮前同 buf 的 V1 与 V2 都已消费完毕。
                if (taskNew >= static_cast<int64_t>(PING_PONG_STAGES)) {
                    CrossCoreWaitFlag(MakeFlag(FLAG_VEC1_DONE_BASE, bufNew));
                    CrossCoreWaitFlag(MakeFlag(FLAG_VEC2_DONE_BASE, bufNew));
                }

                if (offsets_[bufNew].valid) {
                    mmQK.SetTensorA(qGm_[offsets_[bufNew].qOffset], false);
                    mmQK.SetTensorB(kGm_[offsets_[bufNew].kOffset], true);
                    mmQK.SetTail(offsets_[bufNew].actBT, offsets_[bufNew].actBT, td_->kHeadDim);
                    mmQK.IterateAll(attnWsGm_[bufNew], 0);
                }
                CrossCoreSetFlag<2, PIPE_FIX>(MakeFlag(FLAG_CUBE1_DONE_BASE, bufNew));
            }

            // ------ Phase B：Cube23 on task taskOld，写 hWs/vWs[bufOld] ------
            if (taskOld >= 0 && taskOld < taskCount) {
                // 等 V1 把 amWs[bufOld] 写好
                CrossCoreWaitFlag(MakeFlag(FLAG_VEC1_DONE_BASE, bufOld));

                if (offsets_[bufOld].valid) {
                    // Cube2: Q @ H → hWs[bufOld]
                    mmQH.SetTensorA(qGm_[offsets_[bufOld].qOffset], false);
                    mmQH.SetTensorB(hGm_[offsets_[bufOld].hOffset], false);
                    mmQH.SetTail(offsets_[bufOld].actBT, offsets_[bufOld].actBV, td_->kHeadDim);
                    mmQH.IterateAll(hWsGm_[bufOld], 0);

                    // Cube3: amWs[bufOld] @ V → vWs[bufOld]
                    mmAV.SetTensorA(amWsGm_[bufOld], false);
                    mmAV.SetTensorB(vGm_[offsets_[bufOld].vOffset], false);
                    mmAV.SetTail(offsets_[bufOld].actBT, offsets_[bufOld].actBV, offsets_[bufOld].actBT);
                    mmAV.IterateAll(vWsGm_[bufOld], 0);
                }
                CrossCoreSetFlag<2, PIPE_FIX>(MakeFlag(FLAG_CUBE23_DONE_BASE, bufOld));
            }
        }
    }

private:
    __aicore__ inline void InitWorkspaceTensors(GM_ADDR workspace)
    {
        int64_t aicId       = GetBlockIdx();
        int64_t hSlotBytes  = BT * BV_MAX * sizeof(L0_T);
        int64_t attnSlotBytes = BT * BT * sizeof(L0_T);
        int64_t vSlotBytes  = hSlotBytes;
        int64_t amSlotBytes = BT * BT * sizeof(Q_T);

        int64_t hAicBytes  = PING_PONG_STAGES * hSlotBytes;
        int64_t attnAicBytes = PING_PONG_STAGES * attnSlotBytes;
        int64_t vAicBytes  = PING_PONG_STAGES * vSlotBytes;
        int64_t amAicBytes = PING_PONG_STAGES * amSlotBytes;

        for (int p = 0; p < static_cast<int>(PING_PONG_STAGES); ++p) {
            hWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ L0_T*>(
                workspace + td_->hWorkspaceOffset + aicId * hAicBytes + p * hSlotBytes));
            attnWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ L0_T*>(
                workspace + td_->attnWorkspaceOffset + aicId * attnAicBytes + p * attnSlotBytes));
            vWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ L0_T*>(
                workspace + td_->vWorkspaceOffset + aicId * vAicBytes + p * vSlotBytes));
            amWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(
                workspace + td_->aftermaskWorkspaceOffset + aicId * amAicBytes + p * amSlotBytes));
        }
    }

    __aicore__ inline void InitMatmul()
    {
        int64_t K = td_->kHeadDim;
        int64_t V = td_->vHeadDim;
        int64_t Hg = td_->kNumHead;
        // Q@H : A 行步幅 Hg*K（q 切片自 [T,Hg,K]）；B 行步幅 V（h 末两维连续）；C 行步幅 BV
        mmQH.SetOrgShape(BT, BV_MAX, Hg * K, V, BV_MAX);
        // Q@K^T : A/B 行步幅 Hg*K；C 行步幅 BT
        mmQK.SetOrgShape(BT, BT, Hg * K, Hg * K, BT);
        // A@V : A 行步幅 BT（来自 amWs 连续）；B 行步幅 V（v 切片连续）；C 行步幅 BV
        mmAV.SetOrgShape(BT, BV_MAX, BT, V, BV_MAX);
    }

    const ChunkFwdOTilingData* td_ {nullptr};
    TPipe* pipe_ {nullptr};
    int64_t taskBegin_ {0};
    int64_t taskEnd_ {0};

    GlobalTensor<Q_T>   qGm_, kGm_, vGm_, hGm_;
    GlobalTensor<L0_T>  hWsGm_[PING_PONG_STAGES];
    GlobalTensor<L0_T>  attnWsGm_[PING_PONG_STAGES];
    GlobalTensor<L0_T>  vWsGm_[PING_PONG_STAGES];
    GlobalTensor<Q_T>   amWsGm_[PING_PONG_STAGES];
    GlobalTensor<int64_t> cuSeqlensGm_, chunkIndicesGm_;

    GDNFwdOOffsets offsets_[PING_PONG_STAGES];
};

// =============================================================================
// Vec 调度器（AIV）：成对的两个 AIV 共享同一 task 区间，按 BT 行均分。
// =============================================================================
template <typename Q_T>
class ChunkFwdOAIV {
public:
    __aicore__ inline void Init(
        GM_ADDR g, GM_ADDR o,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkFwdOTilingData* td, TPipe* pipe)
    {
        td_   = td;
        pipe_ = pipe;
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(o));
        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
        chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkIndices));

        InitWorkspaceTensors(workspace);

        // UB 缓冲
        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        const uint32_t bvAlign = AlignUp<uint32_t>(BV_MAX, 16);
        pipe->InitBuffer(qhBuf_,   btAlign * bvAlign * sizeof(float));
        pipe->InitBuffer(avBuf_,   btAlign * bvAlign * sizeof(float));
        pipe->InitBuffer(attnBuf_, btAlign * btAlign * sizeof(float));
        pipe->InitBuffer(gBuf_,    btAlign * sizeof(float));
        pipe->InitBuffer(gExpBuf_, btAlign * sizeof(float));
        pipe->InitBuffer(maskBuf_, btAlign * btAlign * sizeof(float));
        pipe->InitBuffer(amBuf_,   btAlign * btAlign * sizeof(Q_T));
        pipe->InitBuffer(outBuf_,  btAlign * bvAlign * sizeof(Q_T));

        BuildCausalMaskOnce();

        // Vec 调度合并索引：blockId = GetBlockIdx()，subId = GetSubBlockIdx()
        // 同 AIC 配对的两个 AIV 共享 task 区间，差异在于行子集。
        int64_t blockId = GetBlockIdx();
        int64_t numCube = td_->numCubeCore;
        if (numCube <= 0) numCube = 1;
        taskBegin_ = (td_->taskNum * blockId) / numCube;
        taskEnd_   = (td_->taskNum * (blockId + 1)) / numCube;
    }

    __aicore__ inline void Process()
    {
        int64_t taskCount = taskEnd_ - taskBegin_;
        for (int64_t s = 1; s <= taskCount + 1; ++s) {
            int64_t taskNew = s - 1;
            int64_t taskOld = s - 2;
            int64_t bufNew  = taskNew % PING_PONG_STAGES;
            int64_t bufOld  = (taskOld >= 0) ? (taskOld % PING_PONG_STAGES) : 0;

            // ------ Phase A：Vec1（apply g + causal mask）------
            if (taskNew >= 0 && taskNew < taskCount) {
                ComputeOffsetsFromTaskId<int64_t, int64_t>(
                    taskBegin_ + taskNew, *td_, cuSeqlensGm_, chunkIndicesGm_, offsets_[bufNew]);

                if (taskNew >= static_cast<int64_t>(PING_PONG_STAGES)) {
                    CrossCoreWaitFlag(MakeFlag(FLAG_CUBE23_DONE_BASE, bufNew));
                }
                CrossCoreWaitFlag(MakeFlag(FLAG_CUBE1_DONE_BASE, bufNew));

                if (offsets_[bufNew].valid) {
                    DoVec1(bufNew, offsets_[bufNew]);
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(MakeFlag(FLAG_VEC1_DONE_BASE, bufNew));
            }

            // ------ Phase B：Vec2（fuse + cast + store）------
            if (taskOld >= 0 && taskOld < taskCount) {
                CrossCoreWaitFlag(MakeFlag(FLAG_CUBE23_DONE_BASE, bufOld));
                if (offsets_[bufOld].valid) {
                    DoVec2(bufOld, offsets_[bufOld]);
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(MakeFlag(FLAG_VEC2_DONE_BASE, bufOld));
            }
        }
    }

private:
    __aicore__ inline void InitWorkspaceTensors(GM_ADDR workspace)
    {
        int64_t aicId       = GetBlockIdx();
        int64_t hSlotBytes  = BT * BV_MAX * sizeof(float);
        int64_t attnSlotBytes = BT * BT * sizeof(float);
        int64_t vSlotBytes  = hSlotBytes;
        int64_t amSlotBytes = BT * BT * sizeof(Q_T);

        int64_t hAicBytes  = PING_PONG_STAGES * hSlotBytes;
        int64_t attnAicBytes = PING_PONG_STAGES * attnSlotBytes;
        int64_t vAicBytes  = PING_PONG_STAGES * vSlotBytes;
        int64_t amAicBytes = PING_PONG_STAGES * amSlotBytes;

        for (int p = 0; p < static_cast<int>(PING_PONG_STAGES); ++p) {
            hWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                workspace + td_->hWorkspaceOffset + aicId * hAicBytes + p * hSlotBytes));
            attnWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                workspace + td_->attnWorkspaceOffset + aicId * attnAicBytes + p * attnSlotBytes));
            vWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                workspace + td_->vWorkspaceOffset + aicId * vAicBytes + p * vSlotBytes));
            amWsGm_[p].SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(
                workspace + td_->aftermaskWorkspaceOffset + aicId * amAicBytes + p * amSlotBytes));
        }
    }

    __aicore__ inline void BuildCausalMaskOnce()
    {
        if (maskInitialized_) return;
        maskInitialized_ = true;
        LocalTensor<float> mask = maskBuf_.Get<float>();
        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        for (uint32_t i = 0; i < BT; ++i) {
            for (uint32_t j = 0; j < BT; ++j) {
                mask.SetValue(i * btAlign + j, (i >= j) ? 1.0f : 0.0f);
            }
        }
    }

    // 子核行区间：成对 AIV 平分 BT 行
    __aicore__ inline void GetRowRange(uint32_t& rowBegin, uint32_t& rowEnd)
    {
        uint32_t subId  = GetSubBlockIdx();
        uint32_t subNum = GetSubBlockNum();
        if (subNum == 0) subNum = 1;
        rowBegin = (BT * subId) / subNum;
        rowEnd   = (BT * (subId + 1)) / subNum;
    }

    __aicore__ inline void DoVec1(int64_t buf, const GDNFwdOOffsets& off)
    {
        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        const uint32_t totalAttn = btAlign * btAlign;

        LocalTensor<float> attn = attnBuf_.Get<float>();
        DataCopy(attn, attnWsGm_[buf], totalAttn);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        bool useG = td_->hasG != 0;
        if (useG) {
            LocalTensor<float> gVec = gBuf_.Get<float>();
            LocalTensor<float> gExp = gExpBuf_.Get<float>();
            DataCopy(gVec, gGm_[off.gOffset], AlignUp<uint32_t>(BT, 8));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Exp(gExp, gVec, AlignUp<uint32_t>(BT, 8));
            PipeBarrier<PIPE_V>();

            // 每个 AIV 处理 BT 行的一半
            uint32_t rowBegin, rowEnd;
            GetRowRange(rowBegin, rowEnd);
            for (uint32_t i = rowBegin; i < rowEnd && i < static_cast<uint32_t>(off.actBT); ++i) {
                float ei = gExp.GetValue(i);
                for (uint32_t j = 0; j <= i && j < static_cast<uint32_t>(off.actBT); ++j) {
                    float ej  = gExp.GetValue(j);
                    float scl = (ej > 0.0f) ? (ei / ej) : 0.0f;
                    if (scl > 1.0f) scl = 0.0f;  // 与 triton safe_exp 等价
                    float v0  = attn.GetValue(i * btAlign + j);
                    attn.SetValue(i * btAlign + j, v0 * scl);
                }
            }
            PipeBarrier<PIPE_V>();
        }

        // 因果 mask（兜底；当 USE_G 为 False 时为必需）
        LocalTensor<float> mask = maskBuf_.Get<float>();
        Mul(attn, attn, mask, totalAttn);
        PipeBarrier<PIPE_V>();

        // cast 到 dtype，写回 amWs[buf]
        LocalTensor<Q_T> amBuf = amBuf_.Get<Q_T>();
        if constexpr (std::is_same<Q_T, half>::value) {
            Cast(amBuf, attn, AscendC::RoundMode::CAST_NONE, totalAttn);
        } else {
            Cast(amBuf, attn, AscendC::RoundMode::CAST_RINT, totalAttn);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(amWsGm_[buf], amBuf, totalAttn);
    }

    __aicore__ inline void DoVec2(int64_t buf, const GDNFwdOOffsets& off)
    {
        const uint32_t btAlign = AlignUp<uint32_t>(BT, 16);
        const uint32_t bvAlign = AlignUp<uint32_t>(BV_MAX, 16);
        const uint32_t totalQH = btAlign * bvAlign;

        LocalTensor<float> qh = qhBuf_.Get<float>();
        DataCopy(qh, hWsGm_[buf], totalQH);
        LocalTensor<float> av = avBuf_.Get<float>();
        DataCopy(av, vWsGm_[buf], totalQH);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        if (td_->hasG != 0) {
            LocalTensor<float> gVec = gBuf_.Get<float>();
            DataCopy(gVec, gGm_[off.gOffset], AlignUp<uint32_t>(BT, 8));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

            uint32_t rowBegin, rowEnd;
            GetRowRange(rowBegin, rowEnd);
            for (uint32_t i = rowBegin; i < rowEnd && i < static_cast<uint32_t>(off.actBT); ++i) {
                float gi = gVec.GetValue(i);
                // 用 ScalarExp 保留与 triton 完全一致的 exp 路径
                float ei = AscendC::ScalarExp<float>(gi);
                Muls(qh[i * bvAlign], qh[i * bvAlign], ei,
                     static_cast<uint32_t>(off.actBV));
            }
            PipeBarrier<PIPE_V>();
        }

        Add(qh, qh, av, totalQH);
        PipeBarrier<PIPE_V>();
        Muls(qh, qh, td_->scale, totalQH);
        PipeBarrier<PIPE_V>();

        LocalTensor<Q_T> out = outBuf_.Get<Q_T>();
        if constexpr (std::is_same<Q_T, half>::value) {
            Cast(out, qh, AscendC::RoundMode::CAST_NONE, totalQH);
        } else {
            Cast(out, qh, AscendC::RoundMode::CAST_RINT, totalQH);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

        // 输出 o 形状 [B,H,T,D]，每行 V 个元素连续；行间隔 V（D）。
        // 由于这里 V == BV_MAX（典型）或 actBV ≤ V，DataCopyPad 顺序写
        // 即可，无需 dstStride（同一 head 的连续 BT 行就是连续 BT*V 元素）。
        DataCopyExtParams oParams {
            static_cast<uint16_t>(off.actBT),
            static_cast<uint32_t>(off.actBV * sizeof(Q_T)),
            static_cast<uint32_t>((bvAlign - off.actBV) * sizeof(Q_T) / 32),  // src srcStride（32B 单位）
            static_cast<uint32_t>((td_->vHeadDim - off.actBV) * sizeof(Q_T)), // dst stride（B）
            0
        };
        DataCopyPad(oGm_[off.oOffset], out, oParams);
    }

    const ChunkFwdOTilingData* td_ {nullptr};
    TPipe* pipe_ {nullptr};
    int64_t taskBegin_ {0};
    int64_t taskEnd_ {0};
    bool maskInitialized_ {false};

    GlobalTensor<float> gGm_;
    GlobalTensor<Q_T>   oGm_;
    GlobalTensor<float> hWsGm_[PING_PONG_STAGES];
    GlobalTensor<float> attnWsGm_[PING_PONG_STAGES];
    GlobalTensor<float> vWsGm_[PING_PONG_STAGES];
    GlobalTensor<Q_T>   amWsGm_[PING_PONG_STAGES];
    GlobalTensor<int64_t> cuSeqlensGm_, chunkIndicesGm_;

    GDNFwdOOffsets offsets_[PING_PONG_STAGES];

    TBuf<TPosition::VECCALC> qhBuf_;
    TBuf<TPosition::VECCALC> avBuf_;
    TBuf<TPosition::VECCALC> attnBuf_;
    TBuf<TPosition::VECCALC> gBuf_;
    TBuf<TPosition::VECCALC> gExpBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> amBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
};

}  // namespace ChunkFwdO

#endif  // __CHUNK_FWD_O_KERNEL_H__
