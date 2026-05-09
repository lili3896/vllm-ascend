/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o.h
 * \brief AscendC implementation of the GDN chunk-forward output kernel.
 *
 * This is the AscendC counterpart of the Triton kernel
 *   vllm_ascend/ops/triton/fla/chunk_o.py::chunk_fwd_kernel_o
 * It is intentionally kept as an AIV-only kernel mirroring the established
 * pattern of the sister operator csrc/recurrent_gated_delta_rule. Heavy
 * matmul-heavy paths can be migrated to AIC (Matmul high-level API) in the
 * future; the corresponding design is documented in
 *   docs/source/developer_guide/Design_Documents/chunk_gated_delta_rule_o_ascendc.md
 *
 * Design highlights:
 *  - Workload split: total work units = N * H * numVTile, distributed
 *    round-robin across AIVs. Each AIV iterates over its local units serially.
 *  - Per work unit: walk over chunks t = 0..NT-1; for each chunk run
 *      step 1: load Q_t, K_t, V_t, H_t, g_t into UB (with bf16 -> fp32 cast)
 *      step 2: compute b_o = Q_t @ H_t   (BT x BV, fp32 accumulator)
 *      step 3: compute b_A = Q_t @ K_t^T (BT x BT, fp32 accumulator)
 *      step 4: optional gating (USE_G):
 *                 b_o *= exp(g_t)[:, None]
 *                 b_A *= safe_exp(g_t[:, None] - g_t[None, :])
 *      step 5: causal mask: b_A = where(i >= j, b_A, 0)
 *      step 6: b_o = scale * b_o + scale * (b_A @ V_t)
 *      step 7: cast to bf16 and store back to o
 *  - Pipeline: PIPE_MTE2 (DataCopyPad) feeds PIPE_V (vector compute) feeds
 *    PIPE_MTE3 (DataCopyPad out). PipeBarrier<PIPE_V> separates dependent
 *    vector instructions; SetFlag<HardEvent::*> / WaitFlag<HardEvent::*> are
 *    used between PIPE_MTE2 -> PIPE_V and PIPE_V -> PIPE_MTE3 transitions
 *    that the compiler cannot infer because of pointer aliasing on the
 *    rolling reduction tile.
 */

#ifndef CHUNK_GATED_DELTA_RULE_O_KERNEL_H
#define CHUNK_GATED_DELTA_RULE_O_KERNEL_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_o_tiling_data.h"

namespace ChunkGatedDeltaRuleO {

using namespace AscendC;

constexpr uint32_t BF16_BYTES        = 2;
constexpr uint32_t FP32_BYTES        = 4;
constexpr uint32_t BF16_PER_BLOCK    = 16;   // 32B / sizeof(bf16)
constexpr uint32_t FP32_PER_BLOCK    = 8;    // 32B / sizeof(fp32)
constexpr uint32_t QUEUE_DEPTH       = 1;    // single-buffered queues
constexpr uint32_t OUT_QUEUE_DEPTH   = 2;    // double-buffered output queues

template <typename T>
__aicore__ inline T CeilDivT(T a, T b) {
    return (b == 0) ? T(0) : T((a + b - 1) / b);
}

template <typename T>
__aicore__ inline T CeilAlignT(T a, T b) {
    return CeilDivT(a, b) * b;
}

/*!
 * \brief Main kernel class for ChunkGatedDeltaRuleO.
 *
 * \tparam IN_T   data type of q/k/v/o (bfloat16_t in production)
 * \tparam HSTATE_T data type of h (bfloat16_t; matches v)
 */
template <typename IN_T, typename HSTATE_T>
class ChunkGatedDeltaRuleOKernel {
public:
    __aicore__ inline ChunkGatedDeltaRuleOKernel() = default;

    /*!
     * \brief Bind kernel inputs and TilingData. Must be called before Process.
     */
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR h, GM_ADDR g,
                                GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                GM_ADDR out,
                                const ChunkGatedDeltaRuleOTilingData *tiling,
                                TPipe *pipe) {
        // Save scalar TilingData fields into __ubuf__-friendly registers.
        b_         = tiling->b;
        tMax_      = tiling->t;
        hg_        = tiling->hg;
        h_         = tiling->h;
        k_         = tiling->k;
        v_         = tiling->v;
        bt_        = tiling->bt;
        bv_        = tiling->bv;
        numVTile_  = tiling->numVTile;
        numCores_  = tiling->numCores;
        useG_      = (tiling->useG == 1);
        isVarlen_  = (tiling->isVarlen == 1);
        alignK_    = tiling->alignK;
        alignV_    = tiling->alignV;
        alignBT_   = tiling->alignBT;
        scale_     = tiling->scale;
        groupSize_ = (hg_ > 0) ? (h_ / hg_) : 1;  // GQA: group size

        blockIdx_ = GetBlockIdx();

        // Bind GlobalTensors.
        qGm_.SetGlobalBuffer((__gm__ IN_T *)q);
        kGm_.SetGlobalBuffer((__gm__ IN_T *)k);
        vGm_.SetGlobalBuffer((__gm__ IN_T *)v);
        hGm_.SetGlobalBuffer((__gm__ HSTATE_T *)h);
        if (useG_) {
            gGm_.SetGlobalBuffer((__gm__ float *)g);
        }
        if (isVarlen_) {
            cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
            chunkOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)chunkOffsets);
        }
        outGm_.SetGlobalBuffer((__gm__ IN_T *)out);

        pipe_ = pipe;
        InitLocalBuffers();
    }

    /*!
     * \brief Driver: walk through all (n, h, v_tile) work units owned by this
     *        AIV and process them one by one.
     */
    __aicore__ inline void Process() {
        // Work-unit enumeration. We intentionally compute per-sequence NT here
        // instead of on the device for varlen, because cuSeqlens lives in GM
        // and the load is one int32 per seq.
        for (uint32_t nIdx = 0; nIdx < b_; ++nIdx) {
            uint32_t bos, eos;
            if (isVarlen_) {
                bos = static_cast<uint32_t>(cuSeqlensGm_.GetValue(nIdx));
                eos = static_cast<uint32_t>(cuSeqlensGm_.GetValue(nIdx + 1));
                bohN_ = static_cast<uint32_t>(chunkOffsetsGm_.GetValue(nIdx));
            } else {
                bos = nIdx * tMax_;
                eos = bos + tMax_;
                bohN_ = nIdx * CeilDivT<uint32_t>(tMax_, bt_);
            }
            uint32_t tThis = (eos > bos) ? (eos - bos) : 0;
            if (tThis == 0) {
                continue;
            }
            uint32_t nt = CeilDivT<uint32_t>(tThis, bt_);

            // Walk every (head, vTile) pair owned by this core.
            for (uint32_t hIdx = 0; hIdx < h_; ++hIdx) {
                for (uint32_t vTileIdx = 0; vTileIdx < numVTile_; ++vTileIdx) {
                    uint64_t unitId = (uint64_t)nIdx * h_ * numVTile_
                                    + (uint64_t)hIdx * numVTile_
                                    + vTileIdx;
                    if ((unitId % numCores_) != static_cast<uint64_t>(blockIdx_)) {
                        continue;
                    }
                    curNidxForG_ = nIdx;
                    curBosForG_  = bos;
                    ProcessHeadVTile(nIdx, hIdx, vTileIdx, bos, tThis, nt);
                }
            }
        }
    }

private:
    /*!
     * \brief Allocate UB queues and scratch buffers.
     *
     * UB layout (sized to BT=64, BV=64, K=128, V=128, bf16):
     *   qInQueue  : BT * alignK * sizeof(bf16)              =  16 KiB
     *   kInQueue  : BT * alignK * sizeof(bf16)              =  16 KiB
     *   vInQueue  : BT * alignBV * sizeof(bf16)             =   8 KiB
     *   hInQueue  : alignK * alignBV * sizeof(bf16)         =  16 KiB
     *   gInQueue  : alignBT * sizeof(fp32)                  = 256 B
     *   outQueue  : BT * alignBV * sizeof(bf16) (depth 2)   =  16 KiB
     *   scratch (TBuf):
     *     b_o   : BT * alignBV * sizeof(fp32)               =  16 KiB
     *     b_A   : BT * alignBT * sizeof(fp32)               =  16 KiB
     *     b_q_f : BT * alignK  * sizeof(fp32)               =  32 KiB
     *     b_k_f : BT * alignK  * sizeof(fp32)               =  32 KiB
     *     b_v_f : BT * alignBV * sizeof(fp32)               =  16 KiB
     *     b_h_f : alignK * alignBV * sizeof(fp32)           =  32 KiB
     *     b_g   : alignBT * sizeof(fp32)                    = 256 B
     *     b_eg  : alignBT * sizeof(fp32) (= exp(g))         = 256 B
     *     row_acc : alignBV * sizeof(fp32) (per-row accum)  = 256 B
     *   Total ~ 220 KiB which fits in 256 KiB UB.
     */
    __aicore__ inline void InitLocalBuffers() {
        const uint32_t qBytes  = bt_ * alignK_  * sizeof(IN_T);
        const uint32_t kBytes  = bt_ * alignK_  * sizeof(IN_T);
        const uint32_t vBytes  = bt_ * alignV_  * sizeof(IN_T);  // V-tile padded
        const uint32_t hBytes  = alignK_ * alignV_ * sizeof(HSTATE_T);
        const uint32_t gBytes  = alignBT_ * sizeof(float);
        const uint32_t oBytes  = bt_ * alignV_  * sizeof(IN_T);

        pipe_->InitBuffer(qInQueue_, QUEUE_DEPTH, qBytes);
        pipe_->InitBuffer(kInQueue_, QUEUE_DEPTH, kBytes);
        pipe_->InitBuffer(vInQueue_, QUEUE_DEPTH, vBytes);
        pipe_->InitBuffer(hInQueue_, QUEUE_DEPTH, hBytes);
        if (useG_) {
            pipe_->InitBuffer(gInQueue_, QUEUE_DEPTH, gBytes);
        }
        pipe_->InitBuffer(outQueue_, OUT_QUEUE_DEPTH, oBytes);

        const uint32_t boFp32Bytes  = bt_ * alignV_  * sizeof(float);
        const uint32_t bAFp32Bytes  = bt_ * alignBT_ * sizeof(float);
        const uint32_t qFp32Bytes   = bt_ * alignK_  * sizeof(float);
        const uint32_t kFp32Bytes   = bt_ * alignK_  * sizeof(float);
        const uint32_t vFp32Bytes   = bt_ * alignV_  * sizeof(float);
        const uint32_t hFp32Bytes   = alignK_ * alignV_ * sizeof(float);
        const uint32_t gExpBytes    = alignBT_ * sizeof(float);
        const uint32_t rowBytes     = alignV_ * sizeof(float);

        const uint32_t scratchBytes = boFp32Bytes + bAFp32Bytes + qFp32Bytes
                                    + kFp32Bytes + vFp32Bytes + hFp32Bytes
                                    + 2 * gExpBytes + rowBytes;
        pipe_->InitBuffer(scratch_, scratchBytes);

        uint32_t off = 0;
        boFp32_ = scratch_.GetWithOffset<float>(bt_ * alignV_, off);  off += boFp32Bytes;
        bAFp32_ = scratch_.GetWithOffset<float>(bt_ * alignBT_, off); off += bAFp32Bytes;
        qFp32_  = scratch_.GetWithOffset<float>(bt_ * alignK_,  off); off += qFp32Bytes;
        kFp32_  = scratch_.GetWithOffset<float>(bt_ * alignK_,  off); off += kFp32Bytes;
        vFp32_  = scratch_.GetWithOffset<float>(bt_ * alignV_,  off); off += vFp32Bytes;
        hFp32_  = scratch_.GetWithOffset<float>(alignK_ * alignV_, off); off += hFp32Bytes;
        gFp32_  = scratch_.GetWithOffset<float>(alignBT_, off);       off += gExpBytes;
        gExp_   = scratch_.GetWithOffset<float>(alignBT_, off);       off += gExpBytes;
        rowAcc_ = scratch_.GetWithOffset<float>(alignV_, off);
    }

    /*!
     * \brief Process all chunks of one (sequence, head, vTile) work unit.
     */
    __aicore__ inline void ProcessHeadVTile(uint32_t nIdx, uint32_t hIdx,
                                            uint32_t vTileIdx, uint32_t bos,
                                            uint32_t tThis, uint32_t nt) {
        const uint32_t kvHead = (groupSize_ > 0) ? (hIdx / groupSize_) : 0;
        const uint32_t vStart = vTileIdx * bv_;
        const uint32_t curBV  = (vStart + bv_ <= v_) ? bv_ : (v_ - vStart);

        for (uint32_t tIdx = 0; tIdx < nt; ++tIdx) {
            const uint32_t tokenStart = tIdx * bt_;
            const uint32_t curBT      = (tokenStart + bt_ <= tThis)
                                            ? bt_
                                            : (tThis - tokenStart);
            const uint64_t globalChunkIdx = bohN_ + tIdx;

            // Step 1: copy Q_t, K_t, V_t, H_t (and g_t if needed) into UB.
            CopyInQK(bos + tokenStart, kvHead, curBT);
            CopyInV(bos + tokenStart, hIdx, vStart, curBT, curBV);
            CopyInH(globalChunkIdx, hIdx, vStart, curBV);
            if (useG_) {
                CopyInG(bos + tokenStart, hIdx, curBT);
            }

            // Step 2 & 3: compute b_o (BT, BV) and b_A (BT, BT).
            ComputeQH(curBT, curBV);                   // b_o += Q_t @ H_t
            ComputeQKt(curBT);                         // b_A  = Q_t @ K_t^T
            // Free q/k/h/v queues now that fp32 working buffers hold the data.
            FreeAfterCompute();

            // Step 4: gating.
            if (useG_) {
                ApplyGating(curBT, curBV);
            }

            // Step 5: causal mask on b_A and Step 6: combine with V_t.
            ApplyCausalMaskAndCombine(curBT, curBV);

            // Step 7: cast to bf16 and store back to o.
            CastAndStoreOut(bos + tokenStart, hIdx, vStart, curBT, curBV);
        }
    }

    // ------------------------------------------------------------------
    // Step 1: Copy Inputs
    // ------------------------------------------------------------------

    /*!
     * \brief Copy a Q/K chunk slice into UB and cast it to fp32 working buffers.
     *
     * Source layout: q,k are [B, T, Hg, K] contiguous.
     *   token_offset_in_seq * (Hg * K) + kv_head * K
     * is the start of the row block this chunk needs.
     * One row of the chunk equals one full token's K vector for this head.
     * We use DataCopyPad with srcStride to skip the other Hg-1 KV heads per
     * token (the kv stride between consecutive tokens).
     */
    __aicore__ inline void CopyInQK(uint32_t globalToken, uint32_t kvHead,
                                    uint32_t curBT) {
        const uint64_t rowStart = (uint64_t)globalToken * hg_ * k_
                                + (uint64_t)kvHead * k_;

        DataCopyExtParams qkParams{
            static_cast<uint16_t>(curBT),
            static_cast<uint32_t>(k_ * sizeof(IN_T)),
            static_cast<uint32_t>((hg_ - 1) * k_ * sizeof(IN_T)),  // srcStride bytes
            0,                                                      // dstStride blocks
            0,                                                      // rsv
        };
        DataCopyPadExtParams<IN_T> qkPad{
            true, 0,
            static_cast<uint8_t>(alignK_ - k_),  // pad in datablock units
            0,
        };

        LocalTensor<IN_T> qLocal = qInQueue_.AllocTensor<IN_T>();
        DataCopyPad(qLocal, qGm_[rowStart], qkParams, qkPad);
        qInQueue_.EnQue<IN_T>(qLocal);

        LocalTensor<IN_T> kLocal = kInQueue_.AllocTensor<IN_T>();
        DataCopyPad(kLocal, kGm_[rowStart], qkParams, qkPad);
        kInQueue_.EnQue<IN_T>(kLocal);

        // Cast to fp32 working buffers; multiply Q by scale on the fly to fold
        // the scale into b_o = scale * (Q @ H) + scale * (b_A @ V) -- since b_o
        // accumulates Q @ H, doing this cast-then-Muls saves one extra pass.
        // Note: we apply *scale* later (to keep both terms consistent and to
        // mirror the Triton kernel which multiplies after the dot products);
        // here we only cast.
        qLocal = qInQueue_.DeQue<IN_T>();
        kLocal = kInQueue_.DeQue<IN_T>();
        Cast(qFp32_, qLocal, RoundMode::CAST_NONE, curBT * alignK_);
        Cast(kFp32_, kLocal, RoundMode::CAST_NONE, curBT * alignK_);
        PipeBarrier<PIPE_V>();
        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);
    }

    /*!
     * \brief Copy V chunk slice (V-tile only) into UB and cast.
     *
     * V is [B, T, H, V_full]; we only need the [BT, BV] submatrix:
     *   for token in [tokenStart, tokenStart+curBT):
     *     v[token, hIdx, vStart : vStart+curBV]
     */
    __aicore__ inline void CopyInV(uint32_t globalToken, uint32_t hIdx,
                                   uint32_t vStart, uint32_t curBT,
                                   uint32_t curBV) {
        const uint64_t rowStart = (uint64_t)globalToken * h_ * v_
                                + (uint64_t)hIdx * v_
                                + vStart;
        DataCopyExtParams vParams{
            static_cast<uint16_t>(curBT),
            static_cast<uint32_t>(curBV * sizeof(IN_T)),
            static_cast<uint32_t>((h_ * v_ - curBV) * sizeof(IN_T)),
            0, 0,
        };
        DataCopyPadExtParams<IN_T> vPad{
            true, 0,
            static_cast<uint8_t>(alignV_ - curBV), 0,
        };

        LocalTensor<IN_T> vLocal = vInQueue_.AllocTensor<IN_T>();
        DataCopyPad(vLocal, vGm_[rowStart], vParams, vPad);
        vInQueue_.EnQue<IN_T>(vLocal);
        vLocal = vInQueue_.DeQue<IN_T>();
        Cast(vFp32_, vLocal, RoundMode::CAST_NONE, curBT * alignV_);
        PipeBarrier<PIPE_V>();
        vInQueue_.FreeTensor(vLocal);
    }

    /*!
     * \brief Copy H chunk slice (K x BV) into UB and cast.
     *
     * H is [B, NT_total, H, K, V_full]. The [K, BV] submatrix at the
     * (globalChunkIdx, hIdx) head, V-tile starting at vStart, is laid out
     * with row stride = V_full bytes.
     */
    __aicore__ inline void CopyInH(uint64_t globalChunkIdx, uint32_t hIdx,
                                   uint32_t vStart, uint32_t curBV) {
        const uint64_t baseElem = (globalChunkIdx * h_ + hIdx)
                                * (uint64_t)k_ * v_
                                + vStart;
        DataCopyExtParams hParams{
            static_cast<uint16_t>(k_),
            static_cast<uint32_t>(curBV * sizeof(HSTATE_T)),
            static_cast<uint32_t>((v_ - curBV) * sizeof(HSTATE_T)),
            0, 0,
        };
        DataCopyPadExtParams<HSTATE_T> hPad{
            true, 0,
            static_cast<uint8_t>(alignV_ - curBV), 0,
        };

        LocalTensor<HSTATE_T> hLocal = hInQueue_.AllocTensor<HSTATE_T>();
        DataCopyPad(hLocal, hGm_[baseElem], hParams, hPad);
        hInQueue_.EnQue<HSTATE_T>(hLocal);
        hLocal = hInQueue_.DeQue<HSTATE_T>();
        Cast(hFp32_, hLocal, RoundMode::CAST_NONE, k_ * alignV_);
        PipeBarrier<PIPE_V>();
        hInQueue_.FreeTensor(hLocal);
    }

    /*!
     * \brief Copy g (BT scalars) into UB.
     *
     * g has been transposed to [B, H, T] on the host so that for a given head
     * the BT values are contiguous. The starting offset is
     *   nIdx*H*T_max + hIdx*T_max + tokenInSeq
     * but here we already have globalToken = bos + tokenStart so the offset
     * collapses to:
     *   globalToken     (within the per-head slice)
     * Wait: g is [B, H, T_max]. nIdx is the *batch* index of g, but
     * globalToken = bos + tokenStart and bos = nIdx*T_max for fixed-len. For
     * varlen, g still has shape [B, H, T_max] (padded), so we have to compute
     * the offset from (nIdx, hIdx, tokenInSeq) explicitly.
     */
    __aicore__ inline void CopyInG(uint32_t globalToken, uint32_t hIdx,
                                   uint32_t curBT) {
        // Recover (nIdx, tokenInSeq) from globalToken: with varlen, globalToken
        // is the absolute token index (already shifted by bos), so we need
        // tokenInSeq = globalToken - bos. We carry bos via curBosForG_.
        const uint32_t tokenInSeq = globalToken - curBosForG_;
        const uint64_t off = (uint64_t)curNidxForG_ * h_ * tMax_
                           + (uint64_t)hIdx * tMax_
                           + tokenInSeq;
        DataCopyExtParams gParams{
            1,
            static_cast<uint32_t>(curBT * sizeof(float)),
            0, 0, 0,
        };
        DataCopyPadExtParams<float> gPad{
            true, 0,
            static_cast<uint8_t>(alignBT_ - curBT), 0,
        };

        LocalTensor<float> gLocal = gInQueue_.AllocTensor<float>();
        DataCopyPad(gLocal, gGm_[off], gParams, gPad);
        gInQueue_.EnQue<float>(gLocal);
        gLocal = gInQueue_.DeQue<float>();
        DataCopy(gFp32_, gLocal, alignBT_);
        PipeBarrier<PIPE_V>();
        gInQueue_.FreeTensor(gLocal);
    }

    // ------------------------------------------------------------------
    // Step 2 & 3: matmul-by-row (AIV emulation)
    // ------------------------------------------------------------------

    /*!
     * \brief Compute b_o = Q_t @ H_t  (shape [BT, alignV]).
     *
     * Outer product accumulation: for each k in [0, K), b_o += Q_t[:, k] *
     * H_t[k, :]. With BT=64 rows and BV=64 cols and K=128 reduction, the inner
     * loop has 128 vector-vector MulAddDst ops of length alignV.
     *
     * Implementation pattern uses MulAddDst across rows, where the per-row
     * scalar Q_t[r, k] is fetched via GetValue (BT * K = 8K scalar loads
     * total). The vector body fully utilises the V-vector unit. This mirrors
     * the MatVecMul pattern in csrc/recurrent_gated_delta_rule.
     */
    __aicore__ inline void ComputeQH(uint32_t curBT, uint32_t curBV) {
        // Initialise the BT x alignV accumulator to zero.
        Duplicate(boFp32_, 0.0f, bt_ * alignV_);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < curBT; ++r) {
            const uint32_t qRowOff = r * alignK_;
            const uint32_t oRowOff = r * alignV_;
            for (uint32_t kk = 0; kk < k_; ++kk) {
                const float qScalar = qFp32_.GetValue(qRowOff + kk);
                // b_o[r, :] += qScalar * H[kk, :]
                // Encoded as: tmp = H[kk, :] * qScalar; b_o[r, :] += tmp.
                // We re-use rowAcc_ (alignV_ floats) as the tmp slot.
                Muls(rowAcc_, hFp32_[kk * alignV_], qScalar, alignV_);
                PipeBarrier<PIPE_V>();
                Add(boFp32_[oRowOff], boFp32_[oRowOff], rowAcc_, alignV_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    /*!
     * \brief Compute b_A = Q_t @ K_t^T  (shape [BT, alignBT]).
     *
     * For each (i, j) pair within the chunk:
     *   b_A[i, j] = <Q_t[i, :], K_t[j, :]>
     * We implement it as an outer-product accumulation similar to ComputeQH:
     *   for k in [0, K): b_A += Q_t[:, k] * K_t[:, k] (broadcast pattern)
     * We pre-compute Q_t[:, k] (BT scalars) once per k and update each row.
     * Total elementary ops: BT * BT * K. With BT=64, K=128 that is 524288
     * fp32 FMAs spread across 64 rows; the inner Axpy length is alignBT (= 64).
     */
    __aicore__ inline void ComputeQKt(uint32_t curBT) {
        // For row i: b_A[i, j] = <Q_t[i, :], K_t[j, :]>
        //                     = sum_k Q_t[i, k] * K_t[j, k]
        // Implementation: accumulate per (i, j) pair as a true scalar inner
        // product. The vector-style approach here would require a transpose
        // of K (since K_t[j, k] with j varying has stride alignK_). For the
        // AIV reference impl we use the explicit scalar loop; the cube path
        // documented in the design doc replaces it with a single Mmad.
        for (uint32_t i = 0; i < curBT; ++i) {
            const uint32_t qRowOff = i * alignK_;
            const uint32_t aRowOff = i * alignBT_;
            for (uint32_t j = 0; j < curBT; ++j) {
                const uint32_t kRowOff = j * alignK_;
                float acc = 0.0f;
                for (uint32_t kk = 0; kk < k_; ++kk) {
                    acc += qFp32_.GetValue(qRowOff + kk)
                         * kFp32_.GetValue(kRowOff + kk);
                }
                bAFp32_.SetValue(aRowOff + j, acc);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FreeAfterCompute() {
        // q/k/v/h queues already freed inside CopyIn*. Nothing to do here.
        // Placeholder kept for future double-buffered prefetching.
    }

    // ------------------------------------------------------------------
    // Step 4: gating
    // ------------------------------------------------------------------

    /*!
     * \brief Apply gate decay to b_o and b_A.
     *
     * For inter-chunk b_o: multiply each row r by exp(g[r]).
     * For intra-chunk b_A: multiply entry (i, j) by safe_exp(g[i] - g[j]).
     * safe_exp(x) := exp(x) for x <= 0, else 0. Implemented as:
     *   safe_exp(x) = (x > 0) ? 0 : exp(x)
     * which avoids overflow in the upper-triangular region (which is later
     * masked out anyway).
     */
    __aicore__ inline void ApplyGating(uint32_t curBT, uint32_t curBV) {
        // gExp_[r] = exp(g[r])
        Exp(gExp_, gFp32_, alignBT_);
        PipeBarrier<PIPE_V>();

        // b_o[r, :] *= gExp_[r]    (broadcast over V)
        for (uint32_t r = 0; r < curBT; ++r) {
            const float scalar = gExp_.GetValue(r);
            Muls(boFp32_[r * alignV_], boFp32_[r * alignV_], scalar, alignV_);
        }
        PipeBarrier<PIPE_V>();

        // b_A[i, j] *= safe_exp(g[i] - g[j])
        for (uint32_t i = 0; i < curBT; ++i) {
            const float gi = gFp32_.GetValue(i);
            for (uint32_t j = 0; j < curBT; ++j) {
                const float diff = gi - gFp32_.GetValue(j);
                const float factor = (diff > 0.0f) ? 0.0f
                                                   : ScalarExp(diff);
                const float prev = bAFp32_.GetValue(i * alignBT_ + j);
                bAFp32_.SetValue(i * alignBT_ + j, prev * factor);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------------
    // Step 5 & 6: causal mask and combine
    // ------------------------------------------------------------------

    __aicore__ inline void ApplyCausalMaskAndCombine(uint32_t curBT,
                                                     uint32_t curBV) {
        // Step 5: causal mask. Set b_A[i, j] = 0 for j > i.
        for (uint32_t i = 0; i < curBT; ++i) {
            for (uint32_t j = i + 1; j < curBT; ++j) {
                bAFp32_.SetValue(i * alignBT_ + j, 0.0f);
            }
        }
        PipeBarrier<PIPE_V>();

        // Step 6: b_o = scale * b_o + scale * (b_A @ V_t)
        //       (the second term is computed as outer-product accumulation
        //        across the BT reduction dim of b_A)
        // First, scale b_o.
        Muls(boFp32_, boFp32_, scale_, bt_ * alignV_);
        PipeBarrier<PIPE_V>();

        // Now accumulate scale * (b_A @ V_t) into b_o.
        // For row i we sweep j in [0, i] (the masked region) and add
        //   (scale * b_A[i, j]) * V_t[j, :]
        // We use Muls(rowAcc_) + Add(boFp32_) for each (i, j) pair.
        for (uint32_t i = 0; i < curBT; ++i) {
            const uint32_t boRowOff = i * alignV_;
            const uint32_t aRowOff  = i * alignBT_;
            for (uint32_t j = 0; j <= i; ++j) {
                const float aij = bAFp32_.GetValue(aRowOff + j);
                if (aij == 0.0f) {
                    continue;
                }
                Muls(rowAcc_, vFp32_[j * alignV_], scale_ * aij, alignV_);
                PipeBarrier<PIPE_V>();
                Add(boFp32_[boRowOff], boFp32_[boRowOff], rowAcc_, alignV_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 7: cast to bf16 and store back
    // ------------------------------------------------------------------

    __aicore__ inline void CastAndStoreOut(uint32_t globalToken, uint32_t hIdx,
                                           uint32_t vStart, uint32_t curBT,
                                           uint32_t curBV) {
        LocalTensor<IN_T> outLocal = outQueue_.AllocTensor<IN_T>();
        Cast(outLocal, boFp32_, RoundMode::CAST_RINT, bt_ * alignV_);
        outQueue_.EnQue<IN_T>(outLocal);
        outLocal = outQueue_.DeQue<IN_T>();

        const uint64_t outBase = (uint64_t)globalToken * h_ * v_
                               + (uint64_t)hIdx * v_
                               + vStart;
        DataCopyExtParams oParams{
            static_cast<uint16_t>(curBT),
            static_cast<uint32_t>(curBV * sizeof(IN_T)),
            0,                                                       // srcStride blocks (UB src is row-packed)
            static_cast<uint32_t>((h_ * v_ - curBV) * sizeof(IN_T)), // dstStride bytes
            0,
        };
        DataCopyPad(outGm_[outBase], outLocal, oParams);
        outQueue_.FreeTensor(outLocal);
    }

    // ------------------------------------------------------------------
    // Misc helpers
    // ------------------------------------------------------------------

    /*!
     * \brief Scalar exp; on AIV we have to use the IRS exponent unit. AscendC
     *        does not expose a stand-alone scalar exp, so emulate with a
     *        1-element vector exp via a tiny scratch slot. We reuse rowAcc_[0].
     */
    __aicore__ inline float ScalarExp(float x) {
        rowAcc_.SetValue(0, x);
        Exp(rowAcc_, rowAcc_, FP32_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        return rowAcc_.GetValue(0);
    }

    // ---- TilingData fields (cached) ---------------------------------------
    uint32_t b_{0}, tMax_{0}, hg_{0}, h_{0}, k_{0}, v_{0}, bt_{0}, bv_{0};
    uint32_t numVTile_{0}, numCores_{0};
    uint32_t alignK_{0}, alignV_{0}, alignBT_{0};
    uint32_t groupSize_{1};
    bool useG_{false}, isVarlen_{false};
    float scale_{1.0f};
    int32_t blockIdx_{0};

    // Per-batch running state.
    uint32_t bohN_{0};
    uint32_t curNidxForG_{0};
    uint32_t curBosForG_{0};

    // ---- GlobalTensors ----------------------------------------------------
    GlobalTensor<IN_T>     qGm_;
    GlobalTensor<IN_T>     kGm_;
    GlobalTensor<IN_T>     vGm_;
    GlobalTensor<HSTATE_T> hGm_;
    GlobalTensor<float>    gGm_;
    GlobalTensor<int32_t>  cuSeqlensGm_;
    GlobalTensor<int32_t>  chunkOffsetsGm_;
    GlobalTensor<IN_T>     outGm_;

    // ---- TPipe & queues ---------------------------------------------------
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN,  QUEUE_DEPTH>     qInQueue_;
    TQue<QuePosition::VECIN,  QUEUE_DEPTH>     kInQueue_;
    TQue<QuePosition::VECIN,  QUEUE_DEPTH>     vInQueue_;
    TQue<QuePosition::VECIN,  QUEUE_DEPTH>     hInQueue_;
    TQue<QuePosition::VECIN,  QUEUE_DEPTH>     gInQueue_;
    TQue<QuePosition::VECOUT, OUT_QUEUE_DEPTH> outQueue_;

    // ---- Scratch (TBuf) and slices ----------------------------------------
    TBuf<TPosition::VECCALC> scratch_;
    LocalTensor<float> boFp32_;
    LocalTensor<float> bAFp32_;
    LocalTensor<float> qFp32_;
    LocalTensor<float> kFp32_;
    LocalTensor<float> vFp32_;
    LocalTensor<float> hFp32_;
    LocalTensor<float> gFp32_;
    LocalTensor<float> gExp_;
    LocalTensor<float> rowAcc_;
};

}  // namespace ChunkGatedDeltaRuleO

#endif  // CHUNK_GATED_DELTA_RULE_O_KERNEL_H
