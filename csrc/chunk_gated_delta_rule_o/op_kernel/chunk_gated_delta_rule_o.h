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
 * \brief AscendC implementation of the GDN chunk-forward output kernel
 *        (AIC + AIV mix mode, KERNEL_TYPE_MIX_AIC_1_2).
 *
 * AIC pipeline (per chunk t):
 *     MM1 = Matmul< Q [BT,K] , H_t [K,BV]   , out fp32 [BT,BV] >
 *     MM2 = Matmul< Q [BT,K] , K_t^T [K,BT] , out fp32 [BT,BT] >  (transposeB)
 *     -- wait MASK_DONE from AIV --
 *     MM3 = Matmul< b_A_masked [BT,BT] bf16 , V_t [BT,BV] bf16 , out fp32 [BT,BV] >
 *
 * AIV pipeline (per chunk t, two AIVs share the AIC's work; AIV0 does the
 * upper half of BT rows, AIV1 does the lower half. The mask + cast workload
 * is naturally split because the upper-triangular zeros do not depend on the
 * lower rows):
 *     -- wait MM12_DONE --
 *     load c1 (mm1) + c2 (mm2) from workspace into UB (fp32)
 *     [optional USE_G]
 *         load g_t [BT] from GM
 *         exp_g = Exp(g_t)
 *         c1 row-wise *= exp_g[r]
 *         c2[i,j] *= safe_exp(g[i] - g[j])           (vectorised by row)
 *     causal mask: zero c2[i, j > i]
 *     cast c2 (fp32) -> b_A_masked (bf16) -> back to workspace
 *     -- set MASK_DONE --
 *     -- wait MM3_DONE --
 *     load c3 (mm3) from workspace into UB
 *     b_o = scale * c1 + scale * c3
 *     cast b_o (fp32) -> bf16, DataCopyPad to out_gm
 *     -- set STORE_DONE --
 *
 * Cross-core sync uses 4 events with hard CrossCoreSetFlag/WaitFlag pairs:
 *   E_MM12_DONE (AIC->AIV) : raised by AIC after MM2 finalises (PIPE_FIX)
 *   E_MASK_DONE (AIV->AIC) : raised by AIV after b_A_masked written (PIPE_MTE3)
 *   E_MM3_DONE  (AIC->AIV) : raised by AIC after MM3 finalises (PIPE_FIX)
 *   E_STORE_DONE(AIV->AIC) : raised by AIV after out store (PIPE_MTE3)
 * Each work unit uses ping-pong workspace slots (slot = chunkIdx % 2) so the
 * AIC can start MM1+MM2 of chunk t+1 while AIV is processing chunk t.
 */

#ifndef CHUNK_GATED_DELTA_RULE_O_KERNEL_H
#define CHUNK_GATED_DELTA_RULE_O_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_o_tiling_data.h"

namespace ChunkGatedDeltaRuleO {

using namespace AscendC;
using namespace matmul;

// --------------------------------------------------------------------------
// Cross-core sync event ids. Picked to avoid the small ids the runtime may
// already use internally.
// --------------------------------------------------------------------------
constexpr int32_t E_MM12_DONE  = 6;   // AIC -> AIV
constexpr int32_t E_MASK_DONE  = 7;   // AIV -> AIC
constexpr int32_t E_MM3_DONE   = 8;   // AIC -> AIV
constexpr int32_t E_STORE_DONE = 9;   // AIV -> AIC (used to gate next chunk)

constexpr uint32_t PING_PONG = 2;
constexpr uint32_t BF16_PER_BLOCK = 16;
constexpr uint32_t FP32_PER_BLOCK = 8;

template <typename T>
__aicore__ inline T CeilDivT(T a, T b) {
    return (b == 0) ? T(0) : T((a + b - 1) / b);
}

// ==========================================================================
// AIC half: drives the three Matmul instances.
// ==========================================================================

template <typename IN_T, typename HSTATE_T, typename ACC_T = float>
class ChunkGatedDeltaRuleOAicCore {
public:
    // -- Matmul template plumbing ------------------------------------------
    using QType  = MatmulType<TPosition::GM, CubeFormat::ND, IN_T,    false>;
    using KTType = MatmulType<TPosition::GM, CubeFormat::ND, IN_T,    true>;   // transpose K
    using HType  = MatmulType<TPosition::GM, CubeFormat::ND, HSTATE_T,false>;
    using VType  = MatmulType<TPosition::GM, CubeFormat::ND, IN_T,    false>;
    using AmaskType = MatmulType<TPosition::GM, CubeFormat::ND, IN_T, false>;  // bf16 b_A_masked
    using CFp32   = MatmulType<TPosition::GM, CubeFormat::ND, ACC_T,  false>;
    using BiasN   = MatmulType<TPosition::GM, CubeFormat::ND, ACC_T,  false>;

    // mm1 = Q [BT,K] * H [K,BV] -> c1 [BT,BV] fp32
    matmul::MatmulImpl<QType, HType, CFp32, BiasN> mm1_;
    // mm2 = Q [BT,K] * K^T [K,BT] -> c2 [BT,BT] fp32  (transposeB)
    matmul::MatmulImpl<QType, KTType, CFp32, BiasN> mm2_;
    // mm3 = b_A_masked [BT,BT] bf16 * V [BT,BV] bf16 -> c3 [BT,BV] fp32
    matmul::MatmulImpl<AmaskType, VType, CFp32, BiasN> mm3_;

    __aicore__ inline ChunkGatedDeltaRuleOAicCore() = default;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                GM_ADDR workspace,
                                const ChunkGatedDeltaRuleOTilingData *tiling,
                                TPipe *pipe) {
        b_  = tiling->b;
        tMax_ = tiling->t;
        hg_ = tiling->hg;
        h_  = tiling->h;
        k_  = tiling->k;
        v_  = tiling->v;
        bt_ = tiling->bt;
        bv_ = tiling->bv;
        numVTile_ = tiling->numVTile;
        aicNum_   = tiling->aicNum;
        useG_     = (tiling->useG == 1);
        isVarlen_ = (tiling->isVarlen == 1);
        alignK_   = tiling->alignK;
        alignV_   = tiling->alignV;
        alignBT_  = tiling->alignBT;
        wsBytesC1_       = tiling->wsBytesC1;
        wsBytesC2_       = tiling->wsBytesC2;
        wsBytesAmaskBf16_= tiling->wsBytesAmaskBf16;
        wsBytesC3_       = tiling->wsBytesC3;
        wsBytesPerSlot_  = tiling->wsBytesPerSlot;
        wsBytesPerUnit_  = tiling->wsBytesPerUnit;
        groupSize_ = (hg_ > 0) ? (h_ / hg_) : 1;

        aicIdx_ = GetBlockIdx();   // index in AIC space (KERNEL_TYPE_MIX_AIC_1_2)

        qGm_.SetGlobalBuffer((__gm__ IN_T *)q);
        kGm_.SetGlobalBuffer((__gm__ IN_T *)k);
        vGm_.SetGlobalBuffer((__gm__ IN_T *)v);
        hGm_.SetGlobalBuffer((__gm__ HSTATE_T *)h);
        if (isVarlen_) {
            cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
            chunkOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)chunkOffsets);
        }
        wsGm_.SetGlobalBuffer((__gm__ uint8_t *)workspace);

        // Bind Matmul instances to TPipe and their precomputed tilings.
        mm1_.SetSubBlockIdx(0);
        mm2_.SetSubBlockIdx(0);
        mm3_.SetSubBlockIdx(0);
        mm1_.Init(&tiling->mm1Tiling, pipe);
        mm2_.Init(&tiling->mm2Tiling, pipe);
        mm3_.Init(&tiling->mm3Tiling, pipe);
    }

    __aicore__ inline void Process() {
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

            for (uint32_t hIdx = 0; hIdx < h_; ++hIdx) {
                for (uint32_t vTileIdx = 0; vTileIdx < numVTile_; ++vTileIdx) {
                    uint64_t unitId = (uint64_t)nIdx * h_ * numVTile_
                                    + (uint64_t)hIdx * numVTile_
                                    + vTileIdx;
                    if ((unitId % aicNum_) != static_cast<uint64_t>(aicIdx_)) {
                        continue;
                    }
                    ProcessAicUnit(nIdx, hIdx, vTileIdx, bos, tThis, nt, unitId);
                }
            }
        }
    }

private:
    __aicore__ inline void ProcessAicUnit(uint32_t nIdx, uint32_t hIdx,
                                          uint32_t vTileIdx, uint32_t bos,
                                          uint32_t tThis, uint32_t nt,
                                          uint64_t unitId) {
        const uint32_t kvHead = (groupSize_ > 0) ? (hIdx / groupSize_) : 0;
        const uint32_t vStart = vTileIdx * bv_;
        const uint32_t curBV  = (vStart + bv_ <= v_) ? bv_ : (v_ - vStart);
        const uint64_t unitWsBase = unitId * wsBytesPerUnit_;

        for (uint32_t tIdx = 0; tIdx < nt; ++tIdx) {
            const uint32_t tokenStart = tIdx * bt_;
            const uint32_t curBT      = (tokenStart + bt_ <= tThis)
                                            ? bt_
                                            : (tThis - tokenStart);
            const uint32_t slot = tIdx & 1U;
            const uint64_t slotBase = unitWsBase + slot * wsBytesPerSlot_;

            // Compute GM offsets for the current (n, h, t, vTile) block.
            const uint64_t qkBase = ((uint64_t)(bos + tokenStart) * hg_
                                     + kvHead) * k_;
            const uint64_t vBase  = ((uint64_t)(bos + tokenStart) * h_
                                     + hIdx) * v_ + vStart;
            const uint64_t hBase  = ((uint64_t)(bohN_ + tIdx) * h_ + hIdx)
                                  * (uint64_t)k_ * v_ + vStart;

            // Workspace pointers (cast back to typed GlobalTensor on AIV side).
            uint64_t off = slotBase;
            __gm__ ACC_T *c1Ptr  = reinterpret_cast<__gm__ ACC_T *>(wsGm_.GetPhyAddr() + off);
            off += wsBytesC1_;
            __gm__ ACC_T *c2Ptr  = reinterpret_cast<__gm__ ACC_T *>(wsGm_.GetPhyAddr() + off);
            off += wsBytesC2_;
            __gm__ IN_T  *amskPtr= reinterpret_cast<__gm__ IN_T *>(wsGm_.GetPhyAddr() + off);
            off += wsBytesAmaskBf16_;
            __gm__ ACC_T *c3Ptr  = reinterpret_cast<__gm__ ACC_T *>(wsGm_.GetPhyAddr() + off);

            GlobalTensor<ACC_T> c1Gm, c2Gm, c3Gm;
            GlobalTensor<IN_T>  amskGm;
            c1Gm.SetGlobalBuffer(c1Ptr);
            c2Gm.SetGlobalBuffer(c2Ptr);
            c3Gm.SetGlobalBuffer(c3Ptr);
            amskGm.SetGlobalBuffer(amskPtr);

            // -------- MM1 : c1 = Q_t @ H_t --------------------------------
            // Q is [T, Hg, K] -> per-token leading dim of A is Hg*K.
            // H is [K, V]      -> per-row leading dim of B is v_ (V_full).
            // C is [BT, BV]    -> packed in workspace with leading dim alignV_.
            mm1_.SetOrgShape(curBT, curBV, hg_ * k_, v_, alignV_);
            mm1_.SetSingleShape(curBT, curBV, k_);
            mm1_.SetTensorA(qGm_[qkBase], false);
            mm1_.SetTensorB(hGm_[hBase], false);
            mm1_.template IterateAll<false>(c1Gm, /*sync=*/0);

            // -------- MM2 : c2 = Q_t @ K_t^T ------------------------------
            // K is [T, Hg, K]   -> with transposeB the cube reads K^T whose
            // leading dim of B (K^T) is also hg_*k_ (the K-axis stride).
            // C is [BT, BT]     -> packed in workspace with leading dim alignBT_.
            mm2_.SetOrgShape(curBT, curBT, hg_ * k_, hg_ * k_, alignBT_);
            mm2_.SetSingleShape(curBT, curBT, k_);
            mm2_.SetTensorA(qGm_[qkBase], false);
            mm2_.SetTensorB(kGm_[qkBase], true);
            mm2_.template IterateAll<false>(c2Gm, /*sync=*/0);

            // Tell AIV that c1, c2 are ready.
            CrossCoreSetFlag<0x2, PIPE_FIX>(E_MM12_DONE);

            // Wait until AIV finished masking c2 -> b_A_masked (bf16).
            CrossCoreWaitFlag(E_MASK_DONE);

            // -------- MM3 : c3 = b_A_masked @ V_t -------------------------
            // b_A_masked is [BT, BT] packed in workspace with M-stride =
            //   alignBT_ (per-row leading dim in K direction).
            // V is [T, H, V] sliced [BT, BV] with K-stride (token stride) =
            //   h_*v_ (per row in K direction).
            // c3 is [BT, BV] packed with leading dim alignV_.
            mm3_.SetOrgShape(curBT, curBV, alignBT_, h_ * v_, alignV_);
            mm3_.SetSingleShape(curBT, curBV, curBT);
            mm3_.SetTensorA(amskGm, false);
            mm3_.SetTensorB(vGm_[vBase], false);
            mm3_.template IterateAll<false>(c3Gm, /*sync=*/0);

            CrossCoreSetFlag<0x2, PIPE_FIX>(E_MM3_DONE);

            // Wait for AIV to finish the store of chunk t before reusing the
            // ping-pong slot two chunks later. We only need to gate when we
            // are about to overwrite the slot, so this is a coarse barrier
            // every iteration; per-slot fine-grained gating is a possible
            // optimisation but adds little since cube and vector latencies
            // are well balanced for these tile sizes.
            CrossCoreWaitFlag(E_STORE_DONE);
        }
    }

    // ---- Cached TilingData ------------------------------------------------
    uint32_t b_{0}, tMax_{0}, hg_{0}, h_{0}, k_{0}, v_{0}, bt_{0}, bv_{0};
    uint32_t numVTile_{0}, aicNum_{0};
    uint32_t alignK_{0}, alignV_{0}, alignBT_{0}, groupSize_{1};
    bool     useG_{false}, isVarlen_{false};
    uint32_t wsBytesC1_{0}, wsBytesC2_{0}, wsBytesAmaskBf16_{0}, wsBytesC3_{0};
    uint32_t wsBytesPerSlot_{0}, wsBytesPerUnit_{0};
    uint32_t bohN_{0};
    int32_t  aicIdx_{0};

    GlobalTensor<IN_T>     qGm_;
    GlobalTensor<IN_T>     kGm_;
    GlobalTensor<IN_T>     vGm_;
    GlobalTensor<HSTATE_T> hGm_;
    GlobalTensor<int32_t>  cuSeqlensGm_;
    GlobalTensor<int32_t>  chunkOffsetsGm_;
    GlobalTensor<uint8_t>  wsGm_;
};

// ==========================================================================
// AIV half: gating, mask, scale composition, cast and store. Two AIVs share
// every AIC; we split the BT row dim 50/50 between them.
// ==========================================================================

template <typename IN_T, typename HSTATE_T, typename ACC_T = float>
class ChunkGatedDeltaRuleOAivCore {
public:
    __aicore__ inline ChunkGatedDeltaRuleOAivCore() = default;

    __aicore__ inline void Init(GM_ADDR g, GM_ADDR cuSeqlens,
                                GM_ADDR chunkOffsets, GM_ADDR out,
                                GM_ADDR workspace,
                                const ChunkGatedDeltaRuleOTilingData *tiling,
                                TPipe *pipe) {
        b_  = tiling->b;
        tMax_ = tiling->t;
        hg_ = tiling->hg;
        h_  = tiling->h;
        k_  = tiling->k;
        v_  = tiling->v;
        bt_ = tiling->bt;
        bv_ = tiling->bv;
        numVTile_ = tiling->numVTile;
        aicNum_   = tiling->aicNum;
        useG_     = (tiling->useG == 1);
        isVarlen_ = (tiling->isVarlen == 1);
        alignK_   = tiling->alignK;
        alignV_   = tiling->alignV;
        alignBT_  = tiling->alignBT;
        scale_    = tiling->scale;
        wsBytesC1_       = tiling->wsBytesC1;
        wsBytesC2_       = tiling->wsBytesC2;
        wsBytesAmaskBf16_= tiling->wsBytesAmaskBf16;
        wsBytesC3_       = tiling->wsBytesC3;
        wsBytesPerSlot_  = tiling->wsBytesPerSlot;
        wsBytesPerUnit_  = tiling->wsBytesPerUnit;
        groupSize_ = (hg_ > 0) ? (h_ / hg_) : 1;

        // In KERNEL_TYPE_MIX_AIC_1_2 each AIC has 2 AIVs. GetBlockIdx() on
        // AIV returns the AIV index. Pair (aiv0, aiv1) belongs to AIC i =
        // aivIdx / 2; subBlock = aivIdx & 1 picks the upper or lower row half.
        aivIdx_ = GetBlockIdx();
        aicIdx_ = aivIdx_ / 2;
        subBlock_ = aivIdx_ & 1;

        if (useG_) {
            gGm_.SetGlobalBuffer((__gm__ float *)g);
        }
        if (isVarlen_) {
            cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
            chunkOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)chunkOffsets);
        }
        outGm_.SetGlobalBuffer((__gm__ IN_T *)out);
        wsGm_.SetGlobalBuffer((__gm__ uint8_t *)workspace);

        pipe_ = pipe;
        InitLocalBuffers();
    }

    __aicore__ inline void Process() {
        for (uint32_t nIdx = 0; nIdx < b_; ++nIdx) {
            uint32_t bos, eos;
            if (isVarlen_) {
                bos = static_cast<uint32_t>(cuSeqlensGm_.GetValue(nIdx));
                eos = static_cast<uint32_t>(cuSeqlensGm_.GetValue(nIdx + 1));
            } else {
                bos = nIdx * tMax_;
                eos = bos + tMax_;
            }
            uint32_t tThis = (eos > bos) ? (eos - bos) : 0;
            if (tThis == 0) {
                continue;
            }
            uint32_t nt = CeilDivT<uint32_t>(tThis, bt_);

            for (uint32_t hIdx = 0; hIdx < h_; ++hIdx) {
                for (uint32_t vTileIdx = 0; vTileIdx < numVTile_; ++vTileIdx) {
                    uint64_t unitId = (uint64_t)nIdx * h_ * numVTile_
                                    + (uint64_t)hIdx * numVTile_
                                    + vTileIdx;
                    if ((unitId % aicNum_) != static_cast<uint64_t>(aicIdx_)) {
                        continue;
                    }
                    curNidx_ = nIdx;
                    curBos_  = bos;
                    ProcessAivUnit(nIdx, hIdx, vTileIdx, bos, tThis, nt, unitId);
                }
            }
        }
    }

private:
    /*!
     * \brief Allocate UB buffers used by the AIV half.
     *
     * UB plan (BT=64, BV=64): each AIV processes BT/2 = 32 rows, so the
     * buffers are sized to halfBT = 32 rows.
     *
     *   c1Que (fp32) : halfBT * alignV  =  8 KiB
     *   c2Que (fp32) : halfBT * alignBT =  8 KiB
     *   c3Que (fp32) : halfBT * alignV  =  8 KiB
     *   amskQue(bf16): halfBT * alignBT =  4 KiB
     *   outQue (bf16): halfBT * alignV  =  4 KiB    (depth 2 = 8 KiB)
     *   gQue   (fp32): alignBT          = 256 B
     *   scratch (TBuf):
     *     gExp_ (fp32, alignBT)             = 256 B
     *     diff_ (fp32, alignBT)             = 256 B
     *     boFp_ (fp32, halfBT*alignV)       = 8 KiB
     *     rowAcc_ (fp32, alignBT)           = 256 B
     *
     *   Total ≈ 50 KiB which leaves abundant slack in the 128 KiB AIV-side UB.
     */
    __aicore__ inline void InitLocalBuffers() {
        halfBT_  = (bt_ + 1) / 2;
        rowOff_  = subBlock_ * halfBT_;          // first row this AIV handles
        rowsThis_ = (subBlock_ == 0) ? halfBT_ : (bt_ - halfBT_);

        const uint32_t c1Bytes  = halfBT_ * alignV_  * sizeof(ACC_T);
        const uint32_t c2Bytes  = halfBT_ * alignBT_ * sizeof(ACC_T);
        const uint32_t c3Bytes  = halfBT_ * alignV_  * sizeof(ACC_T);
        const uint32_t amBytes  = halfBT_ * alignBT_ * sizeof(IN_T);
        const uint32_t oBytes   = halfBT_ * alignV_  * sizeof(IN_T);
        const uint32_t gBytes   = alignBT_           * sizeof(ACC_T);

        pipe_->InitBuffer(c1Que_,  1, c1Bytes);
        pipe_->InitBuffer(c2Que_,  1, c2Bytes);
        pipe_->InitBuffer(c3Que_,  1, c3Bytes);
        pipe_->InitBuffer(amQue_,  1, amBytes);
        pipe_->InitBuffer(outQue_, 2, oBytes);
        if (useG_) {
            pipe_->InitBuffer(gQue_, 1, gBytes);
        }

        const uint32_t scratchBytes =
            /*gExp_ */ alignBT_ * sizeof(ACC_T)
          + /*diff_ */ alignBT_ * sizeof(ACC_T)
          + /*boFp_ */ halfBT_ * alignV_ * sizeof(ACC_T)
          + /*rowAcc*/ alignBT_ * sizeof(ACC_T);
        pipe_->InitBuffer(scratch_, scratchBytes);

        uint32_t off = 0;
        gExp_   = scratch_.GetWithOffset<ACC_T>(alignBT_, off);
        off    += alignBT_ * sizeof(ACC_T);
        diff_   = scratch_.GetWithOffset<ACC_T>(alignBT_, off);
        off    += alignBT_ * sizeof(ACC_T);
        boFp_   = scratch_.GetWithOffset<ACC_T>(halfBT_ * alignV_, off);
        off    += halfBT_ * alignV_ * sizeof(ACC_T);
        rowAcc_ = scratch_.GetWithOffset<ACC_T>(alignBT_, off);
    }

    __aicore__ inline void ProcessAivUnit(uint32_t nIdx, uint32_t hIdx,
                                          uint32_t vTileIdx, uint32_t bos,
                                          uint32_t tThis, uint32_t nt,
                                          uint64_t unitId) {
        const uint32_t vStart = vTileIdx * bv_;
        const uint32_t curBV  = (vStart + bv_ <= v_) ? bv_ : (v_ - vStart);
        const uint64_t unitWsBase = unitId * wsBytesPerUnit_;

        for (uint32_t tIdx = 0; tIdx < nt; ++tIdx) {
            const uint32_t tokenStart = tIdx * bt_;
            const uint32_t curBT      = (tokenStart + bt_ <= tThis)
                                            ? bt_
                                            : (tThis - tokenStart);
            const uint32_t myRows = ComputeMyRows(curBT);
            const uint32_t slot   = tIdx & 1U;
            const uint64_t slotBase = unitWsBase + slot * wsBytesPerSlot_;

            // 1) Wait MM1 + MM2 done on AIC.
            CrossCoreWaitFlag(E_MM12_DONE);

            if (myRows > 0) {
                // 2) Load c1, c2 from workspace into UB; build b_A_masked.
                LocalTensor<ACC_T> c1 = LoadFp32(slotBase + 0,
                                                 c1Que_, myRows, alignV_);
                LocalTensor<ACC_T> c2 = LoadFp32(slotBase + wsBytesC1_,
                                                 c2Que_, myRows, alignBT_);
                LocalTensor<ACC_T> g;
                if (useG_) g = LoadG(bos + tokenStart, hIdx, curBT);
                ApplyGatingAndMask(c1, c2, g, myRows, curBT);
                if (useG_) gQue_.FreeTensor(g);

                // 3) Cast masked c2 (fp32) -> bf16, write back to workspace.
                CastAndStoreAmask(c2, slotBase + wsBytesC1_ + wsBytesC2_,
                                  myRows);
                c2Que_.FreeTensor(c2);

                // Hold c1 for the combine step (re-use after MM3_DONE).
                heldC1_ = c1;
                heldC1Valid_ = true;
            } else {
                heldC1Valid_ = false;
            }

            // Always raise MASK_DONE so AIC can proceed with MM3.
            CrossCoreSetFlag<0x2, PIPE_MTE3>(E_MASK_DONE);

            // 4) Wait MM3 done on AIC.
            CrossCoreWaitFlag(E_MM3_DONE);

            if (myRows > 0) {
                // 5) Load c3 and combine with c1.
                LocalTensor<ACC_T> c3 = LoadFp32(
                    slotBase + wsBytesC1_ + wsBytesC2_ + wsBytesAmaskBf16_,
                    c3Que_, myRows, alignV_);
                ComposeAndStore(heldC1_, c3, bos + tokenStart, hIdx, vStart,
                                curBV, myRows);
                c1Que_.FreeTensor(heldC1_);
                c3Que_.FreeTensor(c3);
                heldC1Valid_ = false;
            }
            // Always raise STORE_DONE so AIC can free this slot.
            CrossCoreSetFlag<0x2, PIPE_MTE3>(E_STORE_DONE);
        }
    }

    __aicore__ inline uint32_t ComputeMyRows(uint32_t curBT) const {
        if (rowOff_ >= curBT) {
            return 0;
        }
        return (rowOff_ + rowsThis_ <= curBT) ? rowsThis_ : (curBT - rowOff_);
    }

    /*!
     * \brief Allocate from `que`, copy `rows × cols` fp32 block from workspace
     *        at `wsByteOff` starting at row `rowOff_` of the BT slot, and
     *        return the dequeued LocalTensor.
     */
    __aicore__ inline LocalTensor<ACC_T> LoadFp32(uint64_t wsByteOff,
                                                  TQue<QuePosition::VECIN, 1> &que,
                                                  uint32_t rows, uint32_t cols) {
        const uint64_t rowOffBytes = (uint64_t)rowOff_ * cols * sizeof(ACC_T);
        __gm__ ACC_T *src = reinterpret_cast<__gm__ ACC_T *>(
            wsGm_.GetPhyAddr() + wsByteOff + rowOffBytes);
        GlobalTensor<ACC_T> srcGm;
        srcGm.SetGlobalBuffer(src);

        LocalTensor<ACC_T> dst = que.AllocTensor<ACC_T>();
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(ACC_T)),
            0, 0, 0,
        };
        DataCopyPadExtParams<ACC_T> pad{true, 0, 0, 0};
        DataCopyPad(dst, srcGm, params, pad);
        que.EnQue<ACC_T>(dst);
        return que.DeQue<ACC_T>();
    }

    /*!
     * \brief Load g [BT] for this chunk and return the dequeued LocalTensor.
     *
     * g has been transposed to [B, H, T] on the host so that values for one
     * (n, h) pair are contiguous over the time dim.
     */
    __aicore__ inline LocalTensor<ACC_T> LoadG(uint32_t globalToken,
                                               uint32_t hIdx,
                                               uint32_t curBT) {
        const uint32_t tokenInSeq = globalToken - curBos_;
        const uint64_t off = (uint64_t)curNidx_ * h_ * tMax_
                           + (uint64_t)hIdx * tMax_ + tokenInSeq;
        DataCopyExtParams params{
            1, static_cast<uint32_t>(curBT * sizeof(ACC_T)), 0, 0, 0,
        };
        DataCopyPadExtParams<ACC_T> pad{
            true, 0, static_cast<uint8_t>(alignBT_ - curBT), 0,
        };
        LocalTensor<ACC_T> g = gQue_.AllocTensor<ACC_T>();
        DataCopyPad(g, gGm_[off], params, pad);
        gQue_.EnQue<ACC_T>(g);
        return gQue_.DeQue<ACC_T>();
    }

    /*!
     * \brief Apply gate decay (if any) and the within-chunk causal mask.
     *
     * For our half (rows in [rowOff_, rowOff_+myRows)):
     *   c1[r, :] *= exp(g[rowOff_+r])
     *   c2[r, j] *= safe_exp(g[rowOff_+r] - g[j])     for j in [0, BT)
     *   c2[r, j]  = 0                                 for j > rowOff_+r
     */
    __aicore__ inline void ApplyGatingAndMask(LocalTensor<ACC_T> &c1,
                                              LocalTensor<ACC_T> &c2,
                                              LocalTensor<ACC_T> &g,
                                              uint32_t myRows,
                                              uint32_t curBT) {
        if (useG_) {
            // Pre-compute exp(g) for all BT positions (c1 row scaling needs
            // exp(g[globalR]) and c2 row scaling needs g[globalR] - g[j]).
            Exp(gExp_, g, alignBT_);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < myRows; ++r) {
                const uint32_t globalR = rowOff_ + r;
                // c1[r,:] *= exp(g[globalR])
                const ACC_T expGr = gExp_.GetValue(globalR);
                Muls(c1[r * alignV_], c1[r * alignV_], expGr, alignV_);

                // diff[j] = g[globalR] - g[j], clamp positives to 0 so that
                // exp(diff) on the upper-triangular region yields 1 (which
                // we then mask away). The lower-triangular region holds the
                // real safe_exp factor.
                const ACC_T gR = g.GetValue(globalR);
                Adds(diff_, g, -gR, alignBT_);                  // g - gR
                PipeBarrier<PIPE_V>();
                Muls(diff_, diff_, ACC_T(-1), alignBT_);        // gR - g
                PipeBarrier<PIPE_V>();
                Mins(diff_, diff_, ACC_T(0), alignBT_);         // safe clamp
                PipeBarrier<PIPE_V>();
                Exp(diff_, diff_, alignBT_);
                PipeBarrier<PIPE_V>();
                Mul(c2[r * alignBT_], c2[r * alignBT_], diff_, alignBT_);
                PipeBarrier<PIPE_V>();
            }
        }

        // Causal mask: zero c2[r, j] for j > rowOff_+r so that the cube MM3
        // sees a strictly lower-triangular b_A_masked.
        for (uint32_t r = 0; r < myRows; ++r) {
            const uint32_t globalR = rowOff_ + r;
            const uint32_t keep = (globalR + 1 <= alignBT_) ? (globalR + 1) : alignBT_;
            const uint32_t zeroLen = alignBT_ - keep;
            if (zeroLen > 0) {
                Duplicate(c2[r * alignBT_ + keep], ACC_T(0), zeroLen);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    /*!
     * \brief Cast c2 (fp32, masked) -> b_A_masked (bf16) and write to
     *        workspace. The fp32 buffer `c2` is left intact for the caller
     *        to free.
     */
    __aicore__ inline void CastAndStoreAmask(LocalTensor<ACC_T> &c2,
                                             uint64_t wsByteOff,
                                             uint32_t myRows) {
        LocalTensor<IN_T> am = amQue_.AllocTensor<IN_T>();
        Cast(am, c2, RoundMode::CAST_RINT, myRows * alignBT_);
        amQue_.EnQue<IN_T>(am);
        am = amQue_.DeQue<IN_T>();

        const uint64_t rowOffBytes = (uint64_t)rowOff_ * alignBT_ * sizeof(IN_T);
        __gm__ IN_T *dst = reinterpret_cast<__gm__ IN_T *>(
            wsGm_.GetPhyAddr() + wsByteOff + rowOffBytes);
        GlobalTensor<IN_T> dstGm;
        dstGm.SetGlobalBuffer(dst);

        DataCopyExtParams params{
            static_cast<uint16_t>(myRows),
            static_cast<uint32_t>(alignBT_ * sizeof(IN_T)),
            0, 0, 0,
        };
        DataCopyPad(dstGm, am, params);
        amQue_.FreeTensor(am);
    }

    /*!
     * \brief Combine c1 + c3 with scale, cast to bf16 and store to o_gm.
     *        The caller frees c1 and c3.
     */
    __aicore__ inline void ComposeAndStore(LocalTensor<ACC_T> &c1,
                                           LocalTensor<ACC_T> &c3,
                                           uint32_t globalToken, uint32_t hIdx,
                                           uint32_t vStart, uint32_t curBV,
                                           uint32_t myRows) {
        const uint32_t cnt = myRows * alignV_;
        // boFp_ = scale * c1
        Muls(boFp_, c1, scale_, cnt);
        PipeBarrier<PIPE_V>();
        // c3 = scale * c3 (in-place to save a temp)
        Muls(c3, c3, scale_, cnt);
        PipeBarrier<PIPE_V>();
        // boFp_ += c3
        Add(boFp_, boFp_, c3, cnt);
        PipeBarrier<PIPE_V>();

        // Cast bf16 and store.
        LocalTensor<IN_T> outLocal = outQue_.AllocTensor<IN_T>();
        Cast(outLocal, boFp_, RoundMode::CAST_RINT, cnt);
        outQue_.EnQue<IN_T>(outLocal);
        outLocal = outQue_.DeQue<IN_T>();

        const uint64_t outBase = (uint64_t)(globalToken + rowOff_) * h_ * v_
                               + (uint64_t)hIdx * v_ + vStart;
        // For the store we have alignV per row in UB but only curBV are valid;
        // dstStride between consecutive token rows is (h_*v_ - curBV) bytes.
        DataCopyExtParams params{
            static_cast<uint16_t>(myRows),
            static_cast<uint32_t>(curBV * sizeof(IN_T)),
            static_cast<uint32_t>(((alignV_ - curBV) * sizeof(IN_T)) / BF16_PER_BLOCK),
            static_cast<uint32_t>((h_ * v_ - curBV) * sizeof(IN_T)),
            0,
        };
        DataCopyPad(outGm_[outBase], outLocal, params);
        outQue_.FreeTensor(outLocal);
    }

    // ---- Cached TilingData ------------------------------------------------
    uint32_t b_{0}, tMax_{0}, hg_{0}, h_{0}, k_{0}, v_{0}, bt_{0}, bv_{0};
    uint32_t numVTile_{0}, aicNum_{0};
    uint32_t alignK_{0}, alignV_{0}, alignBT_{0}, groupSize_{1};
    bool     useG_{false}, isVarlen_{false};
    ACC_T    scale_{1.0f};
    uint32_t wsBytesC1_{0}, wsBytesC2_{0}, wsBytesAmaskBf16_{0}, wsBytesC3_{0};
    uint32_t wsBytesPerSlot_{0}, wsBytesPerUnit_{0};

    int32_t  aivIdx_{0}, aicIdx_{0};
    uint32_t subBlock_{0};        // 0 (upper rows) or 1 (lower rows)
    uint32_t halfBT_{0};
    uint32_t rowOff_{0};          // first BT row this AIV handles
    uint32_t rowsThis_{0};        // rows owned (when curBT == bt_)
    uint32_t curNidx_{0}, curBos_{0};

    // c1 is dequeued before MASK_DONE and held until MM3_DONE so we can use
    // it in the combine step without an extra GM round-trip.
    LocalTensor<ACC_T> heldC1_;
    bool               heldC1Valid_{false};

    GlobalTensor<float>    gGm_;
    GlobalTensor<int32_t>  cuSeqlensGm_;
    GlobalTensor<int32_t>  chunkOffsetsGm_;
    GlobalTensor<IN_T>     outGm_;
    GlobalTensor<uint8_t>  wsGm_;

    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN,  1> c1Que_;
    TQue<QuePosition::VECIN,  1> c2Que_;
    TQue<QuePosition::VECIN,  1> c3Que_;
    TQue<QuePosition::VECIN,  1> gQue_;
    TQue<QuePosition::VECIN,  1> amQue_;   // bf16 cast buffer en route to ws
    TQue<QuePosition::VECOUT, 2> outQue_;  // double-buffered final store

    TBuf<TPosition::VECCALC> scratch_;
    LocalTensor<ACC_T> gExp_;
    LocalTensor<ACC_T> diff_;
    LocalTensor<ACC_T> boFp_;
    LocalTensor<ACC_T> rowAcc_;
};

}  // namespace ChunkGatedDeltaRuleO

#endif  // CHUNK_GATED_DELTA_RULE_O_KERNEL_H
