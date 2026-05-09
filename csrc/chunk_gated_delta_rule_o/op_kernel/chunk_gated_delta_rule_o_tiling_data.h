/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o_tiling_data.h
 * \brief TilingData definition shared between host tiling and device kernel for
 *        the ChunkGatedDeltaRuleO operator.
 *
 * The op realizes the chunk-forward output of Gated Delta-Rule (a.k.a. the
 * Triton kernel implemented in
 *   vllm_ascend/ops/triton/fla/chunk_o.py::chunk_fwd_kernel_o
 * ).
 *
 * This implementation runs in AIC+AIV mix mode (KERNEL_TYPE_MIX_AIC_1_2):
 *   - AIC issues three Cube MMAs per chunk via the high-level Matmul template
 *     (MM1: Q_t @ H_t,  MM2: Q_t @ K_t^T,  MM3: b_A_masked @ V_t).
 *   - AIV applies gating (exp, safe_exp), causal mask, scale composition,
 *     bf16 cast and DataCopyPad store.
 *   - AIC <-> AIV synchronisation via CrossCoreSetFlag / CrossCoreWaitFlag
 *     (events MM12_DONE, MASK_DONE, MM3_DONE).
 *   - Per work-unit ping-pong workspace lets MM1+MM2 of chunk t+1 overlap
 *     with the AIV vector compute of chunk t.
 *
 * Mathematical contract (per chunk t of head h of sequence n, V-tile v):
 *   O_t = scale * exp(g_t) ⊙ Q_t H_t
 *       + scale * ( M ⊙ exp(g_i - g_j) ⊙ Q_t K_t^T ) V_t
 *   where M is the within-chunk causal lower-triangular mask.
 */

#ifndef CHUNK_GATED_DELTA_RULE_O_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_O_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRuleO {

#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleOTilingData {
    // ---- Logical shape parameters -----------------------------------------
    uint32_t b;              // number of sequences (varlen) or batch (fixed-len)
    uint32_t t;              // padded T dimension (max seq length when varlen)
    uint32_t hg;             // number of KV heads (Hg in chunk_o.py)
    uint32_t h;              // number of output heads (>= hg, GQA)
    uint32_t k;              // head dim of Q/K
    uint32_t v;              // head dim of V
    uint32_t bt;             // chunk size BT (default 64)

    // ---- Cube tile parameters --------------------------------------------
    uint32_t bv;             // V-tile per program (default 64)
    uint32_t numVTile;       // ceil(v / bv)
    uint32_t aicNum;         // launch block dim measured in AICs (mix 1:2)
    uint32_t aivNum;         // = 2 * aicNum

    // ---- Switches ---------------------------------------------------------
    uint32_t useG;           // 1 if g != nullptr, else 0
    uint32_t isVarlen;       // 1 if cu_seqlens != nullptr, else 0

    // ---- UB plan ----------------------------------------------------------
    uint32_t alignK;         // K aligned up to 16 (cube basic block)
    uint32_t alignV;         // bv aligned up to 16
    uint32_t alignBT;        // BT aligned up to 16

    // ---- Scalar attribute -------------------------------------------------
    float    scale;          // scale = 1/sqrt(K) by default

    // ---- Per work-unit workspace stride (in BYTES, fp32 for c1/c3 buffers,
    //      fp32 for c2 buffer, bf16 for masked b_A buffer) ------------------
    // workspace layout per work unit, two ping-pong slots:
    //   slot[i] = { c1 (BT*alignV*4) | c2 (BT*alignBT*4) |
    //               b_A_bf16 (BT*alignBT*2) | c3 (BT*alignV*4) }
    uint32_t wsBytesC1;          // BT * alignV * 4
    uint32_t wsBytesC2;          // BT * alignBT * 4
    uint32_t wsBytesAmaskBf16;   // BT * alignBT * 2
    uint32_t wsBytesC3;          // BT * alignV * 4
    uint32_t wsBytesPerSlot;     // sum of the four above
    uint32_t wsBytesPerUnit;     // 2 * wsBytesPerSlot  (ping-pong)

    // ---- Cube tilings for the three MMAs (filled by matmul_tiling) -------
    TCubeTiling mm1Tiling;       // Q_t @ H_t      -> c1
    TCubeTiling mm2Tiling;       // Q_t @ K_t^T    -> c2
    TCubeTiling mm3Tiling;       // b_A_masked @ V -> c3
};
#pragma pack(pop)

}  // namespace ChunkGatedDeltaRuleO

#endif  // CHUNK_GATED_DELTA_RULE_O_TILING_DATA_H
