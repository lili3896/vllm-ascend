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

    // ---- Block / tile parameters ------------------------------------------
    uint32_t bv;             // V-tile per program (default 64)
    uint32_t numVTile;       // ceil(v / bv)
    uint32_t numCores;       // launch block dim (== numAiv used)

    // ---- Switches ---------------------------------------------------------
    uint32_t useG;           // 1 if g != nullptr, else 0
    uint32_t isVarlen;       // 1 if cu_seqlens != nullptr, else 0

    // ---- UB plan ----------------------------------------------------------
    uint32_t alignK;         // K aligned up to BF16 block (32B)
    uint32_t alignV;         // V aligned up to BF16 block (32B)
    uint32_t alignBT;        // BT aligned up to BF16 block (32B)

    // ---- Scalar attribute -------------------------------------------------
    float    scale;          // scale = 1/sqrt(K) by default
};
#pragma pack(pop)

}  // namespace ChunkGatedDeltaRuleO

#endif  // CHUNK_GATED_DELTA_RULE_O_TILING_DATA_H
