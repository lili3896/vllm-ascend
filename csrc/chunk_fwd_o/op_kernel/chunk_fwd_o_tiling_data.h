/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_tiling_data.h
 * \brief Tiling data structure shared between host and device for ChunkFwdO.
 *
 *  ChunkFwdO computes the chunk-wise output of the FLA (Flash-Linear Attention)
 *  forward path.  The numerical algorithm is identical to vllm_ascend's triton
 *  implementation in vllm_ascend/ops/triton/fla/chunk_o.py:
 *
 *      o = (Q @ H) * exp(g) * scale + (causal_mask(Q @ K) * exp(g_i - g_j)) @ V * scale
 *
 *  See docs/source/developer_guide/feature_guide/ascendc_chunk_fwd_o for the
 *  full design document.
 */

#ifndef CHUNK_FWD_O_TILING_DATA_H
#define CHUNK_FWD_O_TILING_DATA_H

#include <cstdint>

namespace ChunkFwdO {

// Algorithmic chunk size and block sizes for tiling the attention math.
constexpr uint32_t CHUNK_FWD_O_BT = 64;          // chunk size BT (tokens per chunk)
constexpr uint32_t CHUNK_FWD_O_BK = 128;         // K-direction inner block
constexpr uint32_t CHUNK_FWD_O_BV = 128;         // V-direction inner block

// dataType encoding (passed via tiling)
constexpr uint32_t CHUNK_FWD_O_DTYPE_BF16 = 0;
constexpr uint32_t CHUNK_FWD_O_DTYPE_FP16 = 1;

#pragma pack(push, 8)
struct alignas(8) ChunkFwdOTilingData {
    int64_t shapeBatch;                 // N (number of sequences)
    int64_t seqlen;                     // T (max sequence length, fixed-shape mode)
    int64_t kNumHead;                   // Hg (KV groups in q/k)
    int64_t vNumHead;                   // H  (full heads in v/o)
    int64_t kHeadDim;                   // K
    int64_t vHeadDim;                   // V
    float   scale;                      // attention scale
    int64_t chunkSize;                  // BT
    int64_t isVariedLen;                // 0 = fixed length, 1 = use cu_seqlens
    int64_t tokenBatch;                 // total tokens (sum of seqlens) in varlen mode, otherwise N*T
    int64_t dataType;                   // 0:BF16 1:FP16
    int64_t totalChunks;                // sum(ceil(seqlen[i]/BT)), drives h workspace base
    int64_t bvNum;                      // ceil(V / BV)
    int64_t bkNum;                      // ceil(K / BK)
    int64_t numCubeCore;                // AIC core count selected
    int64_t numVecCore;                 // AIV core count selected
    int64_t hasG;                       // 1 if g present
    // Workspace base offsets (in bytes from workspace pointer).
    int64_t vWorkspaceOffset;           // (A_masked @ V) intermediate, fp32
    int64_t hWorkspaceOffset;           // (Q @ H)        intermediate, fp32
    int64_t attnWorkspaceOffset;        // (Q @ K) attention scores,    fp32
    int64_t aftermaskWorkspaceOffset;   // attention scores after causal mask + g rescale, dtype
    int64_t maskWorkspaceOffset;        // optional causal mask cached on device, fp32
};
#pragma pack(pop)

}  // namespace ChunkFwdO

#endif  // CHUNK_FWD_O_TILING_DATA_H
