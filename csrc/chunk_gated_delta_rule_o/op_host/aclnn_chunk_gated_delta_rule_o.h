/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_O_H
#define OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_O_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Two-phase aclnn API for the ChunkGatedDeltaRuleO operator.
 *
 * Phase 1: aclnnChunkGatedDeltaRuleOGetWorkspaceSize
 *   - Computes workspace bytes and prepares the executor.
 *
 * Phase 2: aclnnChunkGatedDeltaRuleO
 *   - Launches the previously prepared executor on the given stream.
 *
 * Tensor contracts (matching vllm_ascend/ops/triton/fla/chunk_o.py):
 *   query : [B, T, Hg, K]   bfloat16
 *   key   : [B, T, Hg, K]   bfloat16
 *   value : [B, T, H,  V]   bfloat16
 *   h     : [B, NT, H, K, V] bfloat16
 *   g            : [B, T, H]    float32      (optional, may be nullptr)
 *   cuSeqlens    : [N+1]        int32        (optional)
 *   chunkOffsets : [N+1]        int32        (optional, prebuilt)
 *   out   : [B, T, H, V]   bfloat16          (output, allocated by caller)
 *
 * Attribute:
 *   scaleValue : float, default 1/sqrt(K) on caller side.
 *   chunkSize  : int64_t, default 64.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleOGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *h, const aclTensor *g, const aclTensor *cuSeqlens,
    const aclTensor *chunkOffsets, float scaleValue, int64_t chunkSize,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleO(void *workspace, uint64_t workspaceSize,
                                      aclOpExecutor *executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_O_H
