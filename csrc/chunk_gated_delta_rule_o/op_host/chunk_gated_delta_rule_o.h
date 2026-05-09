/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTA_NPU_OP_API_LEVEL0_CHUNK_GATED_DELTA_RULE_O_H
#define PTA_NPU_OP_API_LEVEL0_CHUNK_GATED_DELTA_RULE_O_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

/*!
 * \brief L0 dispatch for ChunkGatedDeltaRuleO. Allocates the output tensor,
 *        runs InferShape and registers the AICORE launch in the executor.
 */
const aclTensor *ChunkGatedDeltaRuleO(const aclTensor *query,
                                      const aclTensor *key,
                                      const aclTensor *value,
                                      const aclTensor *h,
                                      const aclTensor *g,
                                      const aclTensor *cuSeqlens,
                                      const aclTensor *chunkOffsets,
                                      float scaleValue, int64_t chunkSize,
                                      aclOpExecutor *executor);

}  // namespace l0op

#endif  // PTA_NPU_OP_API_LEVEL0_CHUNK_GATED_DELTA_RULE_O_H
