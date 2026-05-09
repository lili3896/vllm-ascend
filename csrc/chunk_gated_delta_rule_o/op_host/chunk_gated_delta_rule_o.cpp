/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chunk_gated_delta_rule_o.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleO);

const aclTensor *ChunkGatedDeltaRuleO(const aclTensor *query,
                                      const aclTensor *key,
                                      const aclTensor *value,
                                      const aclTensor *h,
                                      const aclTensor *g,
                                      const aclTensor *cuSeqlens,
                                      const aclTensor *chunkOffsets,
                                      float scaleValue, int64_t chunkSize,
                                      aclOpExecutor *executor) {
    L0_DFX(ChunkGatedDeltaRuleO, query, key, value, h, g, cuSeqlens,
           chunkOffsets, scaleValue, chunkSize);

    DataType outType = DataType::DT_BF16;
    Format   format  = Format::FORMAT_ND;
    auto out = executor->AllocTensor(outType, format, format);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."),
             return nullptr);

    auto ret = INFER_SHAPE(ChunkGatedDeltaRuleO,
                           OP_INPUT(query, key, value, h, g, cuSeqlens, chunkOffsets),
                           OP_OUTPUT(out),
                           OP_ATTR(scaleValue, chunkSize));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr,
                        "ChunkGatedDeltaRuleO InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleO,
        OP_INPUT(query, key, value, h, g, cuSeqlens, chunkOffsets),
        OP_OUTPUT(out),
        OP_ATTR(scaleValue, chunkSize));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr,
        "ChunkGatedDeltaRuleO ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}

}  // namespace l0op
