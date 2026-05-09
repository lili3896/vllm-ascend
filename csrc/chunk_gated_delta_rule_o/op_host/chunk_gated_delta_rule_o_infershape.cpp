/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o_infershape.cpp
 * \brief InferShape and InferDataType for ChunkGatedDeltaRuleO.
 *
 * Output `out` matches `value` in shape and bf16 dtype.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "error_log.h"

using namespace gert;

namespace ops {

namespace {
constexpr size_t VALUE_INDEX = 2;
constexpr size_t OUT_INDEX   = 0;
constexpr size_t VALUE_DIM   = 4;
}  // namespace

static ge::graphStatus InferShapeChunkGatedDeltaRuleO(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("ChunkGatedDeltaRuleO", "inference context is null");
        return ge::GRAPH_FAILED;
    }
    auto opName = context->GetNodeName();
    auto valueShape = context->GetInputShape(VALUE_INDEX);
    auto outShape   = context->GetOutputShape(OUT_INDEX);
    if (valueShape == nullptr || outShape == nullptr) {
        OP_LOGE(opName, "[InferShape] value or out shape is null");
        return ge::GRAPH_FAILED;
    }
    const size_t dimNum = valueShape->GetDimNum();
    outShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        outShape->SetDim(i, valueShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRuleO(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleO)
    .InferShape(InferShapeChunkGatedDeltaRuleO)
    .InferDataType(InferDataTypeChunkGatedDeltaRuleO);

}  // namespace ops
