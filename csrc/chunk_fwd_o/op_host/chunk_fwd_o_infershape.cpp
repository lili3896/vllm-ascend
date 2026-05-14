/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_infershape.cpp
 * \brief ChunkFwdO 的 shape/dtype 推导：输出 o 与输入 v 形状/dtype 完全一致。
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "../tiling_base/error_log.h"

namespace ops {

constexpr size_t V_INDEX = 2;
constexpr size_t Q_INDEX = 0;

static ge::graphStatus InferShapeChunkFwdO(gert::InferShapeContext* context)
{
    auto* shapeV = context->GetInputShape(V_INDEX);
    auto* shapeOut = context->GetOutputShape(0);
    if (shapeV == nullptr || shapeOut == nullptr) {
        return ge::GRAPH_FAILED;
    }
    shapeOut->SetDimNum(shapeV->GetDimNum());
    for (size_t i = 0; i < shapeV->GetDimNum(); ++i) {
        shapeOut->SetDim(i, shapeV->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkFwdO(gert::InferDataTypeContext* context)
{
    auto qDtype = context->GetInputDataType(Q_INDEX);
    context->SetOutputDataType(0, qDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkFwdO)
    .InferShape(InferShapeChunkFwdO)
    .InferDataType(InferDataTypeChunkFwdO);

}  // namespace ops
