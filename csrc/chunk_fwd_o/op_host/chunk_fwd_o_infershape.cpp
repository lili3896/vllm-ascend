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
 * \brief ChunkFwdO 的 shape/dtype 推导：输出 o 为 [B,T,H,D]，dtype 与 q 一致。
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "../tiling_base/error_log.h"

namespace ops {

constexpr size_t V_INDEX = 2;
constexpr size_t Q_INDEX = 0;
constexpr size_t RANK_4D = 4;

static ge::graphStatus InferShapeChunkFwdO(gert::InferShapeContext* context)
{
    auto* shapeQ = context->GetInputShape(Q_INDEX);
    auto* shapeV = context->GetInputShape(V_INDEX);
    auto* shapeOut = context->GetOutputShape(0);
    if (shapeQ == nullptr || shapeV == nullptr || shapeOut == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (shapeQ->GetDimNum() != RANK_4D || shapeV->GetDimNum() != RANK_4D) {
        return ge::GRAPH_FAILED;
    }
    // q: [B,T,Hg,D], v: [B,H,T,D] -> o: [B,T,H,D]
    shapeOut->SetDimNum(RANK_4D);
    shapeOut->SetDim(0, shapeQ->GetDim(0));
    shapeOut->SetDim(1, shapeQ->GetDim(1));
    shapeOut->SetDim(2, shapeV->GetDim(1));
    shapeOut->SetDim(3, shapeV->GetDim(3));
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
