/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o_tiling.cpp
 * \brief Host tiling for ChunkGatedDeltaRuleO.
 *
 * Steps (TilingBaseClass framework):
 *   1. GetPlatformInfo (no-op; queried already in InitCompileInfo)
 *   2. GetShapeAttrsInfo:
 *        - Analyse input dtypes (must be bf16 / fp32 / int32)
 *        - Analyse input shapes and fill TilingData scalars
 *        - Read scale_value attr and chunk_size attr
 *        - Detect optional g / cu_seqlens / chunk_offsets presence
 *   3. DoOpTiling:
 *        - PlanBlockDim: pick block dim = min(aivNum, N*H*numVTile)
 *   4. DoLibApiTiling: trivial (we don't use Matmul tiling API in this AIV-only
 *      kernel; reserved for the future cube path)
 *   5. GetTilingKey: a single key today; combine USE_G/IS_VARLEN bits when the
 *      cube path lands.
 *   6. GetWorkspaceSize: 16MiB system workspace as required by the runtime.
 *   7. PostTiling: serialise TilingData blob and set block dim.
 */

#include "chunk_gated_delta_rule_o_tiling.h"

#include <array>
#include "tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(ChunkGatedDeltaRuleO, ChunkGatedDeltaRuleOTiling, 0);

namespace {

constexpr size_t QUERY_INDEX        = 0;
constexpr size_t KEY_INDEX          = 1;
constexpr size_t VALUE_INDEX        = 2;
constexpr size_t H_INDEX            = 3;
constexpr size_t G_INDEX            = 4;
constexpr size_t CU_SEQLENS_INDEX   = 5;
constexpr size_t CHUNK_OFFSETS_INDEX = 6;

constexpr size_t QK_DIM_NUM = 4;     // [B, T, Hg, K]
constexpr size_t V_DIM_NUM  = 4;     // [B, T, H,  V]
constexpr size_t H_DIM_NUM  = 5;     // [B, NT, H, K, V]

constexpr uint32_t DEFAULT_BT = 64;
constexpr uint32_t DEFAULT_BV = 64;
constexpr uint32_t MAX_K_HEAD_DIM = 256;
constexpr uint32_t MAX_V_HEAD_DIM = 256;

constexpr uint32_t BF16_BLOCK_ELEMS = 16;  // 32B / 2B
constexpr uint32_t FP32_BLOCK_ELEMS = 8;   // 32B / 4B

template <typename T>
T CeilDiv(T a, T b) {
    return (b == 0) ? T(0) : T((a + b - 1) / b);
}

template <typename T>
T CeilAlign(T a, T b) {
    return CeilDiv(a, b) * b;
}

}  // namespace

// ---------------------------------------------------------------------------
// Platform / Compile info
// ---------------------------------------------------------------------------

void ChunkGatedDeltaRuleOTiling::InitCompileInfo() {
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &platform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                            compileInfo_.ubSize);
    compileInfo_.aivNum = platform.GetCoreNumAiv();
    if (compileInfo_.aivNum == 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum is zero");
        return;
    }
    tilingData_.numCores = static_cast<uint32_t>(compileInfo_.aivNum);
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::GetPlatformInfo() {
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// Shape & attribute parsing
// ---------------------------------------------------------------------------

ge::graphStatus ChunkGatedDeltaRuleOTiling::GetShapeAttrsInfo() {
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid scale_value."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetChunkSize() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid chunk_size."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DetectOptionalInputs() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Optional inputs invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::CheckContext() {
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(H_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(H_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::AnalyzeDtype() {
    auto qDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto kDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto vDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    auto hDtype = context_->GetInputDesc(H_INDEX)->GetDataType();
    OP_CHECK_IF(qDtype != ge::DT_BF16 || kDtype != ge::DT_BF16
                    || vDtype != ge::DT_BF16 || hDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(),
                        "q,k,v,h must all be bfloat16."),
                return ge::GRAPH_FAILED);
    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
        auto gDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gDtype != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "g must be float32."),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr) {
        auto cuDtype = context_->GetOptionalInputDesc(CU_SEQLENS_INDEX)->GetDataType();
        OP_CHECK_IF(cuDtype != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(),
                            "cu_seqlens must be int32."),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(CHUNK_OFFSETS_INDEX) != nullptr) {
        auto coDtype = context_->GetOptionalInputDesc(CHUNK_OFFSETS_INDEX)
                           ->GetDataType();
        OP_CHECK_IF(coDtype != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(),
                            "chunk_offsets must be int32."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::AnalyzeShapes() {
    const auto &qShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &vShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &hShape = context_->GetInputShape(H_INDEX)->GetOriginShape();

    OP_CHECK_IF(qShape.GetDimNum() != QK_DIM_NUM,
                OP_LOGE(context_->GetNodeName(),
                        "query rank must be 4 [B,T,Hg,K]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDimNum() != V_DIM_NUM,
                OP_LOGE(context_->GetNodeName(),
                        "value rank must be 4 [B,T,H,V]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hShape.GetDimNum() != H_DIM_NUM,
                OP_LOGE(context_->GetNodeName(),
                        "h rank must be 5 [B,NT,H,K,V]."),
                return ge::GRAPH_FAILED);

    tilingData_.b  = static_cast<uint32_t>(qShape.GetDim(0));
    tilingData_.t  = static_cast<uint32_t>(qShape.GetDim(1));
    tilingData_.hg = static_cast<uint32_t>(qShape.GetDim(2));
    tilingData_.k  = static_cast<uint32_t>(qShape.GetDim(3));
    tilingData_.h  = static_cast<uint32_t>(vShape.GetDim(2));
    tilingData_.v  = static_cast<uint32_t>(vShape.GetDim(3));

    OP_CHECK_IF(tilingData_.h % tilingData_.hg != 0,
                OP_LOGE(context_->GetNodeName(),
                        "H must be a multiple of Hg, got H=%u Hg=%u",
                        tilingData_.h, tilingData_.hg),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.k > MAX_K_HEAD_DIM
                    || tilingData_.v > MAX_V_HEAD_DIM,
                OP_LOGE(context_->GetNodeName(),
                        "K/V head dim too large (K=%u, V=%u, max=%u/%u)",
                        tilingData_.k, tilingData_.v,
                        MAX_K_HEAD_DIM, MAX_V_HEAD_DIM),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::GetScale() {
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float *scalePtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scalePtr);
    tilingData_.scale = *scalePtr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::GetChunkSize() {
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *btPtr = attrs->GetAttrPointer<int64_t>(1);
    int64_t bt = (btPtr != nullptr) ? *btPtr : DEFAULT_BT;
    OP_CHECK_IF(bt <= 0 || bt > 128,
                OP_LOGE(context_->GetNodeName(),
                        "chunk_size must be in (0, 128], got %ld", bt),
                return ge::GRAPH_FAILED);
    tilingData_.bt = static_cast<uint32_t>(bt);
    tilingData_.bv = DEFAULT_BV;
    tilingData_.numVTile = CeilDiv(tilingData_.v, tilingData_.bv);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::DetectOptionalInputs() {
    tilingData_.useG     = (context_->GetOptionalInputDesc(G_INDEX) != nullptr) ? 1 : 0;
    tilingData_.isVarlen = (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// Op tiling proper
// ---------------------------------------------------------------------------

ge::graphStatus ChunkGatedDeltaRuleOTiling::DoOpTiling() {
    return PlanBlockDim();
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::PlanBlockDim() {
    tilingData_.alignK  = CeilAlign(tilingData_.k, BF16_BLOCK_ELEMS);
    tilingData_.alignV  = CeilAlign(tilingData_.bv, BF16_BLOCK_ELEMS);
    tilingData_.alignBT = CeilAlign(tilingData_.bt, FP32_BLOCK_ELEMS);

    const uint64_t units = static_cast<uint64_t>(tilingData_.b)
                         * tilingData_.h
                         * tilingData_.numVTile;
    uint64_t cores = (compileInfo_.aivNum > 0) ? compileInfo_.aivNum : 1;
    if (units < cores) {
        cores = units == 0 ? 1 : units;
    }
    tilingData_.numCores = static_cast<uint32_t>(cores);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::DoLibApiTiling() {
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t ChunkGatedDeltaRuleOTiling::GetTilingKey() const {
    return tilingKey_;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::GetWorkspaceSize() {
    constexpr uint64_t SYS_WORKSPACE = 16ULL * 1024 * 1024;
    workspaceSize_ = SYS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::PostTiling() {
    context_->SetBlockDim(tilingData_.numCores);
    auto blob = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, blob);
    const auto sz = sizeof(ChunkGatedDeltaRuleOTilingData);
    errno_t ret = memcpy_s(blob->GetData(), blob->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sz);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(),
                "memcpy_s tilingData failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    blob->SetDataSize(sz);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void ChunkGatedDeltaRuleOTiling::PrintTilingData() const {
    OP_LOGD(context_->GetNodeName(),
            "tiling: b=%u t=%u hg=%u h=%u k=%u v=%u bt=%u bv=%u "
            "numVTile=%u numCores=%u useG=%u isVarlen=%u "
            "alignK=%u alignV=%u alignBT=%u scale=%f",
            tilingData_.b, tilingData_.t, tilingData_.hg, tilingData_.h,
            tilingData_.k, tilingData_.v, tilingData_.bt, tilingData_.bv,
            tilingData_.numVTile, tilingData_.numCores,
            tilingData_.useG, tilingData_.isVarlen,
            tilingData_.alignK, tilingData_.alignV, tilingData_.alignBT,
            tilingData_.scale);
}

// ---------------------------------------------------------------------------
// Tiling registration glue
// ---------------------------------------------------------------------------

static ge::graphStatus ChunkGatedDeltaRuleOTilingFunc(gert::TilingContext *context) {
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleO", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance()
        .DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForChunkGatedDeltaRuleO(
    gert::TilingParseContext *context) {
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleO", "parse context is null"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<ChunkGatedDeltaRuleOCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleO)
    .Tiling(ChunkGatedDeltaRuleOTilingFunc)
    .TilingParse<ChunkGatedDeltaRuleOCompileInfo>(
        TilingPrepareForChunkGatedDeltaRuleO);

}  // namespace optiling
