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
 * \brief Host tiling for ChunkGatedDeltaRuleO (AIC + AIV mix mode).
 *
 * Tiling responsibilities:
 *   1. Validate inputs (dtype, rank, GQA constraints).
 *   2. Pick block_dim equal to min(units, aicNum) where
 *      units = N * H * ceil(V / B_V).
 *   3. Build three TCubeTiling structures via matmul_tiling::MultiCoreMatmulTiling
 *      for the three MMAs (Q@H, Q@K^T, A_masked @ V).
 *   4. Plan the per-work-unit ping-pong workspace and total workspace bytes.
 */

#include "chunk_gated_delta_rule_o_tiling.h"

#include <array>
#include <algorithm>
#include "tiling/tiling_api.h"
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
constexpr uint32_t CUBE_BLOCK = 16;     // 16x16x16 fundamental cube tile
constexpr uint32_t FP32_BLOCK = 8;
constexpr uint32_t BF16_BLOCK = 16;

template <typename T>
T CeilDiv(T a, T b) { return (b == 0) ? T(0) : T((a + b - 1) / b); }
template <typename T>
T CeilAlign(T a, T b) { return CeilDiv(a, b) * b; }

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
    auto platform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,    compileInfo_.ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,    compileInfo_.l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,  compileInfo_.l0aSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,  compileInfo_.l0bSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,  compileInfo_.l0cSize);
    compileInfo_.aivNum = platform.GetCoreNumAiv();
    compileInfo_.aicNum = platform.GetCoreNumAic();
    compileInfo_.socVersion = platform.GetSocVersion();
    if (compileInfo_.aicNum == 0 || compileInfo_.aivNum == 0) {
        OP_LOGE(context_->GetNodeName(), "aicNum/aivNum is zero");
        return;
    }
    tilingData_.aicNum = static_cast<uint32_t>(compileInfo_.aicNum);
    tilingData_.aivNum = static_cast<uint32_t>(compileInfo_.aivNum);
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

    // Choose B_V as the largest power-of-2 tile that divides V and is in
    // [16, 128]. For typical Qwen3-Next (V=128) we get bv=128 (1 V-tile);
    // for V=256 we get bv=128 (2 V-tiles). Falls back to 64 otherwise.
    uint32_t bv = 128;
    while (bv > 16 && tilingData_.v % bv != 0) {
        bv >>= 1;
    }
    if (bv < 16) bv = DEFAULT_BV;
    tilingData_.bv = bv;
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
    PlanBlockDim();
    PlanWorkspaceLayout();
    return ge::GRAPH_SUCCESS;
}

void ChunkGatedDeltaRuleOTiling::PlanBlockDim() {
    tilingData_.alignK  = CeilAlign(tilingData_.k,  CUBE_BLOCK);
    tilingData_.alignV  = CeilAlign(tilingData_.bv, CUBE_BLOCK);
    tilingData_.alignBT = CeilAlign(tilingData_.bt, CUBE_BLOCK);

    const uint64_t units = static_cast<uint64_t>(tilingData_.b)
                         * tilingData_.h
                         * tilingData_.numVTile;
    uint64_t aicNum = (compileInfo_.aicNum > 0) ? compileInfo_.aicNum : 1;
    if (units < aicNum) aicNum = (units == 0) ? 1 : units;
    tilingData_.aicNum = static_cast<uint32_t>(aicNum);
    tilingData_.aivNum = static_cast<uint32_t>(aicNum * 2);  // 1 AIC : 2 AIVs
}

void ChunkGatedDeltaRuleOTiling::PlanWorkspaceLayout() {
    tilingData_.wsBytesC1        = tilingData_.bt * tilingData_.alignV * sizeof(float);
    tilingData_.wsBytesC2        = tilingData_.bt * tilingData_.alignBT * sizeof(float);
    tilingData_.wsBytesAmaskBf16 = tilingData_.bt * tilingData_.alignBT * sizeof(uint16_t);
    tilingData_.wsBytesC3        = tilingData_.bt * tilingData_.alignV * sizeof(float);
    tilingData_.wsBytesPerSlot   = tilingData_.wsBytesC1 + tilingData_.wsBytesC2
                                 + tilingData_.wsBytesAmaskBf16 + tilingData_.wsBytesC3;
    tilingData_.wsBytesPerUnit   = 2u * tilingData_.wsBytesPerSlot;  // ping-pong
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::DoLibApiTiling() {
    return ConfigMatmulTilings();
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::ConfigMatmulTilings() {
    // Build platform info passthrough for the matmul tiling helper.
    matmul_tiling::PlatformInfo platformInfo;
    platformInfo.socVersion = compileInfo_.socVersion;
    platformInfo.l1Size     = compileInfo_.l1Size;
    platformInfo.l0CSize    = compileInfo_.l0cSize;
    platformInfo.ubSize     = compileInfo_.ubSize;
    platformInfo.l0ASize    = compileInfo_.l0aSize;
    platformInfo.l0BSize    = compileInfo_.l0bSize;

    const auto bf16 = matmul_tiling::DataType::DT_BFLOAT16;
    const auto fp32 = matmul_tiling::DataType::DT_FLOAT;

    // -------- MM1 : Q [BT, K] x H [K, BV] -> c1 [BT, BV] (fp32 accum) ----
    {
        matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
        mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16, false);
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16, false);
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, fp32);
        mm.SetOrgShape(tilingData_.bt, tilingData_.bv, tilingData_.k);
        mm.SetShape(tilingData_.bt, tilingData_.bv, tilingData_.k);
        mm.SetFixSplit(tilingData_.bt, tilingData_.bv, tilingData_.k);
        mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0cSize, compileInfo_.ubSize);
        if (mm.GetTiling(tilingData_.mm1Tiling) == -1) {
            OP_LOGE(context_->GetNodeName(), "mm1 tiling failed.");
            return ge::GRAPH_FAILED;
        }
    }

    // -------- MM2 : Q [BT, K] x K^T [K, BT] -> c2 [BT, BT] ---------------
    {
        matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
        mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16, false);
        // transposeB=true so the cube reads K row-major and treats it as K^T
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16, true);
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, fp32);
        mm.SetOrgShape(tilingData_.bt, tilingData_.bt, tilingData_.k);
        mm.SetShape(tilingData_.bt, tilingData_.bt, tilingData_.k);
        mm.SetFixSplit(tilingData_.bt, tilingData_.bt, tilingData_.k);
        mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0cSize, compileInfo_.ubSize);
        if (mm.GetTiling(tilingData_.mm2Tiling) == -1) {
            OP_LOGE(context_->GetNodeName(), "mm2 tiling failed.");
            return ge::GRAPH_FAILED;
        }
    }

    // -------- MM3 : A_masked [BT, BT] x V [BT, BV] -> c3 [BT, BV] -------
    {
        matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
        mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16, false);
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16, false);
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, fp32);
        mm.SetOrgShape(tilingData_.bt, tilingData_.bv, tilingData_.bt);
        mm.SetShape(tilingData_.bt, tilingData_.bv, tilingData_.bt);
        mm.SetFixSplit(tilingData_.bt, tilingData_.bv, tilingData_.bt);
        mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0cSize, compileInfo_.ubSize);
        if (mm.GetTiling(tilingData_.mm3Tiling) == -1) {
            OP_LOGE(context_->GetNodeName(), "mm3 tiling failed.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t ChunkGatedDeltaRuleOTiling::GetTilingKey() const {
    // bit 0: USE_G, bit 1: IS_VARLEN; reserved for future template specialisations
    uint64_t key = 0;
    if (tilingData_.useG)     key |= 1ULL;
    if (tilingData_.isVarlen) key |= (1ULL << 1);
    return key;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::GetWorkspaceSize() {
    // System-side workspace + per-work-unit ping-pong slots.
    constexpr uint64_t SYS_WORKSPACE = 16ULL * 1024 * 1024;
    auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t libApi = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    const uint64_t units = static_cast<uint64_t>(tilingData_.b)
                         * tilingData_.h
                         * tilingData_.numVTile;
    workspaceSize_ = SYS_WORKSPACE + libApi
                   + units * tilingData_.wsBytesPerUnit;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleOTiling::PostTiling() {
    // For mix mode, the runtime expects block_dim measured in AIC count.
    context_->SetBlockDim(tilingData_.aicNum);

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
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void ChunkGatedDeltaRuleOTiling::PrintTilingData() const {
    OP_LOGD(context_->GetNodeName(),
            "tiling: b=%u t=%u hg=%u h=%u k=%u v=%u bt=%u bv=%u "
            "numVTile=%u aicNum=%u aivNum=%u useG=%u isVarlen=%u "
            "alignK=%u alignV=%u alignBT=%u scale=%f wsPerUnit=%u",
            tilingData_.b, tilingData_.t, tilingData_.hg, tilingData_.h,
            tilingData_.k, tilingData_.v, tilingData_.bt, tilingData_.bv,
            tilingData_.numVTile, tilingData_.aicNum, tilingData_.aivNum,
            tilingData_.useG, tilingData_.isVarlen,
            tilingData_.alignK, tilingData_.alignV, tilingData_.alignBT,
            tilingData_.scale, tilingData_.wsBytesPerUnit);
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
