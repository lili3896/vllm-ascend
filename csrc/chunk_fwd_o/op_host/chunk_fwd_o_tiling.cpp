/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_tiling.cpp
 * \brief ChunkFwdO 算子的 host tiling 实现。
 *
 *  - 任务划分：
 *      定长：taskNum = vLoops × shapeBatch × numChunks × vNumHead；
 *      变长：taskNum = vLoops × totalChunks × vNumHead。
 *  - 工作区按 numCubeCore × PING_PONG_STAGES 划分，单 buf 仅持有一个
 *    chunk 所需的中间数据（fp32 中间 + bf16/fp16 mask 结果）。
 */

#include "register/op_def_registry.h"
#include "chunk_fwd_o_tiling.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/error_log.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(ChunkFwdO, ChunkFwdOTiling, 0);

constexpr size_t Q_INDEX            = 0;
constexpr size_t K_INDEX            = 1;
constexpr size_t V_INDEX            = 2;
constexpr size_t H_INDEX            = 3;
constexpr size_t G_INDEX            = 4;
constexpr size_t CU_SEQLENS_INDEX   = 5;
constexpr size_t CHUNK_INDICES_INDEX = 6;

constexpr size_t SCALE_ATTR_INDEX      = 0;
constexpr size_t CHUNK_SIZE_ATTR_INDEX = 1;

constexpr int64_t WORKSPACE_ALIGN = 512;
constexpr int64_t DEFAULT_BV      = 128;
constexpr int64_t MAX_BT          = ::ChunkFwdO::CHUNK_FWD_O_MAX_BT;
constexpr int64_t BT_ALIGN        = 16;

template <typename T>
static inline T CeilAlignT(T a, T align)
{
    return (a + align - 1) / align * align;
}

template <typename T>
static inline T CeilDivT(T a, T b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

void ChunkFwdOTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
}

ge::graphStatus ChunkFwdOTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(AnalyzeDtype()  != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "AnalyzeDtype failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "AnalyzeShapes failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeAttrs()  != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "AnalyzeAttrs failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::AnalyzeDtype()
{
    auto qDesc = context_->GetInputDesc(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qDesc);
    auto dt = qDesc->GetDataType();
    OP_CHECK_IF(dt != ge::DT_BF16 && dt != ge::DT_FLOAT16,
                OP_LOGE(inputParams_.opName, "q dtype must be BF16 or FP16, got %d", static_cast<int>(dt)),
                return ge::GRAPH_FAILED);
    tilingData_.set_dataType(dt == ge::DT_BF16 ? 0 : 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::AnalyzeShapes()
{
    auto qShape   = context_->GetInputShape(Q_INDEX);
    auto vShape   = context_->GetInputShape(V_INDEX);
    auto hShape   = context_->GetInputShape(H_INDEX);
    auto cuShape  = context_->GetInputShape(CU_SEQLENS_INDEX);
    auto ciShape  = context_->GetInputShape(CHUNK_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cuShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ciShape);

    const auto& qS  = qShape->GetOriginShape();
    const auto& vS  = vShape->GetOriginShape();
    const auto& hS  = hShape->GetOriginShape();
    const auto& cS  = cuShape->GetOriginShape();
    const auto& ciS = ciShape->GetOriginShape();

    int64_t B, T, Hg, H, K, V, NT;
    if (qS.GetDimNum() == 4) {
        // q: [B, T, Hg, D]
        B  = qS.GetDim(0);
        T  = qS.GetDim(1);
        Hg = qS.GetDim(2);
        K  = qS.GetDim(3);
    } else {
        OP_LOGE(inputParams_.opName, "q must be 4D [B,T,Hg,D], got rank %zu", qS.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (vS.GetDimNum() == 4) {
        // v: [B, H, T, D]
        H = vS.GetDim(1);
        V = vS.GetDim(3);
        OP_CHECK_IF(vS.GetDim(0) != B || vS.GetDim(2) != T,
                    OP_LOGE(inputParams_.opName,
                            "v must be [B,H,T,D] with B/T matching q, got v[0]=%ld v[2]=%ld qB=%ld qT=%ld",
                            vS.GetDim(0), vS.GetDim(2), B, T),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(inputParams_.opName, "v must be 4D [B,H,T,D], got rank %zu", vS.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (hS.GetDimNum() == 5) {
        // h: [B, H, NT, D, D]
        NT = hS.GetDim(2);
        OP_CHECK_IF(hS.GetDim(0) != B || hS.GetDim(1) != H ||
                        hS.GetDim(3) != K || hS.GetDim(4) != V,
                    OP_LOGE(inputParams_.opName,
                            "h must be [B,H,NT,D,D] matching q/v, got h[0]=%ld h[1]=%ld h[3]=%ld h[4]=%ld",
                            hS.GetDim(0), hS.GetDim(1), hS.GetDim(3), hS.GetDim(4)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(inputParams_.opName, "h must be 5D [B,H,NT,D,D], got rank %zu", hS.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t totalChunks = ciS.GetDimNum() >= 1 ? ciS.GetDim(0) : NT;
    int64_t tokenBatch = cS.GetDimNum() >= 1 && cS.GetDim(0) > 0 ? cS.GetDim(0) - 1 : B;
    int64_t isVariedLen = tokenBatch != B ? 1 : 0;

    tilingData_.set_shapeBatch(B);
    tilingData_.set_seqlen(T);
    tilingData_.set_kNumHead(Hg);
    tilingData_.set_vNumHead(H);
    tilingData_.set_kHeadDim(K);
    tilingData_.set_vHeadDim(V);
    tilingData_.set_numChunks(NT);
    tilingData_.set_totalChunks(totalChunks);
    tilingData_.set_tokenBatch(tokenBatch);
    tilingData_.set_isVariedLen(isVariedLen);
    tilingData_.set_hasG(context_->GetOptionalInputDesc(G_INDEX) != nullptr ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    float scale = *attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    int64_t cs  = *attrs->GetAttrPointer<int64_t>(CHUNK_SIZE_ATTR_INDEX);
    OP_CHECK_IF(cs <= 0 || cs > MAX_BT || (cs % BT_ALIGN) != 0,
                OP_LOGE(inputParams_.opName,
                        "chunk_size must be positive, %ld-aligned and no greater than %ld, got %ld",
                        BT_ALIGN, MAX_BT, cs),
                return ge::GRAPH_FAILED);
    tilingData_.set_scale(scale);
    tilingData_.set_chunkSize(cs);
    int64_t chunksPerBatch = CeilDivT<int64_t>(tilingData_.get_seqlen(), cs);
    int64_t fixedTotalChunks = tilingData_.get_shapeBatch() * chunksPerBatch;
    bool useVarLenChunks = tilingData_.get_tokenBatch() != tilingData_.get_shapeBatch() ||
                           tilingData_.get_totalChunks() != fixedTotalChunks;
    tilingData_.set_isVariedLen(useVarLenChunks ? 1 : 0);
    if (!useVarLenChunks) {
        tilingData_.set_numChunks(chunksPerBatch);
        tilingData_.set_totalChunks(fixedTotalChunks);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::PlanCorePartition()
{
    int64_t V   = tilingData_.get_vHeadDim();
    int64_t H   = tilingData_.get_vNumHead();
    int64_t B   = tilingData_.get_shapeBatch();
    int64_t NT  = tilingData_.get_numChunks();
    int64_t BV  = std::min<int64_t>(DEFAULT_BV, V);

    int64_t vLoops  = CeilDivT<int64_t>(V, BV);
    int64_t total = tilingData_.get_totalChunks();
    int64_t taskChunks = (tilingData_.get_isVariedLen() != 0) ? total : B * NT;
    int64_t taskNum = vLoops * taskChunks * H;

    int64_t aicNum = static_cast<int64_t>(compileInfo_.aicNum > 0 ? compileInfo_.aicNum : 1);
    int64_t aivNum = static_cast<int64_t>(compileInfo_.aivNum > 0 ? compileInfo_.aivNum : aicNum * 2);
    int64_t usedAic = std::min<int64_t>(taskNum, aicNum);
    if (usedAic <= 0) usedAic = 1;
    int64_t usedAiv = usedAic * (aivNum / std::max<int64_t>(aicNum, 1));

    tilingData_.set_vLoops(vLoops);
    tilingData_.set_taskNum(taskNum);
    tilingData_.set_numCubeCore(usedAic);
    tilingData_.set_numVecCore(usedAiv);
    blockDim_ = static_cast<uint32_t>(usedAic);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::PlanWorkspaces()
{
    int64_t BT = tilingData_.get_chunkSize();
    int64_t V  = tilingData_.get_vHeadDim();
    int64_t dtSize = 2;  // bf16 / fp16
    int64_t aic = tilingData_.get_numCubeCore();
    int64_t stages = ChunkFwdO::PING_PONG_STAGES;

    int64_t hSlot   = CeilAlignT<int64_t>(BT * V * sizeof(float), WORKSPACE_ALIGN);
    int64_t attnSlot = CeilAlignT<int64_t>(BT * BT * sizeof(float), WORKSPACE_ALIGN);
    int64_t vSlot   = hSlot;
    int64_t amSlot  = CeilAlignT<int64_t>(BT * BT * dtSize, WORKSPACE_ALIGN);
    int64_t maskSlot = CeilAlignT<int64_t>(BT * BT, WORKSPACE_ALIGN);

    int64_t hPerAic    = stages * hSlot;
    int64_t attnPerAic = stages * attnSlot;
    int64_t vPerAic    = stages * vSlot;
    int64_t amPerAic   = stages * amSlot;

    tilingData_.set_hWorkspaceOffset(0);
    tilingData_.set_attnWorkspaceOffset(hPerAic * aic);
    tilingData_.set_vWorkspaceOffset(tilingData_.get_attnWorkspaceOffset() + attnPerAic * aic);
    tilingData_.set_aftermaskWorkspaceOffset(tilingData_.get_vWorkspaceOffset() + vPerAic * aic);
    tilingData_.set_maskWorkspaceOffset(tilingData_.get_aftermaskWorkspaceOffset() + amPerAic * aic);

    totalWorkspaceBytes_ = tilingData_.get_maskWorkspaceOffset() + maskSlot;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::DoOpTiling()
{
    OP_CHECK_IF(PlanCorePartition() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "PlanCorePartition failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(PlanWorkspaces() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "PlanWorkspaces failed."), return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::DoLibApiTiling()
{
    tilingKey_ = (tilingData_.get_dataType() == 0) ? 0 : 1;
    return ge::GRAPH_SUCCESS;
}

uint64_t ChunkFwdOTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus ChunkFwdOTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<uint64_t>(totalWorkspaceBytes_) + systemWorkspaceBytes_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::PostTiling()
{
    context_->SetBlockDim(blockDim_);
    auto tilingDataSize = sizeof(::optiling::ChunkFwdOTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                            context_->GetRawTilingData()->GetCapacity(),
                            reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void ChunkFwdOTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "ChunkFwdO tiling: B=%ld T=%ld Hg=%ld H=%ld K=%ld V=%ld scale=%f BT=%ld "
            "varlen=%ld tokenBatch=%ld NT=%ld totalChunks=%ld vLoops=%ld taskNum=%ld "
            "AIC=%ld AIV=%ld dtype=%ld ws[h=%ld attn=%ld v=%ld am=%ld mask=%ld]",
            tilingData_.get_shapeBatch(),
            tilingData_.get_seqlen(),
            tilingData_.get_kNumHead(),
            tilingData_.get_vNumHead(),
            tilingData_.get_kHeadDim(),
            tilingData_.get_vHeadDim(),
            tilingData_.get_scale(),
            tilingData_.get_chunkSize(),
            tilingData_.get_isVariedLen(),
            tilingData_.get_tokenBatch(),
            tilingData_.get_numChunks(),
            tilingData_.get_totalChunks(),
            tilingData_.get_vLoops(),
            tilingData_.get_taskNum(),
            tilingData_.get_numCubeCore(),
            tilingData_.get_numVecCore(),
            tilingData_.get_dataType(),
            tilingData_.get_hWorkspaceOffset(),
            tilingData_.get_attnWorkspaceOffset(),
            tilingData_.get_vWorkspaceOffset(),
            tilingData_.get_aftermaskWorkspaceOffset(),
            tilingData_.get_maskWorkspaceOffset());
}

static ge::graphStatus ChunkFwdOTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForChunkFwdO(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* compileInfoPtr = context->GetCompiledInfo<ChunkFwdOCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkFwdO)
    .Tiling(ChunkFwdOTilingFunc)
    .TilingParse<ChunkFwdOCompileInfo>(TilingPrepareForChunkFwdO);

}  // namespace optiling
