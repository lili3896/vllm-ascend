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
 * \brief ChunkFwdO 算子的 host 端 tiling 实现：解析输入 shape/dtype/属性，
 *        进行核间任务划分以及 workspace 规划。
 */

#include "register/op_def_registry.h"
#include "chunk_fwd_o_tiling.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/error_log.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(ChunkFwdO, ChunkFwdOTiling, 0);

constexpr size_t Q_INDEX             = 0;
constexpr size_t K_INDEX             = 1;
constexpr size_t V_INDEX             = 2;
constexpr size_t H_INDEX             = 3;
constexpr size_t G_INDEX             = 4;
constexpr size_t CU_SEQLENS_INDEX    = 5;
constexpr size_t CHUNK_OFFSETS_INDEX = 6;

constexpr size_t SCALE_ATTR_INDEX      = 0;
constexpr size_t CHUNK_SIZE_ATTR_INDEX = 1;

constexpr int64_t WORKSPACE_ALIGN = 512;

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
    auto qShape = context_->GetInputShape(Q_INDEX);
    auto vShape = context_->GetInputShape(V_INDEX);
    auto hShape = context_->GetInputShape(H_INDEX);
    auto cuShape = context_->GetInputShape(CU_SEQLENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cuShape);

    const auto& qS = qShape->GetOriginShape();
    const auto& vS = vShape->GetOriginShape();
    const auto& hS = hShape->GetOriginShape();
    const auto& cS = cuShape->GetOriginShape();

    int64_t numTokens, Hg, K, H, V;
    if (qS.GetDimNum() == 4) {  // (B, T, Hg, K) 固定 shape 模式
        int64_t B = qS.GetDim(0);
        int64_t T = qS.GetDim(1);
        Hg = qS.GetDim(2);
        K  = qS.GetDim(3);
        H  = vS.GetDim(2);
        V  = vS.GetDim(3);
        numTokens = B * T;
        tilingData_.set_isVariedLen(0);
        tilingData_.set_seqlen(T);
        tilingData_.set_shapeBatch(B);
    } else if (qS.GetDimNum() == 3) {  // (Ttotal, Hg, K) varlen 模式
        numTokens = qS.GetDim(0);
        Hg = qS.GetDim(1);
        K  = qS.GetDim(2);
        H  = vS.GetDim(1);
        V  = vS.GetDim(2);
        int64_t N = cS.GetDim(0) - 1;
        tilingData_.set_isVariedLen(1);
        tilingData_.set_seqlen(numTokens);  // 取上界
        tilingData_.set_shapeBatch(N);
    } else {
        OP_LOGE(inputParams_.opName, "q must be 3D or 4D, got rank %zu", qS.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    tilingData_.set_kNumHead(Hg);
    tilingData_.set_vNumHead(H);
    tilingData_.set_kHeadDim(K);
    tilingData_.set_vHeadDim(V);
    tilingData_.set_tokenBatch(numTokens);
    tilingData_.set_totalChunks(hS.GetDim(0));  // h 形状为 (totalChunks, H, K, V)

    tilingData_.set_hasG(context_->GetOptionalInputDesc(G_INDEX) != nullptr ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    float scale = *attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    int64_t cs = *attrs->GetAttrPointer<int64_t>(CHUNK_SIZE_ATTR_INDEX);
    tilingData_.set_scale(scale);
    tilingData_.set_chunkSize(cs);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::PlanWorkspaces()
{
    int64_t BT = tilingData_.get_chunkSize();
    int64_t V  = tilingData_.get_vHeadDim();
    int64_t BV = std::min<int64_t>(128, V);
    int64_t dtSize = tilingData_.get_dataType() == 0 ? 2 : 2;  // bf16/fp16
    int64_t numCubeCore = static_cast<int64_t>(compileInfo_.aicNum > 0 ? compileInfo_.aicNum : 1);

    // 各类 workspace 在 GM 上的单核大小，512 字节对齐以匹配 L2 burst。
    int64_t hwsBytesPerCore   = CeilAlignT<int64_t>(BT * BV * sizeof(float), WORKSPACE_ALIGN);
    int64_t attnBytesPerCore  = CeilAlignT<int64_t>(BT * BT * sizeof(float), WORKSPACE_ALIGN);
    int64_t vwsBytesPerCore   = hwsBytesPerCore;
    int64_t amBytesPerCore    = CeilAlignT<int64_t>(BT * BT * dtSize, WORKSPACE_ALIGN);
    int64_t maskBytesPerCore  = CeilAlignT<int64_t>(BT * BT * sizeof(float), WORKSPACE_ALIGN);

    // 每个 AIC 拥有独立的 workspace 槽位。在 1 AIC + 2 AIV 混合布局下，
    // 同一 AIC 与配对 AIV 共用同一份 workspace，因此只需为每个 AIC
    // 预留一份缓冲。
    tilingData_.set_hWorkspaceOffset(0);
    tilingData_.set_attnWorkspaceOffset(hwsBytesPerCore * numCubeCore);
    tilingData_.set_vWorkspaceOffset(tilingData_.get_attnWorkspaceOffset() + attnBytesPerCore * numCubeCore);
    tilingData_.set_aftermaskWorkspaceOffset(tilingData_.get_vWorkspaceOffset() + vwsBytesPerCore * numCubeCore);
    tilingData_.set_maskWorkspaceOffset(tilingData_.get_aftermaskWorkspaceOffset() + amBytesPerCore * numCubeCore);

    totalWorkspaceBytes_ = tilingData_.get_maskWorkspaceOffset() + maskBytesPerCore * numCubeCore;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkFwdOTiling::PlanCorePartition()
{
    int64_t V  = tilingData_.get_vHeadDim();
    int64_t H  = tilingData_.get_vNumHead();
    int64_t N  = tilingData_.get_shapeBatch();
    int64_t BV = std::min<int64_t>(128, V);
    int64_t BVN = CeilDivT<int64_t>(V, BV);
    int64_t totalTasks = BVN * N * H;

    int64_t aicNum = static_cast<int64_t>(compileInfo_.aicNum > 0 ? compileInfo_.aicNum : 1);
    int64_t aivNum = static_cast<int64_t>(compileInfo_.aivNum > 0 ? compileInfo_.aivNum : aicNum * 2);

    // 当任务量小于核数时，按需收缩，避免空跑核。
    int64_t usedAic = std::min<int64_t>(totalTasks, aicNum);
    int64_t usedAiv = usedAic * (aivNum / std::max<int64_t>(aicNum, 1));

    tilingData_.set_numCubeCore(usedAic);
    tilingData_.set_numVecCore(usedAiv);
    tilingData_.set_bvNum(BVN);
    tilingData_.set_bkNum(CeilDivT<int64_t>(tilingData_.get_kHeadDim(), 128));
    blockDim_ = static_cast<uint32_t>(usedAic);
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

uint64_t ChunkFwdOTiling::GetTilingKey() const
{
    return tilingKey_;
}

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
            "ChunkFwdO tiling: B=%ld T=%ld Hg=%ld H=%ld K=%ld V=%ld scale=%f BT=%ld varlen=%ld dtype=%ld AIC=%ld AIV=%ld",
            tilingData_.get_shapeBatch(),
            tilingData_.get_seqlen(),
            tilingData_.get_kNumHead(),
            tilingData_.get_vNumHead(),
            tilingData_.get_kHeadDim(),
            tilingData_.get_vHeadDim(),
            tilingData_.get_scale(),
            tilingData_.get_chunkSize(),
            tilingData_.get_isVariedLen(),
            tilingData_.get_dataType(),
            tilingData_.get_numCubeCore(),
            tilingData_.get_numVecCore());
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
