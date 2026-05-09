/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o_tiling.h
 * \brief Host tiling for ChunkGatedDeltaRuleO (AIC + AIV mix mode).
 */

#ifndef OP_HOST_CHUNK_GATED_DELTA_RULE_O_TILING_H
#define OP_HOST_CHUNK_GATED_DELTA_RULE_O_TILING_H

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base.h"
#include "error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/chunk_gated_delta_rule_o_tiling_data.h"

namespace optiling {

using namespace ChunkGatedDeltaRuleO;

struct ChunkGatedDeltaRuleOCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t aicNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l0aSize{0UL};
    uint64_t l0bSize{0UL};
    uint64_t l0cSize{0UL};
    platform_ascendc::SocVersion socVersion{platform_ascendc::SocVersion::ASCEND910B};
};

struct ChunkGatedDeltaRuleOInfo {
    const char *opName = "ChunkGatedDeltaRuleO";
};

class ChunkGatedDeltaRuleOTiling
    : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ChunkGatedDeltaRuleOTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context) {
        InitCompileInfo();
    }
    ~ChunkGatedDeltaRuleOTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void InitCompileInfo();
    void PrintTilingData() const;
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus GetScale();
    ge::graphStatus GetChunkSize();
    ge::graphStatus DetectOptionalInputs();
    void PlanBlockDim();
    void PlanWorkspaceLayout();
    ge::graphStatus ConfigMatmulTilings();

    ChunkGatedDeltaRuleOCompileInfo compileInfo_;
    ChunkGatedDeltaRuleOTilingData  tilingData_{};
    ChunkGatedDeltaRuleOInfo        inputParams_;
};

}  // namespace optiling

#endif  // OP_HOST_CHUNK_GATED_DELTA_RULE_O_TILING_H
