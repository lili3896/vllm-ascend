/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o_def.cpp
 * \brief Operator prototype registration for ChunkGatedDeltaRuleO.
 *
 * Inputs (matching the Triton chunk_o.py contract):
 *   query : [B, T, Hg, K] bf16
 *   key   : [B, T, Hg, K] bf16
 *   value : [B, T, H,  V] bf16
 *   h     : [B, NT, H, K, V] bf16  (recurrent state at every chunk start)
 *   g                : [B, T, H] fp32 (optional; cumulative log-decay gate)
 *   cu_seqlens       : [N+1]    int32 (optional; varlen flag)
 *   chunk_offsets    : [N+1]    int32 (optional; pre-built chunk offsets)
 *
 * Output:
 *   out   : [B, T, H, V] bf16
 *
 * Attribute:
 *   scale : float, default 1.0 (caller usually passes 1/sqrt(K))
 */

#include "register/op_def_registry.h"

namespace ops {

class ChunkGatedDeltaRuleO : public OpDef {
public:
    explicit ChunkGatedDeltaRuleO(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("g")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("chunk_offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
    }
};

OP_ADD(ChunkGatedDeltaRuleO);

}  // namespace ops
