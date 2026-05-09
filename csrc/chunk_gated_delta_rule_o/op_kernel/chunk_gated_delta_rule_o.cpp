/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_o.cpp
 * \brief Kernel entry for ChunkGatedDeltaRuleO. Runs in AIC + AIV mix mode
 *        (1 AIC : 2 AIV) and dispatches to the corresponding template class.
 */

#include "chunk_gated_delta_rule_o.h"
#include "chunk_gated_delta_rule_o_tiling_data.h"

using namespace AscendC;
using namespace ChunkGatedDeltaRuleO;

extern "C" __global__ __aicore__ void
chunk_gated_delta_rule_o(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                         GM_ADDR h, GM_ADDR g,
                         GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                         GM_ADDR out,
                         GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleOTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    // Cube/Vector mix mode: every AIC is paired with two AIVs. Use the macros
    // ASCEND_IS_AIC / ASCEND_IS_AIV to compile-time select the half each
    // physical core executes.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;

    // Reserve user-side workspace area (the leading bytes of `workspaceGM`
    // are reserved by the runtime for sync state).
    GM_ADDR userWs = GetUserWorkspace(workspaceGM);

    if ASCEND_IS_AIC {
        ChunkGatedDeltaRuleOAicCore<bfloat16_t, bfloat16_t> aic;
        aic.Init(query, key, value, h,
                 cuSeqlens, chunkOffsets,
                 userWs, &tilingData, &pipe);
        aic.Process();
    }
    if ASCEND_IS_AIV {
        ChunkGatedDeltaRuleOAivCore<bfloat16_t, bfloat16_t> aiv;
        aiv.Init(g, cuSeqlens, chunkOffsets, out, userWs, &tilingData, &pipe);
        aiv.Process();
    }
}
