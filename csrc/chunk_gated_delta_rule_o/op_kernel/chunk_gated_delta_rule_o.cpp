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
 * \brief Kernel entry for ChunkGatedDeltaRuleO. Wraps the templated kernel
 *        class with a C ABI symbol that the runtime can launch.
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
    // 1. Register and fetch the TilingData blob produced by the host tiling.
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleOTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    // 2. The kernel only uses the AIV (vector) cores; declare so the runtime
    //    skips AIC scheduling.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 3. Prepare the TPipe and launch the templated kernel.
    TPipe pipe;
    ChunkGatedDeltaRuleOKernel<bfloat16_t, bfloat16_t> op;
    op.Init(query, key, value, h, g, cuSeqlens, chunkOffsets, out,
            &tilingData, &pipe);
    op.Process();

    (void)workspaceGM;  // currently unused; reserved for future cube path
}
