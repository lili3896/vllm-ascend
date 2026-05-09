/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o.cpp
 * \brief AscendC kernel entry point of ChunkFwdO. KERNEL_TYPE_MIX_AIC_1_2.
 */

#include "kernel_operator.h"
#include "chunk_fwd_o.h"

using namespace AscendC;
using namespace ChunkFwdO;

template <typename Q_T>
__aicore__ inline void ChunkFwdODispatch(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                                         GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                         GM_ADDR o, GM_ADDR workspace, TPipe* pipe,
                                         const ChunkFwdOTilingData* td)
{
    int64_t H  = td->vNumHead;
    int64_t V  = td->vHeadDim;
    int64_t N  = td->shapeBatch;
    int64_t BV = (BV_MAX < V) ? BV_MAX : V;
    int64_t BVN = CeilDiv<int64_t>(V, BV);
    int64_t totalTasks = BVN * N * H;

    int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    int64_t numCubeCores = td->numCubeCore;
    int64_t taskBegin = (totalTasks * blockId) / numCubeCores;
    int64_t taskEnd   = (totalTasks * (blockId + 1)) / numCubeCores;
    if (taskBegin >= taskEnd) {
        return;
    }

    if (g_coreType == AIC) {
        ChunkFwdOAIC<Q_T> op;
        op.Init(td, pipe);
        op.Process(q, k, v, h, cuSeqlens, chunkOffsets, workspace, taskBegin, taskEnd);
    } else {
        ChunkFwdOAIV<Q_T> op;
        op.Init(td, pipe);
        op.Process(g, o, cuSeqlens, chunkOffsets, workspace, taskBegin, taskEnd);
    }
}

extern "C" __global__ __aicore__ void
chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
            GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
            GM_ADDR o, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ChunkFwdOTilingData);
    GET_TILING_DATA(td, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    GM_ADDR userWs = AscendC::GetUserWorkspace(workspaceGM);

    if (td.dataType == CHUNK_FWD_O_DTYPE_BF16) {
        ChunkFwdODispatch<bfloat16_t>(q, k, v, h, g, cuSeqlens, chunkOffsets,
                                      o, userWs, &pipe, &td);
    } else {
        ChunkFwdODispatch<half>(q, k, v, h, g, cuSeqlens, chunkOffsets,
                                o, userWs, &pipe, &td);
    }
}
