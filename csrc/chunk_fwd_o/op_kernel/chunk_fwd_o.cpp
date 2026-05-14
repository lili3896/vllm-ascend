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
 * \brief ChunkFwdO 的 AscendC kernel 入口，使用 1 AIC + 2 AIV 的 MIX 类型。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_fwd_o.h"

using namespace AscendC;
using namespace ChunkFwdO;

template <typename Q_T>
__aicore__ inline void RunAic(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h,
                              GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                              GM_ADDR workspace,
                              const ChunkFwdOTilingData* td, TPipe* pipe)
{
    ChunkFwdOAIC<Q_T> op;
    // 高阶 Matmul 必须先注册到 TPipe，然后再被使用
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), op.mmQH, op.mmQK, op.mmAV);
    op.Init(q, k, v, h, cuSeqlens, chunkIndices, workspace, td, pipe);
    op.Process();
}

template <typename Q_T>
__aicore__ inline void RunAiv(GM_ADDR g, GM_ADDR o,
                              GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                              GM_ADDR workspace,
                              const ChunkFwdOTilingData* td, TPipe* pipe)
{
    ChunkFwdOAIV<Q_T> op;
    op.Init(g, o, cuSeqlens, chunkIndices, workspace, td, pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void
chunk_fwd_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
            GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
            GM_ADDR o, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ChunkFwdOTilingData);
    GET_TILING_DATA(td, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    GM_ADDR userWs = AscendC::GetUserWorkspace(workspaceGM);

    if (g_coreType == AIC) {
        if (td.dataType == CHUNK_FWD_O_DTYPE_BF16) {
            RunAic<bfloat16_t>(q, k, v, h, cuSeqlens, chunkIndices, userWs, &td, &pipe);
        } else {
            RunAic<half>(q, k, v, h, cuSeqlens, chunkIndices, userWs, &td, &pipe);
        }
    } else {
        if (td.dataType == CHUNK_FWD_O_DTYPE_BF16) {
            RunAiv<bfloat16_t>(g, o, cuSeqlens, chunkIndices, userWs, &td, &pipe);
        } else {
            RunAiv<half>(g, o, cuSeqlens, chunkIndices, userWs, &td, &pipe);
        }
    }
}
