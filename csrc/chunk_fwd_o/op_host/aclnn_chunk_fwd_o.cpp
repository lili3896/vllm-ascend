/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_chunk_fwd_o.cpp
 * \brief ChunkFwdO 的两段式 aclnn 入口实现：参数检查 + 输入连续化 + 调用 L0。
 */

#include "aclnn_chunk_fwd_o.h"
#include "chunk_fwd_o.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace {

// 各输入支持的数据类型清单。
static const std::initializer_list<op::DataType> QKV_DT  = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> G_DT    = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> SEQ_DT  = {op::DataType::DT_INT64};

// 入参合法性校验：必选 tensor 非空、各 tensor dtype 在支持范围内。
static bool CheckParams(const aclTensor* q, const aclTensor* k, const aclTensor* v,
                        const aclTensor* h, const aclTensor* g,
                        const aclTensor* cuSeqlens, const aclTensor* chunkIndices,
                        const aclTensor* o)
{
    OP_CHECK_NULL(q, return false);
    OP_CHECK_NULL(k, return false);
    OP_CHECK_NULL(v, return false);
    OP_CHECK_NULL(h, return false);
    OP_CHECK_NULL(cuSeqlens, return false);
    OP_CHECK_NULL(chunkIndices, return false);
    OP_CHECK_NULL(o, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(q, QKV_DT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(k, QKV_DT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(v, QKV_DT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(h, QKV_DT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(o, QKV_DT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(cuSeqlens, SEQ_DT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(chunkIndices, SEQ_DT, return false);
    if (g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(g, G_DT, return false);
    }
    return true;
}

}  // namespace

aclnnStatus aclnnChunkFwdOGetWorkspaceSize(const aclTensor* q, const aclTensor* k, const aclTensor* v,
                                           const aclTensor* h, const aclTensor* g,
                                           const aclTensor* cuSeqlens, const aclTensor* chunkIndices,
                                           float scale, int64_t chunkSize,
                                           aclTensor* o, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnChunkFwdO,
                   DFX_IN(q, k, v, h, g, cuSeqlens, chunkIndices, scale, chunkSize),
                   DFX_OUT(o));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckParams(q, k, v, h, g, cuSeqlens, chunkIndices, o), ACLNN_ERR_PARAM_INVALID);

    // 把所有输入转成连续布局，避免 kernel 内部需要处理 stride。
    auto qC  = l0op::Contiguous(q,  uniqueExecutor.get());
    auto kC  = l0op::Contiguous(k,  uniqueExecutor.get());
    auto vC  = l0op::Contiguous(v,  uniqueExecutor.get());
    auto hC  = l0op::Contiguous(h,  uniqueExecutor.get());
    auto cuC = l0op::Contiguous(cuSeqlens, uniqueExecutor.get());
    auto ciC = l0op::Contiguous(chunkIndices, uniqueExecutor.get());
    const aclTensor* gC = nullptr;
    if (g != nullptr) {
        gC = l0op::Contiguous(g, uniqueExecutor.get());
    }

    // 调用 L0 接口，加入 launcher。
    auto outRet = l0op::ChunkFwdO(qC, kC, vC, hC, gC, cuC, ciC, scale, chunkSize, uniqueExecutor.get());
    CHECK_RET(outRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out_ = l0op::Contiguous(o, uniqueExecutor.get());
    auto viewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkFwdO(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdO);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
