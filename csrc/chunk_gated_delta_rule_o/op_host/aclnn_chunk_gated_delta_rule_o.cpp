/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_chunk_gated_delta_rule_o.h"
#include "chunk_gated_delta_rule_o.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr size_t QKV_DIM_NUM   = 4;
constexpr size_t H_DIM_NUM     = 5;

static const std::initializer_list<op::DataType> QKVH_DTYPES   = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> G_DTYPES      = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> SEQ_DTYPES    = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_DTYPES    = {op::DataType::DT_BF16};

struct CGDROParams {
    const aclTensor *query{nullptr};
    const aclTensor *key{nullptr};
    const aclTensor *value{nullptr};
    const aclTensor *h{nullptr};
    const aclTensor *g{nullptr};
    const aclTensor *cuSeqlens{nullptr};
    const aclTensor *chunkOffsets{nullptr};
    float            scaleValue{1.0f};
    int64_t          chunkSize{64};
    const aclTensor *out{nullptr};
};

static inline bool CheckNotNull(const CGDROParams &p) {
    OP_CHECK_NULL(p.query, return false);
    OP_CHECK_NULL(p.key,   return false);
    OP_CHECK_NULL(p.value, return false);
    OP_CHECK_NULL(p.h,     return false);
    OP_CHECK_NULL(p.out,   return false);
    return true;
}

static inline bool CheckDtype(const CGDROParams &p) {
    OP_CHECK_DTYPE_NOT_SUPPORT(p.query, QKVH_DTYPES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.key,   QKVH_DTYPES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.value, QKVH_DTYPES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.h,     QKVH_DTYPES, return false);
    if (p.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(p.g, G_DTYPES, return false);
    }
    if (p.cuSeqlens != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(p.cuSeqlens, SEQ_DTYPES, return false);
    }
    if (p.chunkOffsets != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(p.chunkOffsets, SEQ_DTYPES, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(p.out, OUT_DTYPES, return false);
    return true;
}

}  // namespace

aclnnStatus aclnnChunkGatedDeltaRuleOGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *h, const aclTensor *g, const aclTensor *cuSeqlens,
    const aclTensor *chunkOffsets, float scaleValue, int64_t chunkSize,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleO,
                   DFX_IN(query, key, value, h, g, cuSeqlens, chunkOffsets,
                          scaleValue, chunkSize),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CGDROParams params{query, key, value, h, g, cuSeqlens, chunkOffsets,
                       scaleValue, chunkSize, out};
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params), ACLNN_ERR_PARAM_INVALID);

    auto query_  = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_    = l0op::Contiguous(key,   uniqueExecutor.get());
    auto value_  = l0op::Contiguous(value, uniqueExecutor.get());
    auto h_      = l0op::Contiguous(h,     uniqueExecutor.get());
    auto g_      = (g != nullptr) ? l0op::Contiguous(g, uniqueExecutor.get())
                                  : nullptr;
    auto cuSeq_  = (cuSeqlens != nullptr)
                       ? l0op::Contiguous(cuSeqlens, uniqueExecutor.get())
                       : nullptr;
    auto chunkO_ = (chunkOffsets != nullptr)
                       ? l0op::Contiguous(chunkOffsets, uniqueExecutor.get())
                       : nullptr;
    auto out_    = l0op::Contiguous(out, uniqueExecutor.get());

    auto outRet = l0op::ChunkGatedDeltaRuleO(query_, key_, value_, h_, g_,
                                             cuSeq_, chunkO_,
                                             scaleValue, chunkSize,
                                             uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopy = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (viewCopy == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleO(void *workspace, uint64_t workspaceSize,
                                      aclOpExecutor *executor,
                                      aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleO);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
