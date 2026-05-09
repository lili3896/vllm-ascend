/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_CHUNK_FWD_O_H
#define OP_API_ACLNN_CHUNK_FWD_O_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ChunkFwdO 第一段接口（FLA chunk 前向输出）：根据具体的计算流程，
 *        构建 op 执行器并计算所需的 workspace 大小。
 *
 * @param [in] q：数据类型支持 BF16、FLOAT16。
 * @param [in] k：数据类型支持 BF16、FLOAT16。
 * @param [in] v：数据类型支持 BF16、FLOAT16。
 * @param [in] h：数据类型支持 BF16、FLOAT16，shape 为 (totalChunks, H, K, V)。
 * @param [in] g：可选 gate，数据类型支持 FLOAT32。
 * @param [in] cuSeqlens：累加序列长度，数据类型 INT64。
 * @param [in] chunkOffsets：每个 batch 的 chunk 累加偏移，INT64。
 * @param [in] scale：注意力缩放系数。
 * @param [in] chunkSize：chunk 大小（默认 64）。
 * @param [out] o：输出 tensor，shape/dtype 与 v 一致。
 * @param [out] workspaceSize：device 端需申请的 workspace 字节数。
 * @param [out] executor：返回的 op 执行器。
 * @return aclnnStatus：状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* v, const aclTensor* h,
    const aclTensor* g, const aclTensor* cuSeqlens, const aclTensor* chunkOffsets,
    float scale, int64_t chunkSize,
    aclTensor* o, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief ChunkFwdO 第二段接口：在指定 stream 上执行算子。
 *
 * @param [in] workspace：device 上分配的 workspace 起始地址。
 * @param [in] workspaceSize：第一段接口返回的 workspace 大小。
 * @param [in] executor：op 执行器。
 * @param [in] stream：acl stream。
 * @return aclnnStatus：状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnChunkFwdO(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_CHUNK_FWD_O_H
