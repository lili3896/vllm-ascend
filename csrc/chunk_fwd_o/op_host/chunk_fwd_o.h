/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_CHUNK_FWD_O_H
#define OP_API_CHUNK_FWD_O_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* ChunkFwdO(const aclTensor* q, const aclTensor* k, const aclTensor* v,
                           const aclTensor* h, const aclTensor* g,
                           const aclTensor* cuSeqlens, const aclTensor* chunkOffsets,
                           float scale, int64_t chunkSize,
                           aclOpExecutor* executor);

}  // namespace l0op

#endif  // OP_API_CHUNK_FWD_O_H
