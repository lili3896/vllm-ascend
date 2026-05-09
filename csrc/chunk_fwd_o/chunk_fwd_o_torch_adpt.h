/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef CHUNK_FWD_O_TORCH_ADPT_H
#define CHUNK_FWD_O_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_chunk_fwd_o(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& h,
    const c10::optional<at::Tensor>& g,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_offsets,
    double scale,
    int64_t chunk_size)
{
    at::Tensor output = at::empty(v.sizes(), v.options());
    float scale_real = static_cast<float>(scale);
    EXEC_NPU_CMD(aclnnChunkFwdO,
                 q,
                 k,
                 v,
                 h,
                 g,
                 cu_seqlens,
                 chunk_offsets,
                 scale_real,
                 chunk_size,
                 output);
    return output;
}

}  // namespace vllm_ascend
#endif  // CHUNK_FWD_O_TORCH_ADPT_H
