/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * 本文件以 Apache License 2.0 协议发布，详见仓库根目录下的 LICENSE 文件。
 */

// ChunkFwdO 算子的 PyTorch 适配层：把 at::Tensor 转发到 aclnnChunkFwdO。

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
    const at::Tensor& chunk_indices,
    double scale,
    int64_t chunk_size)
{
    // L0 算子输出布局为 [B,T,H,D]；v 的对外输入布局为 [B,H,T,D]。
    at::Tensor output = at::empty({q.size(0), q.size(1), v.size(1), v.size(3)},
                                  v.options().dtype(q.scalar_type()));
    float scale_real = static_cast<float>(scale);
    EXEC_NPU_CMD(aclnnChunkFwdO,
                 q,
                 k,
                 v,
                 h,
                 g,
                 cu_seqlens,
                 chunk_indices,
                 scale_real,
                 chunk_size,
                 output);
    return output;
}

}  // namespace vllm_ascend
#endif  // CHUNK_FWD_O_TORCH_ADPT_H
