# aclnnChunkFwdO 接口说明

## 功能说明

`aclnnChunkFwdO` 是 `ChunkFwdO` 的 aclnn 两段式接口，用于执行 FLA
chunk 前向输出计算：

```text
O_t = scale * (Q_t @ H_t + causal_mask(Q_t @ K_t^T, g_t) @ V_t)
```

当 `g` 非空时，`Q_t @ H_t` 乘 `exp(g_i)`，`Q_t @ K_t^T` 乘
`safe_exp(g_i - g_j)`；当 `g` 为空时仅执行 causal mask。

## 第一段接口

```cpp
aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor* q,
    const aclTensor* k,
    const aclTensor* v,
    const aclTensor* h,
    const aclTensor* g,
    const aclTensor* cuSeqlens,
    const aclTensor* chunkIndices,
    float scale,
    int64_t chunkSize,
    aclTensor* o,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `q` | 输入 | 查询向量，BF16/FP16，ND，shape `[B, T, Hg, K]` |
| `k` | 输入 | 键向量，BF16/FP16，ND，shape `[B, T, Hg, K]` |
| `v` | 输入 | 值向量，BF16/FP16，ND，shape `[B, H, T, V]` |
| `h` | 输入 | chunk 起点隐状态，BF16/FP16，ND；host tiling 读取第 3 维作为 `numChunks` |
| `g` | 输入 | 可选 gate，FP32，ND，shape `[B, H, T]`；可传 `nullptr` |
| `cuSeqlens` | 输入 | 累积序列长度，INT64，ND，shape `[N + 1]` |
| `chunkIndices` | 输入 | chunk 索引表，INT64，ND，shape `[totalChunks, 2]` |
| `scale` | 输入 | 输出缩放系数，常用 `1 / sqrt(K)` |
| `chunkSize` | 输入 | chunk 大小，当前仅支持 64 |
| `o` | 输出 | 输出 tensor，BF16/FP16，ND，shape 与 `v` 一致 |
| `workspaceSize` | 输出 | 返回执行所需 device workspace 字节数 |
| `executor` | 输出 | 返回 op executor，供第二段接口执行 |

### 检查与处理流程

1. 检查必选 tensor 非空：`q/k/v/h/cuSeqlens/chunkIndices/o`。
2. 检查 dtype：
   * `q/k/v/h/o` 支持 BF16、FP16。
   * `g` 支持 FP32，且允许为空。
   * `cuSeqlens/chunkIndices` 支持 INT64。
3. 通过 `l0op::Contiguous` 将输入连续化。
4. 调用 L0 op `l0op::ChunkFwdO` 创建输出临时 tensor 并加入 AICore launcher。
5. 通过 `ViewCopy` 将 L0 输出复制到用户传入的 `o`。
6. 返回 executor 的 workspace 大小并释放 executor 所有权给调用方。

## 第二段接口

```cpp
aclnnStatus aclnnChunkFwdO(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `workspace` | 输入 | device workspace 起始地址，由调用方按第一段返回大小申请 |
| `workspaceSize` | 输入 | 第一段接口返回的 workspace 字节数 |
| `executor` | 输入 | 第一段接口返回的 op executor |
| `stream` | 输入 | acl runtime stream |

第二段接口调用 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)`
在指定 stream 上执行 kernel。

## 约束

* `chunkSize` 必须为 64。
* Python 分发器要求 `K % 16 == 0`、`V % 16 == 0`，不满足时回落 triton。
* 输出 shape 由 infer shape 设置为与 `v` 完全一致，输出 dtype 设置为与 `q`
  一致。
* 当前支持芯片配置为 `ascend910b` 和 `ascend910_93`。
* `g == nullptr` 时跳过 gate 分支，但仍执行 causal mask。

## PyTorch 绑定

`csrc/chunk_fwd_o/chunk_fwd_o_torch_adpt.h` 提供 PyTorch 适配层：

```cpp
at::Tensor npu_chunk_fwd_o(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& h,
    const c10::optional<at::Tensor>& g,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_indices,
    double scale,
    int64_t chunk_size);
```

`csrc/torch_binding.cpp` 将该接口注册为：

```text
torch.ops._C_ascend.npu_chunk_fwd_o
```

Python 入口 `vllm_ascend.ops.triton.fla.chunk_o_ascendc.chunk_fwd_o_ascendc`
优先调用该自定义算子；自定义算子不可用、环境变量禁用或 shape 不支持时回落
triton 参考实现。
