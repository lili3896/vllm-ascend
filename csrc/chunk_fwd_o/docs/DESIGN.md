# ChunkFwdO 设计文档

## 一、需求

### 1.1 需求背景和描述

`ChunkFwdO` 面向 vLLM Ascend 中 Flash-Linear-Attention（FLA）系列模型
（如 Gated DeltaNet、RWKV-7）的 chunk 前向输出计算。原有
`vllm_ascend/ops/triton/fla/chunk_o.py` 通过 triton kernel 完成三个矩阵乘
与若干向量后处理，算子语义清晰，但矩阵乘没有下放到 Ascend Cube 单元，NPU
吞吐存在瓶颈。

本需求是在 `csrc/chunk_fwd_o` 中提供 AscendC 自定义算子，保持与 triton
参考实现等价，并通过 1 AIC + 2 AIV 的 MIX kernel 将 `Q@K^T`、`Q@H`、
`A_masked@V` 三个矩阵乘放到 AIC，高频向量逻辑放到 AIV，以提升 FLA
chunk 输出阶段性能。PyTorch 侧通过
`torch.ops._C_ascend.npu_chunk_fwd_o` 调用，未注册或 shape 不支持时回落到
triton 路径。

### 1.2 验收标准

#### 1.2.1 精度标准

* **标杆算子:** 以 `vllm_ascend.ops.triton.fla.chunk_o.chunk_fwd_o` 作为
  精度比较基准；无 NPU 自定义算子环境下以 Python 分发器回落行为为 CI
  基准。
* **精度通过标准**：AscendC 输出转 fp32 后与 triton 输出
  `torch.testing.assert_close(atol=1e-2, rtol=1e-2)`。
* **验收范围**：业务 shape 覆盖 `chunk_size = 64`，泛化 shape 覆盖
  `chunk_size` 为 16 对齐且不超过 64 的场景；`K % 16 == 0`、
  `V % 16 == 0`、dtype 为 BF16/FP16；NPU 数值对比需在自定义算子已注册且
  NPU 可用时执行。

#### 1.2.2 性能标准

* **性能基线**: triton 参考实现 `chunk_fwd_o`。
* **性能通过标准**：在支持 shape 上，device 侧应通过 Cube 矩阵乘和
  AIC/AIV ping-pong 流水降低 chunk 输出阶段耗时；具体提升比例以目标交付
  环境 benchmark 为准。
* **验收范围**：重点验收业务 shape；泛化验收覆盖不同 `B/T/H/K/V`、
  varlen/fixed length、BF16/FP16 与 `g` 可选分支。

### 1.3 业务 shape

当前 UT 中已有 NPU 数值用例覆盖：

| 场景 | B | T | H/Hg | K | V | dtype | 说明 |
|------|---|---|------|---|---|-------|------|
| basic BF16 | 2 | 512 | 4 | 64 | 64 | BF16 | 固定长度 |
| large BF16 | 1 | 2048 | 8 | 128 | 128 | BF16 | 较长序列 |
| basic FP16 | 2 | 256 | 4 | 64 | 64 | FP16 | 固定长度 |
| varlen | 2 | 同业务 T | H | K | V | BF16/FP16 | `cu_seqlens` 非空 |

泛化范围：

* `chunk_size` 当前支持 16 对齐且不超过 64 的正整数；典型业务取 64。
* `K`、`V` 需 16 对齐；`V` 按 `BV = min(128, V)` 分块，`vLoops = ceil(V/BV)`。
* `q/k` shape 为 `[B, T, Hg, K]`，`v` shape 为 `[B, H, T, V]`，
  `o` shape 为 `[B, T, H, V]`。
* `H` 需能按 GQA 规则映射到 `Hg`，kernel 使用 `i_hg = Hg == H ? i_h : i_h / (H / Hg)`。

## 二、约束和周边影响评估

### 2.1 周边依赖

1. docker 镜像
   * 仓库未在 `csrc/chunk_fwd_o` 中固定镜像地址；按交付环境或 CI 镜像补齐。
   * 项目贡献文档示例使用 Ascend CANN 镜像，实际地址以发布/CI 配置为准。
2. CANN
   * 依赖 CANN 自定义算子编译与 aclnn 两段式执行框架。
   * 构建入口：`bash csrc/build_aclnn.sh <repo_root> ascend910b` 或
     `bash csrc/build_aclnn.sh <repo_root> ascend910_93`。
   * CANN CMC 链接未在仓库中声明，按交付环境补齐。
3. PyTorch
   * `pyproject.toml` 声明 `torch==2.9.0`、`torch-npu==2.9.0`、
     `triton-ascend==3.2.0`。
   * PyTorch CMC 链接未在仓库中声明，按交付环境补齐。
4. 其他
   * `ascend910b`/`ascend910_93` 自定义算子构建依赖
     `csrc/third_party/catlass/include`，构建脚本会尝试初始化 submodule。

### 2.2 支持芯片型号

OpDef 中 AICore 配置支持：

* `ascend910b`
* `ascend910_93`

构建脚本注释中 `ascend910b` 对应 Ascend 910B（A2）系列，
`ascend910_93` 对应 Ascend 910C（A3）系列。

## 三、算子功能及定义

### 3.1 算子功能分析

* 算子功能：计算 FLA chunk 前向输出 `o`。输入包含当前 chunk 的
  `q/k/v`、chunk 起点隐状态 `h`、可选门控 `g`、序列切分信息和缩放系数。
* 是否涉及量化：否。本算子支持 BF16/FP16 计算输入输出，内部中间结果使用
  FP32 和输入 dtype。

对每个 batch、head、chunk `t` 和 V 方向分块 `i_v`，令
`Q_t/K_t/V_t/g_t` 为当前 chunk 切片，`H_t` 为 chunk 起点隐状态：

```text
A_t       = Q_t @ K_t^T
O_cross   = Q_t @ H_t
if g is not None:
    O_cross = O_cross * exp(g_t)[:, None]
    A_t = A_t * safe_exp(g_t[:, None] - g_t[None, :])
A_masked  = causal_mask(A_t)
O_intra   = A_masked @ V_t
O_t       = (O_cross + O_intra) * scale
```

其中 `causal_mask` 清零 chunk 内上三角，保证 token `i` 不读取未来 token
`j > i`。`safe_exp` 的等价逻辑在 kernel 中通过 `scl = exp(g_i) / exp(g_j)`
并在 `scl > 1.0` 时置 0 实现。

### 3.2 参数说明

| 参数名 | 参数描述 | 可选/必选 | 数据类型 | 数据格式 | 维度 | 值域 | 是否支持非连续张量 | 是否有数据对齐要求 | 是否支持空 tensor | 异常值域 nan/inf/-inf |
|--------|---------|----------|---------|---------|------|------|----------------|----------------|--------------|------------------|
| `q` | 查询向量 | 必选 | BF16/FP16 | ND | `[B, T, Hg, K]` | 典型随机有限值 | aclnn 入口会连续化 | `K % 16 == 0` | 不支持 | 不建议；按浮点传播 |
| `k` | 键向量 | 必选 | BF16/FP16 | ND | `[B, T, Hg, K]` | 典型随机有限值 | aclnn 入口会连续化 | `K % 16 == 0` | 不支持 | 不建议；按浮点传播 |
| `v` | 值向量 | 必选 | BF16/FP16 | ND | `[B, H, T, V]` | 典型随机有限值 | aclnn 入口会连续化 | `V % 16 == 0` | 不支持 | 不建议；按浮点传播 |
| `h` | chunk 起点隐状态 | 必选 | BF16/FP16 | ND | `[B, H, NT, K, V]`，host 读取 `h.shape[2]` 作为 `numChunks` | 典型随机有限值 | aclnn 入口会连续化 | `K/V` 与 `q/v` 一致 | 不支持 | 不建议；按浮点传播 |
| `g` | 可选 gate，AIV 端用于 `exp(g)` 衰减 | 可选 | FP32 | ND | `[B, H, T]` | 通常为非正累积衰减 | aclnn 入口会连续化 | T 维连续 | 不传时跳过 gate | 不建议；exp 可能放大异常 |
| `cu_seqlens` | 累积序列长度 | 必选 | INT64 | ND | `[N + 1]` | 单调递增，首元素为 0 | aclnn 入口会连续化 | 无 | 不支持 | 不适用 |
| `chunk_indices` | chunk 索引表，列 0 为序列 id，列 1 为序列内 chunk id | 必选 | INT64 | ND | `[totalChunks, 2]` | 合法序列 id 和 chunk id | aclnn 入口会连续化 | 无 | 不支持 | 不适用 |
| `scale` | 输出缩放系数 | 必选 attr | FP32 标量 | - | 标量 | 默认通常为 `1/sqrt(K)` | - | - | - | 需为有限值 |
| `chunk_size` | T 轴 chunk 大小 | 必选 attr | INT64 标量 | - | 标量 | 典型 64；泛化支持 16 对齐且不超过 64 | - | `16/32/48/64` | - | 不适用 |
| `o` | 输出 tensor，dtype 与 `q` 一致，布局为 token-major | 必选输出 | BF16/FP16 | ND | `[B, T, H, V]` | 由输入决定 | ViewCopy 到调用方输出 | `V % 16 == 0` | 不支持 | 按浮点传播 |

**其他约束**：

* `q/k/v/h/o` dtype 必须为 BF16 或 FP16，且 `o` dtype 与 `q` 一致。
* `cu_seqlens`、`chunk_indices` 必须为 INT64。
* `chunk_size` 必须为 16 对齐且不超过 64 的正整数；超出该范围会在 host tiling
  阶段返回失败，Python 分发器会提前回落 triton。
* Python 分发器 `_supports_shape` 要求 `K % 16 == 0` 且 `V % 16 == 0`。
* 当前 kernel 通过 `totalChunks = chunk_indices.shape[0]` 遍历 chunk；`q/k`
  从 `[B,T,Hg,D]` 按 head 维跳步搬入，等效内部 `[B,Hg,T,D]`；`o` 按
  `[B,T,H,D]` 跳步写回，实现相对 `v` 的转置输出。

### 3.3 其他算子功能支持

* 是否支持图模式：OpDef 开启 dynamic shape/dynamic rank/dynamic format，
  支持以 AICore 自定义算子形式接入图执行；实际图模式验收依赖上层集成测试。
* 是否支持确定性计算：同一输入和同一执行环境下计算路径固定，无随机分支。
* 是否涉及反向测试：否，本算子仅覆盖前向输出。

#### 3.3.2 算子原型定义

OpDef 名称：`ChunkFwdO`

输入：

```text
q, k, v, h, g(optional), cu_seqlens, chunk_indices
```

属性：

```text
scale: float
chunk_size: int64
```

输出：

```text
o
```

#### 3.3.3 接口定义

* aclnn 第一段接口：

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

* aclnn 第二段接口：

```cpp
aclnnStatus aclnnChunkFwdO(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

* PyTorch 绑定接口：

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

* Python 分发器接口：

```python
def chunk_fwd_o_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

## 四、详细方案设计

### 4.1 计算逻辑

kernel 入口 `chunk_fwd_o` 使用 `KERNEL_TYPE_MIX_AIC_1_2`。每个 block
包含 1 个 AIC 和配对的 2 个 AIV：

1. AIC 路径：
   * `Cube1`: `Q_t @ K_t^T -> attnWs`
   * 等待 AIV 完成 gate + mask 后的 `amWs`
   * `Cube2`: `Q_t @ H_t -> hWs`
   * `Cube3`: `A_masked @ V_t -> vWs`
2. AIV 路径：
   * `Vec1`: 读取 `attnWs`，应用 `g` 衰减和 causal mask，cast 后写入
     `amWs`。
   * `Vec2`: 读取 `hWs` 和 `vWs`，可选乘 `exp(g)`，执行 `Add + scale + cast`
     并写回 `o`。

调度器采用 `PING_PONG_STAGES = 2` 双缓冲软件流水。主循环中
`taskNew = s - 1` 执行 Phase A，`taskOld = s - 2` 执行 Phase B，使当前任务
的 `Q@K^T/Vec1` 与上一任务的 `Q@H + A@V/Vec2` 交错。

跨核同步 flag：

| flag | 生产者 | 消费者 | 含义 |
|------|--------|--------|------|
| `CUBE1_DONE[buf]` | AIC | AIV | `Q@K^T` 已写入 `attnWs[buf]` |
| `VEC1_DONE[buf]` | AIV | AIC | gate + mask 后 `A` 已写入 `amWs[buf]` |
| `CUBE23_DONE[buf]` | AIC | AIV | `Q@H`、`A@V` 已写入 `hWs/vWs[buf]` |
| `VEC2_DONE[buf]` | AIV | AIC | 输出已写回，workspace 可复用 |

### 4.2 Tiling 方案

#### 多核切分策略

| 项目 | 描述 |
|------|------|
| 切分维度 | 将 `(vLoop, chunk/global chunk, head)` 展开为一维 `taskId` |
| 每核处理量 | AIC 按 `[taskNum * blockId / numCubeCore, taskNum * (blockId + 1) / numCubeCore)` 均分 |
| AI Core 数量 | `numCubeCore = min(taskNum, platform.aicNum)`；AIV 数为配套 `numVecCore` |

`taskId` 解码逻辑：

```text
i_v  = taskId % vLoops
rest = taskId / vLoops
i_tg = rest % totalChunks
i_h  = rest / totalChunks
chunk_indices[i_tg] = [i_n, i_t]
```

其中 `i_hg = Hg == H ? i_h : i_h / (H / Hg)` 处理 GQA 头映射。

#### UB 切分策略

| 项目 | 描述 |
|------|------|
| 单次处理数据量 | `BT = chunk_size` token（当前实现上限 64），`BV = min(128, V)` value 维 |
| 是否分 chunk | 是，T 维按 `chunk_size` 切 chunk；V 维按 `BV = min(128, V)` 切块 |
| chunk 大小计算公式 | `numChunks = h.shape[2]`，`totalChunks = chunk_indices.shape[0]`，`vLoops = ceil(V / BV)` |

AIV 内按 `GetSubBlockIdx()` 将 `BT` 行均分给两个 AIV 子核：

```text
rowBegin = BT * subId / subNum
rowEnd   = BT * (subId + 1) / subNum
```

#### Buffer 规划

| Buffer 名称 | 用途 | 大小计算公式 |
|------------|------|-------------|
| `qhBuf_` | `Q@H` 中间结果 | `align(BT,16) * align(BV,16) * sizeof(float)` |
| `avBuf_` | `A@V` 中间结果 | `align(BT,16) * align(BV,16) * sizeof(float)` |
| `attnBuf_` | `Q@K^T` 注意力分数 | `align(BT,16) * align(BT,16) * sizeof(float)` |
| `gBuf_` | gate 输入 | `align(BT,8) * sizeof(float)` |
| `gExpBuf_` | `exp(g)` | `align(BT,8) * sizeof(float)` |
| `maskBuf_` | causal mask | `align(BT,16) * align(BT,16) * sizeof(float)` |
| `amBuf_` | mask 后 A，供 Cube3 使用 | `align(BT,16) * align(BT,16) * sizeof(Q_T)` |
| `outBuf_` | 输出写回缓存 | `align(BT,16) * align(BV,16) * sizeof(Q_T)` |

GM workspace 按 AIC 和 ping-pong stage 划分，512B 对齐：

| Workspace 段 | 用途 | 单槽大小 |
|--------------|------|----------|
| `hWorkspace` | `Q@H` FP32 结果 | `align512(BT * BV * sizeof(float))` |
| `attnWorkspace` | `Q@K^T` FP32 结果 | `align512(BT * BT * sizeof(float))` |
| `vWorkspace` | `A@V` FP32 结果 | `align512(BT * BV * sizeof(float))` |
| `aftermaskWorkspace` | gate/mask 后 A | `align512(BT * BT * sizeof(Q_T))` |

#### 分支场景覆盖

| 分支 | 条件 | 处理策略 |
|------|------|---------|
| BF16 | `dataType == 0` | 模板参数 `bfloat16_t`，cast 使用 `CAST_RINT` |
| FP16 | `dataType == 1` | 模板参数 `half`，cast 使用 `CAST_NONE` |
| 有 gate | `hasG != 0` | Vec1 应用 `exp(g_i - g_j)`，Vec2 对 `Q@H` 乘 `exp(g_i)` |
| 无 gate | `hasG == 0` | 跳过 gate，仍执行 causal mask |
| 尾 chunk | `actBT < BT` | `SetTail` 和 `DataCopyPad` 控制有效行 |
| V 尾块 | `actBV < BV` | `SetTail` 和 `DataCopyPad` 控制有效列 |
| varlen | `cu_seqlens.shape[0] > 2` | 使用 `cu_seqlens` 和 `chunk_indices` 计算 `bos/eos/i_t` |

#### TilingData 结构体

```cpp
struct alignas(8) ChunkFwdOTilingData {
    int64_t shapeBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    float   scale;
    int64_t chunkSize;
    int64_t isVariedLen;
    int64_t totalChunks;
    int64_t numChunks;
    int64_t vLoops;
    int64_t taskNum;
    int64_t numCubeCore;
    int64_t numVecCore;
    int64_t dataType;
    int64_t hasG;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t vWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
};
```

### 4.3 Kernel 方案

#### 模板划分

| 模板 | 触发条件 | 模板参数 | 适用场景 |
|-----|---------|---------|---------|
| 模板一 | `dataType == CHUNK_FWD_O_DTYPE_BF16` | `Q_T = bfloat16_t` | BF16 输入输出 |
| 模板二 | `dataType == CHUNK_FWD_O_DTYPE_FP16` | `Q_T = half` | FP16 输入输出 |

TilingKey 由 host 侧设置：BF16 为 0，FP16 为 1。

#### API 映射

| 计算步骤 | Ascend C API | 参数签名 | 约束说明 |
|---------|-------------|---------|---------|
| 注册矩阵乘对象 | `REGIST_MATMUL_OBJ` | `pipe, GetSysWorkSpacePtr(), mmQH, mmQK, mmAV` | 高阶 Matmul 使用前注册 |
| 矩阵乘输入 | `SetTensorA/SetTensorB` | `GlobalTensor[offset], transpose` | `Q@K^T` 的 K 使用 transpose |
| 矩阵乘尾块 | `SetTail` | `M, N, K` | 处理尾 chunk 和 V 尾块 |
| 矩阵乘执行 | `IterateAll` | `dstWorkspace, 0` | 输出写入 GM workspace |
| GM 到 UB | `DataCopy` | `LocalTensor, GlobalTensor, len` | 读取中间结果和 gate |
| 向量计算 | `Exp/Mul/Muls/Add/Cast` | LocalTensor 组合 | gate、mask、融合输出 |
| UB 到 GM | `DataCopy` / `DataCopyPad` | `GlobalTensor, LocalTensor, params` | 输出按有效 BT/BV 写回 |
| 跨核同步 | `CrossCoreSetFlag/CrossCoreWaitFlag` | `MakeFlag(base, buf)` | 保护 ping-pong workspace 复用 |

#### 数据流

```text
1. AIC: GM(q, k) -> Matmul(Q@K^T) -> GM(attnWs)
2. AIV: GM(attnWs, g) -> UB gate/mask/cast -> GM(amWs)
3. AIC: GM(q, h) -> Matmul(Q@H) -> GM(hWs)
4. AIC: GM(amWs, v) -> Matmul(A@V) -> GM(vWs)
5. AIV: GM(hWs, vWs, g) -> UB add/scale/cast -> GM(o)，写回时按
   `[B,T,H,D]` 的 token-major stride 跳写
```

#### 内存管理

| 内存区域 | 大小计算 | 说明 |
|---------|---------|------|
| 输入 UB | `attnBuf_ + gBuf_ + gExpBuf_ + maskBuf_` | AIV 读取 attention 和 gate，mask 初始化一次 |
| 输出 UB | `outBuf_` | cast 后写回 `o` |
| 临时缓冲区 | `qhBuf_ + avBuf_ + amBuf_` | 存放 `Q@H`、`A@V`、mask 后 A |
| Workspace | `numCubeCore * PING_PONG_STAGES * (hSlot + attnSlot + vSlot + amSlot)` | Cube/AIV 跨核传递中间结果 |

## 五、测试策略

### 5.1 测试参数说明

业务 shape：

* 典型：`B=1`，TokenBatch（子序列数）`[1,4]`，`T` 覆盖
  `[1K, 64K * TokenBatch]`，`H` 覆盖 `[2,32]`，`D=128`，
  `chunk_size=64`，dtype=BF16。
* 现有 UT：`(B=2, T=512, H=4, K=64, V=64, dtype=BF16)`、
  `(B=1, T=2048, H=8, K=128, V=128, dtype=BF16)`、
  `(B=2, T=256, H=4, K=64, V=64, dtype=FP16)`。

泛化 shape：

* 泛化：`B=1`，TokenBatch（子序列数）`[1,128]`，`T >= 1`，
  `H` 覆盖 `[2,1024]`，`H % Hg == 0`。
* `K`、`V` 为 16 的倍数。
* `T` 可非 64 整倍，尾 chunk 由 `actBT` 处理。
* 支持 `cu_seqlens` 表示的 varlen 场景，`chunk_indices` 需与
  `cu_seqlens/chunk_size` 一致。

range 值域：

* `q/k` 典型值域 `[-1,1]`；`v` 典型值域 `[-50,50]`；
  `h` 典型值域 `[-100,100]`；泛化值域按 L1。
* `g` 为 FP32，通常为非正累积衰减；异常值 `nan/inf/-inf` 不作为支持范围。
* `scale` 需为有限 FP32，默认可取 `K ** -0.5`。

### 5.2 测试脚本

#### 5.2.1 CPU golden 实现

当前仓库未为该算子提供 CPU golden。无 NPU 自定义算子时，CI 侧验证
Python 分发器：

* AscendC op 未注册时回落 triton。
* 非法 `chunk_size`（非 16 对齐、非正数或大于 64）时回落 triton。
* `chunk_size=32` 等非 64 但满足约束的值会进入 AscendC 支持范围。
* 环境变量 `VLLM_ASCEND_DISABLE_CHUNK_FWD_O_ASCENDC=1` 时强制回落 triton。

命令：

```bash
pytest -sv tests/ut/ops/test_chunk_fwd_o_ascendc.py::TestChunkFwdODispatcher
```

#### 5.2.2 NPU 小算子实现

在 NPU 且自定义算子已注册时，运行数值对比：

```bash
pytest -sv tests/ut/ops/test_chunk_fwd_o_ascendc.py::TestChunkFwdONumerical
```

该测试以 triton `chunk_fwd_o` 为标杆，比较 AscendC 路径输出：

```python
torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
```
