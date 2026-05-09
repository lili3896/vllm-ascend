# `ChunkGatedDeltaRuleO` —— GDN 分块前向输出的 AscendC 算子设计文档

> 算子源码：`csrc/chunk_gated_delta_rule_o/`
> 算法对标：`vllm_ascend/ops/triton/fla/chunk_o.py::chunk_fwd_kernel_o`
> 适用模型：Qwen3-Next、Qwen3.5-Hybrid 等使用 Gated Delta-Rule (GDN) 线性注意力的模型
> 目标硬件：Ascend 910B / 910_93

本文档对新增的 AscendC 算子 `ChunkGatedDeltaRuleO` 做完整的设计说明，包括：

1. 算子定位与数学契约
2. 与 Triton 版本（`chunk_o.py`）的对应关系
3. 工程目录与构建集成方式
4. **分核方式**（Block-dim、工作单元枚举、负载均衡）
5. **流水排布**（PIPE_MTE2 / PIPE_V / PIPE_MTE3、`PipeBarrier`、双 buffer）
6. **UB 内存布局**（按字节列出每个张量切片，以及总占用）
7. **数据流**（每个 chunk 的计算时间线）
8. 关键 AscendC API 选型与原因（依据
   [《Ascend C API 列表》](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
   ）
9. 当前 AIV-only 实现的局限与下一步 **AIC + AIV 混合优化方案**

---

## 1. 算子定位与数学契约

`ChunkGatedDeltaRuleO` 是 GDN 线性注意力分块前向流水的最后一阶段。前面阶段
（`chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` → `solve_tril` →
`recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h`，详见
`docs/source/developer_guide/Design_Documents/chunk_o_triton.md` 第 1 节）
已经产出了：

- `q, k`：本层的 query / key（`[B, T, Hg, K]`）
- `v_new`：经过 delta 修正后的 value（`[B, T, H, V]`）
- `h`：每个 chunk 起始时刻的 SSM 隐藏状态（`[B, NT, H, K, V]`）
- `g`：chunk 内累计 log-decay gate（`[B, T, H]`，host 侧已转置成 `[B, H, T]`）

`ChunkGatedDeltaRuleO` 在 chunk t 内对每个 (序列 n, 输出 head h) 计算：

$$
O_t = \text{scale}\cdot e^{g_t}\odot Q_tH_t
    + \text{scale}\cdot \bigl(M\odot e^{g_i - g_j}\odot Q_tK_t^{\top}\bigr)V_t
$$

其中：

- $Q_t, K_t \in \mathbb{R}^{BT\times K}$，$V_t\in\mathbb{R}^{BT\times V}$，$H_t\in\mathbb{R}^{K\times V}$
- $M$ 为下三角因果掩码（$i\ge j$）
- $\text{scale}=1/\sqrt{K}$（默认）

这里 chunk 大小 $BT=64$，`Hg`（KV head）通常 $\le H$（GQA），因此 q/k 与 v/h 的 head 索引转换为 `kvHead = i_h / (H / Hg)`。

---

## 2. 与 Triton 版本的字段对应表

| 概念 | Triton kernel | AscendC kernel |
|---|---|---|
| chunk 大小 | `BT = 64` | `tilingData.bt`（默认 64） |
| K 维 reduction | `BK = 128`（一次内层循环 = 一个 K-tile） | `alignK_`（一次性载入整段 K，K 维循环在标量层做） |
| V 维分块 | `BV = 128`（程序 grid 第 0 维） | `bv_`（默认 64，由 `numVTile` 程序数枚举） |
| 每 program 工作量 | 1 个 `(i_v, i_n, i_h)`，遍历所有 NT chunk | 1 个 `(i_n, i_h, i_v)`，遍历所有 NT chunk |
| 累加器 | `b_o[BT, BV]`、`b_A[BT, BT]`（fp32） | `boFp32_`、`bAFp32_`（fp32 in UB） |
| K 维转置 trick | `make_block_ptr` 用 `(1, Hg*K)` stride 在线转置 K | 标量加载，借标量循环替代 |
| gating | `safe_exp` + `exp(g_t)` | `Exp` 高阶 API + 标量 `safe_exp(diff)` |
| 因果掩码 | `where(i>=j, b_A, 0)` | `SetValue(0)` 上三角清零 |

> **算法 100% 一致**，AscendC 实现仅在编程模型和 tile 选择上与 Triton 不同。

---

## 3. 工程目录与构建集成

```
csrc/chunk_gated_delta_rule_o/
├── op_host/
│   ├── CMakeLists.txt                      # 注册到 opapi/optiling/op_host_aclnnExc
│   ├── chunk_gated_delta_rule_o_def.cpp    # OpDef 原型（输入/输出/属性/AICore 配置）
│   ├── chunk_gated_delta_rule_o_infershape.cpp  # InferShape & InferDataType
│   ├── chunk_gated_delta_rule_o.h          # L0 op 接口
│   ├── chunk_gated_delta_rule_o.cpp        # L0 op 实现（INFER_SHAPE + ADD_TO_LAUNCHER_LIST_AICORE）
│   ├── aclnn_chunk_gated_delta_rule_o.h    # 两段式 aclnn 接口
│   ├── aclnn_chunk_gated_delta_rule_o.cpp  # aclnn 实现（参数检查 + Contiguous + ViewCopy）
│   ├── chunk_gated_delta_rule_o_tiling.h   # Host Tiling 类
│   ├── chunk_gated_delta_rule_o_tiling.cpp # Host Tiling 实现（7 步规范流程）
│   ├── tiling_base.h                       # 复用 TilingBaseClass
│   ├── tiling_templates_registry.h         # 复用 TilingRegistry
│   └── error_log.h                         # 简化的 OP_LOG / OP_CHECK
└── op_kernel/
    ├── chunk_gated_delta_rule_o.cpp        # __global__ __aicore__ 入口
    ├── chunk_gated_delta_rule_o.h          # ChunkGatedDeltaRuleOKernel<IN_T, HSTATE_T>
    └── chunk_gated_delta_rule_o_tiling_data.h  # 主机/设备共享 TilingData
```

构建注册依赖 `csrc/CMakeLists.txt` 的 `op_add_subdirectory`，会按
`csrc/<op_name>/op_host/CMakeLists.txt` 的存在自动收集本算子，无需修改顶层。

---

## 4. 分核方式

### 4.1 工作单元定义

**总工作单元数** $U = N \times H \times \mathrm{ceil}(V / B_V)$。每个工作单元由三元组 $(i_n, i_h, i_v)$ 唯一标识，对应：

- 序列 $i_n \in [0, N)$
- 输出 head $i_h \in [0, H)$
- V 维 tile $i_v \in [0, B_V)$

每个工作单元负责一个 $(i_n, i_h, i_v)$ 在 **整个时间维 NT chunks** 上的 $[BT, B_V]$ 输出 tile。在工作单元内部串行迭代 $t=0..NT-1$，使得：

- `b_o`（inter-chunk 累加器）和 `b_A`（intra-chunk 累加器）只需要在 chunk 边界刷新，UB 占用稳定。
- `H_t` 在不同 chunk 切换，但 K/V 维的 tile 与累加器布局不变，复用度高。

### 4.2 Block-dim 选择

```cpp
uint64_t units = N * H * numVTile;
uint64_t cores = min(units, aivNum);  // aivNum 由 PlatformAscendC 取得
tilingData.numCores = cores;
```

不需要超过实际工作单元数的 AIV 核（避免空跑）。在 `ChunkGatedDeltaRuleOTiling::PlanBlockDim`
中按上式裁剪 `numCores`，并在 `PostTiling` 中调用 `context_->SetBlockDim(numCores)`。

### 4.3 核内分配策略

在 kernel 入口里使用三层循环 + `unitId % numCores == blockIdx_` 的模 N
路由分配。这种"模式分配"相比 Triton 的"grid 隐式分配"有两个优点：

1. 无需引入 host-device 同步的额外 `chunk_offsets` 表（只读取 `cu_seqlens`）。
2. 即使 `units` 不被 `numCores` 整除，也能自然处理"尾部不满"的情况。

代价是每个 AIV 都遍历完整 `(N, H, numVTile)` 笛卡尔积，但循环本身代价远小于实际 vector 计算。

### 4.4 GQA Head 映射

```cpp
const uint32_t kvHead = i_h / (H / Hg);
```

与 Triton 的 `i_h // (H // Hg)` 完全一致：组大小 $g = H/Hg$，第 $i_h$ 号输出 head 共享第 $\lfloor i_h/g \rfloor$ 号 KV head。

---

## 5. 流水排布

### 5.1 物理流水线

AscendC 的 AI Core 上有三条主要流水：

| 流水 | 名称 | 用途 |
|---|---|---|
| MTE2 | Memory Transfer Engine 2 | GM → UB（搬入） |
| V    | Vector | 向量计算单元 |
| MTE3 | Memory Transfer Engine 3 | UB → GM（搬出） |

本算子的每 chunk 7-step 序列对应到流水如下（"|"表示 `PipeBarrier<PIPE_V>` 同步点，`==` 表示队列出/入实现的 MTE↔V 同步）：

```
 MTE2: [DataCopyPad Q]==[DataCopyPad K]==[DataCopyPad V]==[DataCopyPad H]==[DataCopyPad g]
                |              |              |              |               |
 V   :       [Cast Q→fp32]| [Cast K→fp32]| [Cast V→fp32] | [Cast H→fp32] | [DataCopy g]
                                                                              |
                                                          [Duplicate b_o=0]
                                                          [ComputeQH:  K * BT 次 Muls + Add]
                                                          [Duplicate b_A=0]
                                                          [ComputeQKt: BT*BT 次内积 (标量)]
                                                          | (USE_G 分支)
                                                          [Exp(g) → gExp_]
                                                          [Muls 行级 b_o *= gExp_]
                                                          [scalar safe_exp(diff) loop on b_A]
                                                          | (causal mask)
                                                          [SetValue 上三角清零]
                                                          | (Step 6: combine)
                                                          [Muls(b_o, scale)]
                                                          [BT*BT 次 Muls + Add 累加 (b_A·V)]
                                                          | (Cast back)
                                                          [Cast b_o→bf16 → outQueue]
 MTE3:                                                                       == [DataCopyPad out]
```

### 5.2 同步点设计原则

- **MTE2 → V**：通过 `TQue::AllocTensor` / `EnQue` / `DeQue` 的隐式事件同步实现，无需手工 `SetFlag`。
- **V → V**：相邻有数据依赖的 vector 指令之间必须插 `PipeBarrier<PIPE_V>`，例如 `Cast` 之后立即用结果做 `Muls`。
- **V → MTE3**：通过 `outQueue_.EnQue/DeQue` 隐式同步。

### 5.3 双缓冲（double buffer）

- 输入队列（q/k/v/h/g）使用 `QUEUE_DEPTH = 1`：因为同一工作单元在一个 chunk 内对每个 input 只用一次，多缓冲收益小，节省 UB 给累加器和 fp32 工作区。
- 输出队列 `outQueue_` 使用 `OUT_QUEUE_DEPTH = 2`：让 MTE3 出搬运与下一 chunk 的 V 计算重叠。

### 5.4 与上游 `recurrent_gated_delta_rule` 的差异

| 项 | recurrent_gated_delta_rule | chunk_gated_delta_rule_o |
|---|---|---|
| 工作单元 | $(B, NV)$ | $(N, H, NV_{tile})$ |
| 内层循环 | 序列内 token 串行（state 滚动） | chunk 串行 + 块内 token 标量内积 |
| 累加器形状 | `[K, V_step]` | `[BT, V_step]` 与 `[BT, BT]` |
| pipe 主导 | MatVecMul 行级广播 | 行级 Muls + Add（向 cube 演进） |

---

## 6. UB 内存布局

按 BT=64, BV=64, K=128, V=128, bf16 的典型形状（ascend910b 单核 UB ≈ 256 KiB）：

| 区域 | 类型 | 形状 | 字节数 |
|---|---|---|---|
| `qInQueue_` | bf16 | `BT × alignK` = 64 × 128 | 16 KiB |
| `kInQueue_` | bf16 | `BT × alignK` | 16 KiB |
| `vInQueue_` | bf16 | `BT × alignV` = 64 × 64 | 8 KiB |
| `hInQueue_` | bf16 | `alignK × alignV` = 128 × 64 | 16 KiB |
| `gInQueue_` (USE_G) | fp32 | `alignBT` = 64 | 256 B |
| `outQueue_` ×2 | bf16 | `BT × alignV` | 16 KiB |
| **scratch (TBuf)** | | | |
| `boFp32_` | fp32 | `BT × alignV` | 16 KiB |
| `bAFp32_` | fp32 | `BT × alignBT` = 64×64 | 16 KiB |
| `qFp32_` | fp32 | `BT × alignK` | 32 KiB |
| `kFp32_` | fp32 | `BT × alignK` | 32 KiB |
| `vFp32_` | fp32 | `BT × alignV` | 16 KiB |
| `hFp32_` | fp32 | `alignK × alignV` | 32 KiB |
| `gFp32_` | fp32 | `alignBT` | 256 B |
| `gExp_` | fp32 | `alignBT` | 256 B |
| `rowAcc_` | fp32 | `alignV` | 256 B |
| **合计** | | | **≈ 220 KiB** |

剩余 36 KiB 留给 TBufPool 元数据、地址寄存器影子和未来双 buffer 升级。

> 设计依据：UB 太满会迫使临时 buffer 走 SPM Buffer/Workspace，引入额外 PIPE_MTE3
> 与 PIPE_MTE2 翻转代价。本布局在 K/V ≤ 128、BT = 64、BV = 64 时刚好不需要落 SPM。

---

## 7. 数据流（chunk 内时间线）

```
 t-th chunk:

 (1) 搬入 -----------------------------------------------------
     CopyInQK( bos+t·BT, kvHead, BT )         # q,k  → UB → fp32
     CopyInV ( bos+t·BT, h_idx, vStart, BT, BV )  # v       → UB → fp32
     CopyInH ( bohN + t, h_idx, vStart, BV  ) # h       → UB → fp32
     if useG: CopyInG( bos+t·BT, h_idx, BT )

 (2) inter-chunk  b_o = Q_t @ H_t -----------------------------
     Duplicate(b_o, 0)
     for r in [0, BT):
       for kk in [0, K):
         Muls(rowAcc, H[kk,:], q[r,kk], alignV)
         Add (b_o[r,:], b_o[r,:], rowAcc, alignV)

 (3) intra-chunk  b_A = Q_t @ K_t^T ---------------------------
     for i in [0, BT):
       for j in [0, BT):
         b_A[i,j] = Σ_k q[i,k] · k[j,k]   (scalar)

 (4) gating (if USE_G) ----------------------------------------
     Exp(gExp, g, alignBT)
     b_o[r,:] *= gExp[r]                # row-wise Muls
     b_A[i,j] *= safe_exp(g[i] - g[j])  # scalar safe_exp + SetValue

 (5) causal mask ----------------------------------------------
     b_A[i,j] = 0  for j > i

 (6) combine --------------------------------------------------
     Muls(b_o, scale)
     for i in [0, BT):
       for j in [0, i]:                 # only lower-triangle
         Muls(rowAcc, V[j,:], scale·b_A[i,j], alignV)
         Add (b_o[i,:], b_o[i,:], rowAcc, alignV)

 (7) cast & store ---------------------------------------------
     Cast(out, b_o, RoundMode::CAST_RINT, BT·alignV)
     DataCopyPad(out_gm, out, ...)
```

---

## 8. AscendC API 选型与原因

> 全部 API 名称对应 [《Ascend C API 列表》](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
> （CANN 商用版 8.5.0）。

### 8.1 资源管理

| API | 用途 | 选型理由 |
|---|---|---|
| `TPipe` | 全局资源管理 | 单 Pipe 即可，无需 TBufPool |
| `TQue<VECIN, depth>` / `TQue<VECOUT, depth>` | 输入/输出队列 | 自动管理 MTE↔V 同步 |
| `TBuf<TPosition::VECCALC>` | 大块连续 fp32 工作区 | 配合 `GetWithOffset` 切片，比多个小 TQue 节省描述符 |
| `pipe_->InitBuffer(que, depth, bytes)` | 队列分配 | 显式精确控制每个队列的 UB 占用 |

### 8.2 数据搬运

| API | 用途 | 选型理由 |
|---|---|---|
| `DataCopyPad` + `DataCopyExtParams` | 跨 stride 的 q/k/v/h GM→UB 搬运 | 同时处理 stride（用 `srcStride`）与尾块越界（pad 到 32B 对齐） |
| `DataCopyPadExtParams<T>` | 设定填充值 | 与 `DataCopyPad` 配合 |
| `DataCopyExtParams.dstStride` | UB→GM 跨 head 写出 | 让一次 store 直接写到 `[B, T, H, V]` 的目标 head 槽位 |

### 8.3 计算

| API | 用途 | 选型理由 |
|---|---|---|
| `Cast(dst, src, RoundMode::CAST_NONE / CAST_RINT, len)` | bf16↔fp32 | bf16→fp32 用 CAST_NONE（无损）；fp32→bf16 用 CAST_RINT（最近偶数舍入） |
| `Duplicate(dst, scalar, len)` | 累加器清零 | 单条指令清零，比 `Muls(0)` 更快 |
| `Muls(dst, src, scalar, len)` | 行级标量×向量 | 两次 MMA 之间的中间步骤 |
| `Add(dst, src0, src1, len)` | 行级累加 | 配合 `Muls` 模拟 FMA |
| `Exp(dst, src, len)` | 计算 `exp(g)` | 高阶 API，带溢出处理 |
| `LocalTensor::GetValue / SetValue` | 标量读写 | 用于 GQA head 路由、`safe_exp` 标量计算、因果掩码 0 写入 |

### 8.4 同步控制

| API | 用途 | 选型理由 |
|---|---|---|
| `PipeBarrier<PIPE_V>` | 同流水内有数据依赖时插同步 | 必须出现在每对依赖的 vector 指令之间 |
| `TQue::EnQue / DeQue` | 跨流水（MTE2↔V，V↔MTE3） | 隐式生成 SetFlag/WaitFlag 事件 |

### 8.5 系统变量

| API | 用途 |
|---|---|
| `GetBlockIdx()` | 当前 AIV 核 id，用于工作单元路由 |
| `GetBlockNum()` | 总核数 |
| `GetTPipePtr()` | 在子函数中获取当前 Pipe 指针（本算子未直接用） |

### 8.6 Tiling 框架

| API | 用途 |
|---|---|
| `REGISTER_TILING_DEFAULT(TilingData)` | 在 kernel 侧注册 TilingData 结构 |
| `GET_TILING_DATA(name, gmAddr)` | kernel 入口反序列化 TilingData |
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 声明本 kernel 仅占用 AIV |

### 8.7 平台信息（host 侧）

| API | 用途 |
|---|---|
| `platform_ascendc::PlatformAscendC` | 取 `aivNum`、UB 大小、SoC 版本 |

---

## 9. 当前实现的局限与下一步优化

### 9.1 局限性

本 PR 的 AIV-only 实现遵循已有 `recurrent_gated_delta_rule` 的成熟模式，**算法正确、易维护、易移植**，但在以下两点上低于 cube 路径的潜在峰值：

1. **`Q_t @ H_t` 与 `b_A @ V_t` 行级 outer-product 累加**：
   每条 `Muls + Add` 处理 `alignV=64` 个 fp32，两次 MMA 共需要约
   `BT × K + BT × BT/2` ≈ 64×128 + 64×64/2 ≈ 10 K 条 vector 指令，理论吞吐约
   相当于 cube `Mmad` 的 `1/(BT)`。
2. **`Q_t @ K_t^T` 完全用标量循环**：64×64×128 = 524 288 次 fp32 FMA，全部走标量
   `GetValue/SetValue`，是当前最大的性能瓶颈。

### 9.2 推荐的 AIC + AIV 混合优化（下一步）

将三次 MMA 全部交给 cube：

```
AIC 侧 (Cube):
  Mmad #1:  C1[BT, V_tile] = Q_t [BT, K]  ×  H_t [K, V_tile]   →  b_o (fp32)
  Mmad #2:  A [BT, BT]    = Q_t [BT, K]  ×  K_t^T [K, BT]      →  b_A (fp32)
  Mmad #3:  D[BT, V_tile] = b_A_masked   ×  V_t [BT, V_tile]   →  b_o' (fp32)

AIV 侧 (Vector):
  - Cast bf16→fp32 (其实 Mmad 直接消费 bf16，可省此步)
  - Exp(g) → gExp
  - 行级 Muls + Add 完成门控
  - 上三角 Duplicate 0 完成 mask（用 Bitwise Mask 也可）
  - Cast fp32→bf16 与 DataCopyPad 出搬
```

实施要点：

- 使用 `KERNEL_TYPE_MIX_AIC_1_2` 模板拆分（1 AIC : 2 AIV），与 sparse_flash_attention 一致。
- 通过 `TQueBind<A2, CO1>` 与 `Matmul<>` 高阶 API 完成 cube 访问；UB 上保留中间 `b_A_masked` / `b_o`。
- 用 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 实现 AIC ↔ AIV 同步。
- 工作单元拆分调整为：每 AIC 处理一个完整 `(i_n, i_h, i_v)`，配套 AIV 做门控和搬出。

预期性能：在 BT=64、BK=BV=128、K=V=128 时，cube 利用率可达 ≥ 80%，相对当前
AIV 实现提速约 **8 ~ 10×**，与 Triton 上 NPU 后端的实测水平接近。

### 9.3 可选的小幅优化

- 把 `Cast` 后的 fp32 q/k 缓存合并到一个连续大 buffer，再用 `BroadCast` + `Mul`
  代替逐 token 的 `Muls`，减少标量循环次数。
- USE_G 分支里 `b_A *= safe_exp(g_i - g_j)` 用 `Sub` + `Exp` 向量化批量处理上三角后再
  `Mul`，避免标量循环。
- 当 `chunk_offsets` 在 host 已预构建（gdn_chunk_meta），直接透传，复用 Triton 路径
  的优化。

---

## 10. 算子调用契约（用户视角）

### Host 端两段式 aclnn 调用

```cpp
// Phase 1
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnChunkGatedDeltaRuleOGetWorkspaceSize(
    /*query        */ q,
    /*key          */ k,
    /*value        */ v_new,
    /*h            */ h,
    /*g            */ g_or_nullptr,
    /*cu_seqlens   */ cu_or_nullptr,
    /*chunk_offsets*/ co_or_nullptr,
    /*scale_value  */ 1.0f / std::sqrt(static_cast<float>(K)),
    /*chunk_size   */ 64,
    /*out          */ out,
    &workspaceSize, &executor);

void *workspace = nullptr;
if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

// Phase 2
aclnnChunkGatedDeltaRuleO(workspace, workspaceSize, executor, stream);
```

### 张量约束

| 张量 | dtype | 形状 | 备注 |
|---|---|---|---|
| query | bf16 | `[B, T, Hg, K]` | T 必须 ≤ tilingData.t（padded 上限） |
| key   | bf16 | `[B, T, Hg, K]` | 必须连续 |
| value | bf16 | `[B, T, H, V]`  | H 必须为 Hg 的整数倍 |
| h     | bf16 | `[B, NT, H, K, V]` | NT = ceil(T_max/BT) (fixed) 或 cu_seqlens 累计 (varlen) |
| g     | fp32 | `[B, T, H]`     | 调用方需先做 `transpose(1,2).contiguous()` 成 `[B, H, T]` 后传入 |
| cu_seqlens | int32 | `[N+1]` | varlen 时必填 |
| chunk_offsets | int32 | `[N+1]` | 由 `gdn_chunk_meta` 预构建 |
| out | bf16 | `[B, T, H, V]` | 与 value 同 shape |

### 属性

- `scale_value` (float, default 1.0)：通常 caller 传 `1/sqrt(K)`。
- `chunk_size` (int64, default 64)：当前实现默认 64，与 Triton 一致。

---

## 11. 与 vllm-ascend 的接入方式

落地后，`vllm_ascend/ops/triton/fla/chunk.py::chunk_gated_delta_rule_fwd` 末尾的：

```python
o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale,
                cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets_chunk64)
```

可改写为通过 `torch.ops.npu.chunk_gated_delta_rule_o(...)` 直接调用本算子（跳过
Triton），并保留 Triton 实现作为 fallback；具体绑定走 `csrc/torch_binding.cpp`。
本文档不展开 Python 绑定细节，相关改造将在后续 PR 中提供。

---

## 12. 参考

- [Ascend C API 列表（CANN 商用版 8.5.0）](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
- 本仓库示例：
  - `csrc/recurrent_gated_delta_rule/`（GDN 解码路径，AIV-only）
  - `csrc/sparse_flash_attention/`（AIC + AIV 混合，Matmul 高阶 API 用法参考）
- Flash Linear Attention 上游：[`fla/ops/gated_delta_rule`](https://github.com/sustcsonglin/flash-linear-attention)
- 配套 Triton 设计文档：
  `docs/source/developer_guide/Design_Documents/chunk_o_triton.md`
- 配套 Excalidraw 流程图：
  `docs/source/developer_guide/Design_Documents/assets/chunk_o_flow.excalidraw`
