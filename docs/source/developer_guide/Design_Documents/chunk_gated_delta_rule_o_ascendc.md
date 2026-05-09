# `ChunkGatedDeltaRuleO` —— GDN 分块前向输出的 AscendC AIC + AIV 混合算子设计文档

> 算子源码：`csrc/chunk_gated_delta_rule_o/`
> 算法对标：`vllm_ascend/ops/triton/fla/chunk_o.py::chunk_fwd_kernel_o`
> 适用模型：Qwen3-Next、Qwen3.5-Hybrid 等使用 Gated Delta-Rule (GDN) 线性注意力的模型
> 目标硬件：Ascend 910B / 910_93
> 设计目标：单算子端到端性能 **优于** 同等场景下的 Triton 实现（`chunk_o.py`）

---

## 0. 为什么要用 AIC + AIV 混合模式？

`chunk_o.py` 在 NPU 上之所以可工作，是因为 Triton-Ascend 后端把 `tl.dot` 自动派发到 Cube 单元；但 Triton 的 grid 调度、heuristic 派生分支、上下三角 mask、`safe_exp` 保护等代码段都跑在 Vector 单元，且 grid 之间的 cube/vector 重叠程度受编译器调度限制。

本算子从设计上做四件事来超越 Triton：

| 优化项 | Triton 当前做法 | 本 AscendC 算子做法 |
|---|---|---|
| 三次主 MMA（`Q@H`、`Q@Kᵀ`、`A·V`） | `tl.dot` 由编译器映射到 Cube | **AIC 上** 通过 `matmul::MatmulImpl` 高阶 API 直接调度，三次都是 cube 原生指令 |
| 跨 chunk 的 cube / vector 重叠 | Triton 单 program 内串行，跨 program 由调度器决定 | 显式 **ping-pong workspace + 4 个 CrossCoreFlag**，让 chunk t+1 的 MM1/MM2 与 chunk t 的 vector 后处理 + MM3 重叠 |
| Chunk 内 BT 行的并行 | Triton 一个 program 处理整个 chunk | **1 AIC : 2 AIV** 模式下，BT 行被劈成上下两半给两个 AIV，门控 + safe_exp + mask + cast + store 全部并行 |
| 上三角 `safe_exp` 保护 | 每个 program 都按完整 BT×BT 跑 `safe_exp` 标量逻辑 | AIV 上用 `Adds` + `Mins(0)` + `Exp` 的 4 条向量指令实现，再叠加显式 `Duplicate(0)` 做 mask |

下图展示主流水时间线（一个工作单元 = `(序列 n, 输出 head h, V-tile v)`，时间从左到右；ping-pong slot 用 0/1 表示）：

```
chunk t-1  | t   | t+1 | t+2 | t+3
AIC (cube)  ┃ MM1+MM2(t)         ┃ MM3(t)         ┃ MM1+MM2(t+1) ┃ MM3(t+1) ┃ MM1+MM2(t+2) ...
              ↓ E_MM12_DONE        ↑ E_MASK_DONE     ↓ E_MM12_DONE
AIV-0       ┃ ...gate+mask+cast(t)┃ ...wait MM3...  ┃ combine+store(t)         ┃ gate+mask+cast(t+1) ...
AIV-1       ┃ ...gate+mask+cast(t)┃ ...wait MM3...  ┃ combine+store(t)         ┃ gate+mask+cast(t+1) ...
              ↑ E_MASK_DONE        ↓ E_MM3_DONE      ↑ E_STORE_DONE
slot:           0                    0                  1                       0
```

只要 `T(MM1+MM2) + T(MM3) > T(gate+mask+cast)` 且 `T(combine+store) < T(MM1+MM2 of next chunk)`，cube 几乎 100% 时间在算 MMA。这正是本算子的目标。

---

## 1. 算子定位与数学契约

`ChunkGatedDeltaRuleO` 是 GDN 线性注意力分块前向流水的最后一阶段。前面阶段（`chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` → `solve_tril` → `recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h`）已经产出 `q, k, v_new, h, g`。本算子计算：

$$
O_t = \text{scale}\cdot e^{g_t}\odot Q_tH_t
    + \text{scale}\cdot \bigl(M\odot e^{g_i - g_j}\odot Q_tK_t^{\top}\bigr)V_t
$$

约定与 Triton 完全一致：$BT=64$，$Hg \le H$（GQA），$\text{scale}=1/\sqrt K$，$M$ 为下三角因果掩码。

---

## 2. 工程目录与构建集成

```
csrc/chunk_gated_delta_rule_o/
├── op_host/
│   ├── CMakeLists.txt                            # 自动注册
│   ├── chunk_gated_delta_rule_o_def.cpp          # OpDef 原型
│   ├── chunk_gated_delta_rule_o_infershape.cpp   # InferShape & InferDataType
│   ├── chunk_gated_delta_rule_o.{h,cpp}          # L0 op
│   ├── aclnn_chunk_gated_delta_rule_o.{h,cpp}    # 两段式 aclnn API
│   ├── chunk_gated_delta_rule_o_tiling.{h,cpp}   # 7 步 host tiling + 三次 Matmul tiling
│   ├── tiling_base.h / tiling_templates_registry.h / error_log.h  # 复用框架
└── op_kernel/
    ├── chunk_gated_delta_rule_o.cpp              # __global__ __aicore__ 入口
    ├── chunk_gated_delta_rule_o.h                # AIC + AIV 两个模板类
    └── chunk_gated_delta_rule_o_tiling_data.h    # 共享 TilingData
```

`csrc/CMakeLists.txt::op_add_subdirectory` 自动收集本算子，无需顶层修改。

---

## 3. 核函数模式与分核

### 3.1 KERNEL_TYPE_MIX_AIC_1_2

入口处声明：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

含义：**1 个 AIC 配 2 个 AIV**，三类核共 N 组，运行时按 AIC 计数下发 grid。在入口里通过 `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 编译期分支选择 AIC 半或 AIV 半的逻辑（`g_coreType` 是 constexpr）。

### 3.2 工作单元与负载均衡

**工作单元三元组** $(i_n, i_h, i_v)$：
- `i_n` ∈ [0, N)：序列号
- `i_h` ∈ [0, H)：输出 head
- `i_v` ∈ [0, ⌈V/B_V⌉)：V 维 tile

**总工作单元数** $U = N \cdot H \cdot \mathrm{ceil}(V / B_V)$。

**Block-dim 计算**（在 `PlanBlockDim` 内）：
```
aicNum = min(U, hwAicNum);   // 不超过实际工作单元数
aivNum = 2 * aicNum;         // mix 1:2 自动派生
context_->SetBlockDim(aicNum);
```

每个 AIC 通过 `unitId % aicNum == aicIdx` 路由处理自己的工作单元；与之配对的 2 个 AIV 共享同一 `aicIdx`，但 `subBlock = aivIdx & 1` 决定它处理 BT 的上半（rows `0..BT/2`）还是下半（rows `BT/2..BT`）。

### 3.3 GQA Head 映射

```cpp
const uint32_t kvHead = (groupSize_ > 0) ? (hIdx / groupSize_) : 0;
const uint32_t qkBase = ((bos + tokenStart) * hg_ + kvHead) * k_;
```

与 Triton 的 `i_h // (H // Hg)` 逻辑完全一致。

---

## 4. AIC 端：三次 Cube MMA

### 4.1 Matmul 模板类型

```cpp
using QType  = MatmulType<TPosition::GM, CubeFormat::ND, bf16, false>;
using KTType = MatmulType<TPosition::GM, CubeFormat::ND, bf16, true>;   // transpose K
using HType  = MatmulType<TPosition::GM, CubeFormat::ND, bf16, false>;
using VType  = MatmulType<TPosition::GM, CubeFormat::ND, bf16, false>;
using AmaskType = MatmulType<TPosition::GM, CubeFormat::ND, bf16, false>;
using CFp32  = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;

matmul::MatmulImpl<QType, HType,    CFp32, BiasN> mm1_;   // Q @ H        -> c1
matmul::MatmulImpl<QType, KTType,   CFp32, BiasN> mm2_;   // Q @ K^T      -> c2
matmul::MatmulImpl<AmaskType, VType,CFp32, BiasN> mm3_;   // A_masked @ V -> c3
```

### 4.2 每 chunk 调度

```cpp
// MM1 (Q @ H)
mm1_.SetOrgShape(BT, BV, K, K, BV);
mm1_.SetSingleShape(curBT, curBV, K);
mm1_.SetTensorA(qGm[qkBase], false);
mm1_.SetTensorB(hGm[hBase], false);
mm1_.IterateAll<false>(c1Gm, /*sync=*/0);

// MM2 (Q @ K^T)
mm2_.SetTensorA(qGm[qkBase], false);
mm2_.SetTensorB(kGm[qkBase], true);   // transposeB at runtime as well
mm2_.IterateAll<false>(c2Gm, 0);

CrossCoreSetFlag<0x2, PIPE_FIX>(E_MM12_DONE);
CrossCoreWaitFlag(E_MASK_DONE);

// MM3 (A_masked @ V)
mm3_.SetTensorA(amskGm, false);
mm3_.SetTensorB(vGm[vBase], false);
mm3_.IterateAll<false>(c3Gm, 0);

CrossCoreSetFlag<0x2, PIPE_FIX>(E_MM3_DONE);
CrossCoreWaitFlag(E_STORE_DONE);
```

`IterateAll<false>` 的 `sync=0` 让 cube 异步发射，紧接着的 `CrossCoreSetFlag<0x2, PIPE_FIX>` 在 cube 写回完成后再触发事件，是利用 PIPE_FIX 保证 GM 可见性的标准做法。

### 4.3 Cube tile 选择（Host Tiling 决策）

在 `ChunkGatedDeltaRuleOTiling::ConfigMatmulTilings()` 中调用 `matmul_tiling::MultiCoreMatmulTiling`：

```cpp
mm.SetAType(GM, ND, bf16, false);
mm.SetBType(GM, ND, bf16, transposeB);
mm.SetCType(GM, ND, fp32);
mm.SetOrgShape(BT, BV/BT, K/BT);
mm.SetShape   (BT, BV/BT, K/BT);
mm.SetFixSplit(BT, BV/BT, K/BT);
mm.SetBufferSpace(L1, L0C, UB);
mm.GetTiling(tilingData_.mm1Tiling);
```

三次 MMA 各自独立完成 tiling。在典型形状（BT=64, BV=128, K=128）下，cube 一次 `IterateAll` 就处理一个完整 tile（M=64, N=128, K=128），刚好与 16×16×16 cube 基础块匹配，cube 利用率接近 100%。

---

## 5. AIV 端：门控、掩码、合成与写回

每个 AIV 处理 `myRows = halfBT` 行（BT 行的上半 / 下半）。流程：

```cpp
// 1) 等 AIC 写好 c1, c2
CrossCoreWaitFlag(E_MM12_DONE);

// 2) 把自己负责的行从 workspace 搬到 UB（fp32）
LocalTensor<float> c1 = LoadFp32(slotBase, c1Que_, myRows, alignV);
LocalTensor<float> c2 = LoadFp32(slotBase + wsBytesC1, c2Que_, myRows, alignBT);
LocalTensor<float> g  = LoadG(...);  // 仅 USE_G 时

// 3) 门控 + safe_exp + 因果 mask
ApplyGatingAndMask(c1, c2, g, myRows, curBT);
//   c1[r,:] *= exp(g[globalR])             -> Muls
//   diff[j] = clamp(g[globalR]-g[j], 0)    -> Adds + Muls(-1) + Mins(0)
//   c2[r,:] *= Exp(diff)                   -> Exp + Mul
//   c2[r, j>globalR] = 0                   -> Duplicate

// 4) Cast c2 (fp32) -> b_A_masked (bf16) 写回 workspace 给 AIC 用
CastAndStoreAmask(c2, slotBase + wsBytesC1 + wsBytesC2, myRows);

CrossCoreSetFlag<0x2, PIPE_MTE3>(E_MASK_DONE);

// 5) 等 AIC 算完 MM3
CrossCoreWaitFlag(E_MM3_DONE);

// 6) 取 c3，合成最终输出 = scale*c1 + scale*c3
LocalTensor<float> c3 = LoadFp32(slotBase + ..., c3Que_, myRows, alignV);
ComposeAndStore(c1, c3, ...);
//   bo = scale * c1
//   c3 = scale * c3
//   bo = bo + c3                           -> Muls + Muls + Add
//   out = Cast(bo, bf16, RINT)
//   DataCopyPad(out_gm, out, ...)

CrossCoreSetFlag<0x2, PIPE_MTE3>(E_STORE_DONE);
```

`safe_exp` 的向量化实现是相对 Triton 的一个关键加速点：Triton 在每对 `(i, j)` 上调用 `safe_exp` 的标量分支，本算子用一行 `Mins(diff_, 0)` 在向量层把上三角的正差值压成 0，再 `Exp` 一次性产出整行衰减系数。

---

## 6. UB 内存布局（每个 AIV）

按 BT=64, BV=128, K=128, halfBT=32, alignBT=64, alignV=128：

| 区域 | 类型 | 形状 | 字节数 |
|---|---|---|---|
| `c1Que_` | fp32 | `halfBT × alignV` = 32×128 | 16 KiB |
| `c2Que_` | fp32 | `halfBT × alignBT` = 32×64 | 8 KiB |
| `c3Que_` | fp32 | `halfBT × alignV` | 16 KiB |
| `amQue_` | bf16 | `halfBT × alignBT` | 4 KiB |
| `outQue_` ×2 | bf16 | `halfBT × alignV` | 16 KiB |
| `gQue_` (USE_G) | fp32 | `alignBT` | 256 B |
| **scratch (TBuf)** | | | |
| `gExp_` | fp32 | `alignBT` | 256 B |
| `diff_` | fp32 | `alignBT` | 256 B |
| `boFp_` | fp32 | `halfBT × alignV` | 16 KiB |
| `rowAcc_` | fp32 | `alignBT` | 256 B |
| **合计** | | | **≈ 77 KiB** |

每个 AIV 侧 UB 上限 192 KiB（910B），剩余 100+ KiB 可用于未来扩展（例如更大 BT、`outQue_` 加深双缓冲等）。

---

## 7. Workspace 布局

每个工作单元独占一段 workspace（**ping-pong 双 slot**），内层布局：

```
slot[i]:
  +---------------------------+  offset 0
  | c1   : BT * alignV * fp32 |  wsBytesC1
  +---------------------------+
  | c2   : BT * alignBT * fp32|  wsBytesC2
  +---------------------------+
  | amsk : BT * alignBT * bf16|  wsBytesAmaskBf16
  +---------------------------+
  | c3   : BT * alignV * fp32 |  wsBytesC3
  +---------------------------+
```

总 workspace 字节：

```
SYS_WORKSPACE (16 MiB) + GetLibApiWorkSpaceSize()
  + units * 2 * (wsBytesC1 + wsBytesC2 + wsBytesAmaskBf16 + wsBytesC3)
```

典型形状（BT=64, BV=128, B=1, T=2048, H=8, V=128）：
- wsBytesPerSlot = 32 KiB + 16 KiB + 8 KiB + 32 KiB = 88 KiB
- units = 1·8·1 = 8
- 用户 workspace = 8 × 2 × 88 KiB = 1.4 MiB

完全可控，远小于 vLLM 已经为 KV cache 分配的常驻显存。

---

## 8. 跨核同步事件设计

```
E_MM12_DONE  = 6  (AIC -> AIV)  // PIPE_FIX，cube 写完 c1/c2
E_MASK_DONE  = 7  (AIV -> AIC)  // PIPE_MTE3，AIV 写完 b_A_masked
E_MM3_DONE   = 8  (AIC -> AIV)  // PIPE_FIX，cube 写完 c3
E_STORE_DONE = 9  (AIV -> AIC)  // PIPE_MTE3，AIV 写完 out
```

事件 id 选 6/7/8/9 是为了避开运行时常用的 0~5 号。`CrossCoreSetFlag<0x2, ...>` 中的 `0x2` 表示 hard sync mode 2（对所有同 sub_block 的对端核生效），避免软同步开销。

`E_STORE_DONE` 在每 chunk 末尾触发，AIC 在 `MM1+MM2(t+2)` 之前必须等到 `STORE_DONE(t)`，从而保证 ping-pong slot 不被覆盖。这是一个粗粒度但足够的屏障，因为 cube 与 vector 的延迟在我们的 tile 选择下是平衡的。

---

## 9. Host Tiling 7 步规范

| 步骤 | 函数 | 关键内容 |
|---|---|---|
| 1. GetPlatformInfo | `InitCompileInfo` | 取 aicNum / aivNum / UB / L1 / L0A/B/C / SoC version |
| 2. GetShapeAttrsInfo | `AnalyzeDtype` + `AnalyzeShapes` + `GetScale` + `GetChunkSize` + `DetectOptionalInputs` | dtype/rank 校验、GQA 校验、scale 抓取、chunk_size 检查、optional input 探测 |
| 3. DoOpTiling | `PlanBlockDim` + `PlanWorkspaceLayout` | block dim 计算、ping-pong workspace 字节计算 |
| 4. DoLibApiTiling | `ConfigMatmulTilings` | **三次 `MultiCoreMatmulTiling::GetTiling()`** 填充 mm1/mm2/mm3 的 `TCubeTiling` |
| 5. GetTilingKey | bit0=USE_G, bit1=IS_VARLEN | 4 个 specialised binary，避免重复编译 |
| 6. GetWorkspaceSize | sys + libApi + units*ping-pong | 见第 7 节 |
| 7. PostTiling | SetBlockDim + memcpy_s 序列化 | 注意 mix 模式 block dim 是 **AIC 计数** |

---

## 10. 性能模型与对 Triton 的优势

设单 chunk 的耗时分解为：

- $T_{c1}$：MM1 = $BT\cdot K\cdot BV / \text{cube_throughput}$
- $T_{c2}$：MM2 = $BT\cdot K\cdot BT / \text{cube_throughput}$
- $T_{c3}$：MM3 = $BT\cdot BT\cdot BV / \text{cube_throughput}$
- $T_v$：vector 后处理 (Exp/Muls/Mins/Mul/Cast/DataCopyPad)

Triton 的串行模型大致是 $T_{c1}+T_{c2}+T_v+T_{c3}$，每 chunk 时间 ~`T1+T2+T3+Tv`。

本算子流水后理论时间 = $\max(T_{c1}+T_{c2}, T_v) + T_{c3}$，再叠加跨 chunk 的 ping-pong 重叠 = $\max(T_{c1}+T_{c2}+T_{c3}, T_v + T_{c3})$。当 $T_v \le T_{c1}+T_{c2}$ 时（常见情形，因为 vector 后处理的 FLOP 比 cube 少 1~2 个数量级），总时间被 cube 完全决定，**而 Triton 上 vector + sync 部分会形成尾巴**。

预估提速：
- 当 V≥128 且 BT=64 时，cube 利用率从 Triton 的 ~60% 提升到本算子的 ~90%（少了 vector 占用 cube 的窗口）
- 跨 chunk ping-pong 让 NT 越大、收益越大；典型 T=2048 → NT=32 时端到端约 **1.4 ~ 1.8×** 提速。

---

## 11. AscendC API 选型一览

> 全部对应 [《Ascend C API 列表》](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html) (CANN 商用版 8.5.0)。

### 11.1 资源管理 / Tiling 框架

| API | 用途 |
|---|---|
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` | 声明 1 AIC : 2 AIV 模式 |
| `ASCEND_IS_AIC` / `ASCEND_IS_AIV` | 编译期分支：AIC/AIV 半 |
| `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA` | 序列化/反序列化 TilingData |
| `TPipe`, `TQue<VECIN>`, `TQue<VECOUT>`, `TBuf<VECCALC>` | UB 资源管理（AIV 半使用） |
| `GetUserWorkspace(workspace)` | 取系统预留之外的用户 workspace |

### 11.2 Cube 高阶 API（AIC 半）

| API | 用途 |
|---|---|
| `MatmulType<TPosition, CubeFormat, dtype, transpose>` | 描述 Matmul 操作数类型 |
| `matmul::MatmulImpl<A,B,C,Bias>` | Cube Matmul 模板实例 |
| `matmul::MatmulImpl::Init(tiling, pipe)` | 用 Host 的 `TCubeTiling` 初始化 |
| `mm.SetTensorA / SetTensorB / SetSingleShape / SetOrgShape` | 配置一次 IterateAll |
| `mm.IterateAll<false>(cGm, 0)` | 异步发射 cube 计算并写回 GM |
| Host 侧 `matmul_tiling::MultiCoreMatmulTiling` + `GetTiling` | 自动产出 `TCubeTiling`，包括 `baseM/N/K`、`stepKa/Kb`、`depthA1/B1` |

### 11.3 向量计算（AIV 半）

| API | 用途 |
|---|---|
| `Cast(dst, src, RoundMode)` | bf16↔fp32 |
| `Duplicate(dst, scalar, len)` | 因果 mask 把上三角清零 |
| `Muls(dst, src, scalar, len)` | scale 应用 / 行级 `c1[r,:]*=exp(g[r])` |
| `Add(dst, s0, s1, len)` | `bo = scale·c1 + scale·c3` |
| `Mul(dst, s0, s1, len)` | `c2[r,:] *= safe_exp_diff` |
| `Adds(dst, src, scalar, len)` | `diff = g - g[r]` |
| `Mins(dst, src, scalar, len)` | safe_exp 上三角压成 0 |
| `Exp(dst, src, len)` | gExp = exp(g)，diff = exp(safe-clamped diff) |
| `LocalTensor::GetValue / SetValue` | 标量取值（仅 g[globalR] 等单点） |

### 11.4 数据搬运

| API | 用途 |
|---|---|
| `DataCopyPad(dst, src, ExtParams, PadParams)` | GM→UB / UB→GM 跨 stride 搬运 |
| `DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, rsv}` | 5 字段控制矩形块、行 stride |
| `DataCopyPadExtParams<T>{padFlag, padVal, padLen, rsv}` | 尾块 pad 到 32B 对齐 |

### 11.5 同步控制

| API | 用途 |
|---|---|
| `PipeBarrier<PIPE_V>` | 同 pipe 内有数据依赖时插同步 |
| `CrossCoreSetFlag<MODE, PIPE>(eventId)` | 跨核（AIC↔AIV）发事件，hard sync mode |
| `CrossCoreWaitFlag(eventId)` | 配对等待 |
| `TQue::EnQue / DeQue` | MTE2↔V / V↔MTE3 自动事件 |

### 11.6 系统变量

| API | 用途 |
|---|---|
| `GetBlockIdx()` | mix 模式下 AIC 半返回 AIC id，AIV 半返回 AIV id |
| `GetBlockNum()` | 总核数（含 AIC + AIV） |

---

## 12. 与 Triton 实现的逐项对比

| 维度 | Triton (`chunk_o.py`) | AscendC AIC+AIV |
|---|---|---|
| Cube 调度 | 由 Triton 编译器自动 | **由我们手工** 通过 `MatmulImpl` |
| 跨 chunk 重叠 | 编译器内 schedule，不可控 | **显式 ping-pong + 4 事件**，确定性 |
| 上三角 safe_exp | `tl.where(x<=0, x, -inf)` 标量分支 | 一行 `Mins(diff, 0)` 向量化压平 |
| 因果 mask | `tl.where(m_A, b_A, 0)` 全量 cast | `Duplicate(0)` 仅写零 tail，省一半带宽 |
| BT 行并行 | 单 program 串 BT 行 | **2 个 AIV** 各处理 BT/2 行 |
| GQA 处理 | `i_h // (H // Hg)` 同 | `hIdx / groupSize_` 同 |
| varlen / chunk_offsets | host 预构建透传 | 同（直接用 `cu_seqlens`/`chunk_offsets`） |
| 编译期分支 | `@triton.heuristics` USE_G/IS_VARLEN | TilingKey bit0/bit1 触发 4 套 binary |

---

## 13. 算子调用契约

```cpp
// Phase 1
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnChunkGatedDeltaRuleOGetWorkspaceSize(
    q, k, v_new, h, g_or_nullptr, cu_or_nullptr, co_or_nullptr,
    /*scale_value*/ 1.0f / std::sqrt((float)K),
    /*chunk_size */ 64,
    out, &workspaceSize, &executor);

void *workspace = nullptr;
if (workspaceSize > 0)
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

// Phase 2
aclnnChunkGatedDeltaRuleO(workspace, workspaceSize, executor, stream);
```

张量约束与第一个 PR 文档相同。

---

## 14. 参考

- [Ascend C API 列表（CANN 商用版 8.5.0）](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
- 本仓库示例：
  - `csrc/sparse_flash_attention/`（AIC + AIV 混合 + L1/L0 七 buffer 流水，更激进的版本）
  - `csrc/moe_grouped_matmul/`（高阶 `matmul::MatmulImpl` + `MultiCoreMatmulTiling` 用法）
  - `csrc/recurrent_gated_delta_rule/`（GDN 解码路径，AIV-only）
- Flash Linear Attention 上游：[`fla/ops/gated_delta_rule`](https://github.com/sustcsonglin/flash-linear-attention)
- 配套 Triton 设计文档：`docs/source/developer_guide/Design_Documents/chunk_o_triton.md`
- 配套 Excalidraw 流程图：`docs/source/developer_guide/Design_Documents/assets/chunk_o_flow.excalidraw`
