# `chunk_o.py` —— Gated DeltaNet 分块前向输出 Triton 算子详解

> 文件路径：`vllm_ascend/ops/triton/fla/chunk_o.py`
> 配套流程图：`docs/source/developer_guide/Design_Documents/assets/chunk_o_flow.excalidraw`
> 适用模型：Qwen3-Next、Qwen3.5 Hybrid 等使用 **Gated Delta-Rule (GDN)** 线性注意力的模型

本文档对 `chunk_o.py` 的 Triton 内核 `chunk_fwd_kernel_o` 与其 Python 包装器 `chunk_fwd_o` 进行系统性梳理，包括：算子在整张 GDN 计算图中的位置、数学推导、Triton 实现的并行划分与数据流、以及**逐行注释**。

---

## 1. 算子在 GDN 前向流水中的位置

`chunk_fwd_o` 是 `chunk_gated_delta_rule_fwd`（位于 `vllm_ascend/ops/triton/fla/chunk.py`）的**最后一个**主计算阶段，负责由前面阶段产出的中间量计算 GDN 注意力的最终输出 `o`。

```
chunk_local_cumsum(g)            -> 在每个 chunk 内累计 log-decay 得到 g_cumsum
chunk_scaled_dot_kkt_fwd(k,beta,g)-> 计算 chunk 内 KᵀK 加权矩阵 A
solve_tril(A)                    -> 求解下三角 (I - A)⁻¹，得到 WY 表示
recompute_w_u_fwd(k,v,beta,A,g)  -> 得到 w 与新 v(即 u)
chunk_gated_delta_rule_fwd_h     -> 沿时间维滚动，得到每个 chunk 起点的 SSM 隐藏状态 h, v_new
[可选 PCP] chunk_fwd_o_update    -> 多卡 PCP 时，融合跨卡更新
chunk_fwd_o(q, k, v_new, h, g)   -> 本算子：组合 inter-chunk + intra-chunk 得到 o
```

`o` 的形状与 `v` 完全一致，为 `[B, T, H, V]`。`chunk_fwd_o` 自身不做反向，对应在 `ChunkGatedDeltaRuleFunction.forward` 里调用。

调用关系（自上而下）：
- `vllm_ascend/ops/gdn.py::AscendGatedDeltaNet.forward` → `chunk_gated_delta_rule`
- `chunk_gated_delta_rule` (vllm-ascend 版本，用于在 NPU 上替换 vLLM 上游同名实现，见 `vllm_ascend/patch/worker/patch_triton.py`)
- → `ChunkGatedDeltaRuleFunction.apply` → `chunk_gated_delta_rule_fwd` → `chunk_fwd_o`

---

## 2. 数学原理

GDN 的核心思想是把序列切成 **chunk**（本算子里 `BT = 64`），先用一次"全局递推" `chunk_delta_h` 算出每个 chunk 起点的 SSM 状态 `H_t ∈ R^{K×V}`，然后 chunk 内部用一次小型注意力补齐余下的因果项。

对于第 `t` 个 chunk（`BT` 个 token），输出可以写成两项之和：

$$
O_t \;=\; \text{scale}\cdot \underbrace{e^{g_t}\odot Q_t H_t}_{\text{Inter-chunk: 历史状态贡献}} \;+\; \text{scale}\cdot \underbrace{\bigl(\, M\odot e^{g_t-g_t^{\top}}\odot Q_tK_t^{\top}\,\bigr)\,V_t}_{\text{Intra-chunk: 块内因果注意力}}
$$

其中：

- $Q_t \in R^{BT\times K}$，$K_t \in R^{BT\times K}$，$V_t \in R^{BT\times V}$ 为第 $t$ 个 chunk 的 q/k/v 切片；
- $H_t \in R^{K\times V}$ 为该 chunk **起始时刻**的 SSM 隐藏状态（由 `chunk_delta_h` 产出）；
- $g_t \in R^{BT}$ 为 chunk 内 log-gate 的累计值（由 `chunk_local_cumsum` 产出，已经在调用前被 `transpose(1,2).contiguous()` 重排为 `[B, H, T]`）；
- $M$ 为下三角因果掩码；
- $\odot$ 表示元素级相乘。

> **注意 1**：第一项的 `exp(g_t)` 之所以只乘在 inter-chunk 部分，而第二项乘的是 `exp(g_i - g_j)`（其中 $i\ge j$），是因为 chunk 内每个 token 看到 chunk 起点已经经过累计衰减 `exp(g_i)`，而看到 chunk 内更早 token 的衰减是 `exp(g_i - g_j)`。
>
> **注意 2**：`g_i - g_j`（$i\ge j$）一定非正，但浮点累计可能让上三角 ($i<j$) 取到正值导致 `exp` 溢出，因此用 `safe_exp`（见 `utils.py`）将正值预先压成 `-inf` 再 `exp`，得到 `0`，配合后面的因果掩码做"双保险"。

---

## 3. 张量约定 (shape / stride)

| 张量 | 形状 | 说明 |
|---|---|---|
| `q` | `[B, T, Hg, K]` | GQA Query，**Hg** 为 query head 数（按 KV 分组前的总数 / group size） |
| `k` | `[B, T, Hg, K]` | GQA Key |
| `v` | `[B, T, H, V]` | Value，**H** 为输出头数 |
| `h` | `[B, NT, H, K, V]` | 每个 chunk 起点的 SSM 状态，`NT = ceil(T/BT)` 或 varlen 的累计 chunk 数 |
| `g` | `[B, T, H]` → 转置为 `[B, H, T]` | chunk 内累计 log-gate |
| `cu_seqlens` | `[N+1]` | varlen 的累计长度 |
| `chunk_offsets` | `[N+1]` | 每条序列起始处在 `h` 第 1 维的偏移 |
| `o` | `[B, T, H, V]` | 输出，与 `v` 同形 |

> **关键 stride 关系**（用于内核里的 `make_block_ptr`）：`q`、`k` 沿 token 维的 stride 为 `Hg*K`；`v`、`o` 沿 token 维的 stride 为 `H*V`。`h` 在 `(K,V)` 子矩阵内的 stride 为 `(V, 1)`。

---

## 4. 并行划分

`grid = (cdiv(V, BV), N*H)`，其中 `BV = 128`，`BK = 128`，`BT = 64`：

- **`program_id(0) = i_v`**：负责输出 V 维的第 `i_v` 个 tile（宽 `BV`）。
- **`program_id(1) = i_nh`**：被进一步拆为 `i_n = i_nh // H`（序列号）和 `i_h = i_nh % H`（输出 head 号）。

每个 program 负责"**某条序列、某个 head、某个 V tile**"的所有 chunk。在 program 内串行循环 `i_t = 0..NT-1`；对 K 维再拆 `i_k = 0..ceil(K/BK)-1` 做 reduction。这样设计的好处：

1. 每个 program 把一段时间维上的累加都放在寄存器里（`b_o`、`b_A` 是寄存器累加器），减少 HBM 写回；
2. 不同 chunk 的 inter-chunk 项 `Q_t @ H_t` 之间没有依赖，所以可以串行高效访问而不需要更复杂的 grid 设计；
3. `BV=BK=128, BT=64` 在 Ascend NPU 上 cube 单元利用率较高。

---

## 5. 内核数据流 (一次 program 的执行)

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ program_id = (i_v, i_nh)                                             │
 │ i_n, i_h  = i_nh // H, i_nh % H                                      │
 │ 1) 解析 (bos, eos, T_this, NT, boh)                                  │
 │ 2) 把 q/k/v/o 的 base ptr 调到「序列 i_n、head i_h 起始 token」      │
 │                                                                      │
 │  for i_t in range(NT):           ← 沿时间维串行                      │
 │     b_o = 0  (BT,BV)             ← inter-chunk 累加器               │
 │     b_A = 0  (BT,BT)             ← intra-chunk QKᵀ 累加器           │
 │                                                                      │
 │     for i_k in range(K/BK):      ← K 维 reduction                   │
 │         load Q_t [BT,BK]                                             │
 │         load K_tᵀ [BK,BT]                                            │
 │         load H_t  [BK,BV]                                            │
 │         b_o += Q_t @ H_t         ← cube MMA, fp32 累加              │
 │         b_A += Q_t @ K_tᵀ        ← cube MMA, fp32 累加              │
 │                                                                      │
 │     if USE_G:                                                        │
 │         load b_g [BT]                                                │
 │         b_o *= exp(b_g)[:,None]                                      │
 │         b_A *= safe_exp(b_g[:,None] - b_g[None,:])                   │
 │                                                                      │
 │     causal mask: b_A = where(i>=j, b_A, 0)                           │
 │                                                                      │
 │     load V_t [BT,BV]                                                 │
 │     b_o = scale*b_o + scale*(b_A @ V_t)                              │
 │     store o[BT,BV]                                                   │
 └──────────────────────────────────────────────────────────────────────┘
```

整体上，单个 program 的 HBM I/O：

- **读**：每个 chunk 读一次 `Q_t`、`K_t`、`H_t`、`V_t` 切片（沿 K 维分多次）、一次 `b_g`；
- **写**：每个 chunk 写一次 `o[BT,BV]` 切片。

`H_t` 是 `[K,V]` 矩阵，因此每跨一个 chunk 都会换 `H_t`，但同一 program 内 `K` 维 tile 会被复用。

---

## 6. 逐行注释

下面把 `chunk_o.py` **完整代码**按行剖析。每一行解释"为什么这么写"。

### 6.1 头部 import 与装饰器

```13:17:vllm_ascend/ops/triton/fla/chunk_o.py
import torch
from vllm.triton_utils import tl, triton

from .utils import prepare_chunk_offsets, safe_exp
```

- `from vllm.triton_utils import tl, triton`：通过 vLLM 提供的 `triton_utils` 入口取得 `tl`/`triton`。这层封装能在没有 GPU/Triton 时退化成空 stub，避免 import 失败；并便于在 Ascend 上接管 Triton 后端。
- `prepare_chunk_offsets`：把 `cu_seqlens` 转成 chunk 起点的累计偏移 `[N+1]`，让内核能在 varlen 情况下用 `O(1)` 查到该序列在 `h` 中的起点。
- `safe_exp`：见前文，对 `>0` 的输入返回 `exp(-inf)=0`，专为 `g_i - g_j` 上三角防溢出而设计。

### 6.2 `@triton.heuristics`

```19:24:vllm_ascend/ops/triton/fla/chunk_o.py
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
```

- **为什么用 heuristics 而不是写在签名里？** Triton 的 heuristics 会根据**运行时**参数自动派生 `tl.constexpr` 标志位，进而触发不同的特化版本。这两个布尔标志会改变 kernel 的代码路径（`if USE_G`、`if IS_VARLEN`），把它们做成 `constexpr` 可以让编译器把分支彻底消除，生成两套（实际 4 套）专门优化的二进制，避免运行时分支。

### 6.3 `@triton.jit(do_not_specialize=...)`

```25:25:vllm_ascend/ops/triton/fla/chunk_o.py
@triton.jit(do_not_specialize=["chunk_offsets", "scale", "T", "H", "Hg", "K", "V"])
```

- 默认情况下 Triton 会按"参数的具体数值"生成特化版本。**对于 `T/H/Hg/K/V` 这种会随请求变化的标量，如果让它们参与特化，会引发频繁重新编译**（在 vLLM 推理里每个请求长度都不同，会非常昂贵）。把它们列入 `do_not_specialize` 后，它们以普通 runtime 标量传入；性能差异可忽略，但显著降低首请求的编译开销。
- `chunk_offsets` 是张量指针，写在这里只是显式声明不要根据指针地址特化。
- 这一行是 vllm-ascend 在 **PR #7482 / #7483 / #7481** 中针对启动时长做的关键优化。

### 6.4 内核入口与 grid 解析

```26:48:vllm_ascend/ops/triton/fla/chunk_o.py
def chunk_fwd_kernel_o(
    q, k, v, h, g, o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T, H, Hg, K, V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
```

- `i_v` 取 `program_id(0)`：当前 program 处理输出 V 维的第 `i_v` 个 tile（每 tile 宽度 `BV`）。
- `i_nh` 取 `program_id(1)`：把"序列号 × head 号"打平，`i_n = i_nh // H`，`i_h = i_nh % H`。这样 grid 的第二维只有 `N*H` 这一个数，方便启动；同时避免了三维 grid 在某些后端上的对齐限制。

```49:49:vllm_ascend/ops/triton/fla/chunk_o.py
    T_max = T
```

- `T_max` 备份**最大序列长度**（即 padded 维度）。下面在 varlen 分支会把 `T` 覆盖为本序列实际长度，但 `g_ptr` 计算时仍然要按 padded 后的维度跨步（因为 `g` 一直是按最大长度分配的张量），所以必须先把 `T` 备份。这是 varlen 场景下最容易踩坑的细节。

### 6.5 varlen 与 fixed-len 两条分支

```51:59:vllm_ascend/ops/triton/fla/chunk_o.py
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
```

- **varlen 分支**：从 `cu_seqlens` 读出本序列的 `[bos, eos)`，用 `int32` 是因为序列长度永远不会爆 32-bit；`boh` 是 `chunk_offsets[i_n]`，告诉我们"本序列的第 0 个 chunk 在 `h` 张量第 1 维的位置"，必须用 `int64`，因为 `(boh * H + i_h) * K * V` 会很大。
- **fixed-len 分支**：所有序列等长，`bos = i_n * T`，`boh = i_n * NT` 直接计算，无需查表。
- 把两个分支以 `IS_VARLEN: tl.constexpr` 区分后，编译器会把整段 `if/else` 消除掉，只保留实际生效的那一半代码，无运行时分支。

### 6.6 把 q/k/v/o 指针推进到本"序列+head"的起点

```61:65:vllm_ascend/ops/triton/fla/chunk_o.py
    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
```

这四行是 GQA 适配的核心：

- `q/k` 是 `[B, T, Hg, K]`，`Hg` 通常 < `H`，多个输出 head 共享同一个 KV head。`H // Hg` 是 group size，`i_h // (H // Hg)` 把"输出 head 号"映射成"KV head 号"。比如 `H=8, Hg=4` 时 group size 为 2，输出 head `0,1` 共享 KV head 0，`2,3` 共享 KV head 1，依此类推。
- 加上 `bos * Hg` 把指针推到本序列起始 token；再加 `i_h//(H//Hg)` 选 KV head。乘 `K` 推到该 head 的特征起点。
- `v`、`o` 是 `[B, T, H, V]`，每个 head 都独占一份，所以直接 `bos*H + i_h`。

> 这里**直接修改了 kernel 形参 `q/k/v/o` 的本地指针变量**，后面的 `make_block_ptr` 都基于新的 base 指针，相当于在 program 内做了"逻辑切片"。

### 6.7 chunk 主循环

```67:72:vllm_ascend/ops/triton/fla/chunk_o.py
    for i_t in range(NT):
        i_tg = boh + i_t
        h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
```

- `i_tg` 是 chunk 在**全局 chunk 维**上的索引（跨序列累计后的 chunk 号）。
- `h_base` 把 `h` 指针推到 `(i_tg, i_h)` 这块 `[K,V]` 子矩阵的起点。`(i_tg * H + i_h)` 必须 `to(int64)`，因为 `K*V` 可能很大（如 `K=V=128` 时单块 16K 元素），全局乘积容易超过 int32 范围。
- `b_o` / `b_A`：用 fp32 在寄存器里累加。**不在 K-tile 循环外面用 bf16 累加是必要的**，否则会有显著精度损失；输出阶段再 cast 回 `o` 的 dtype。

### 6.8 K 维 tile 循环：inter-chunk 与 intra-chunk 同时累加

```73:87:vllm_ascend/ops/triton/fla/chunk_o.py
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1))

            # [BT, BK] @ [BK, BV] -> [BT, BV]
            b_o += tl.dot(b_q, b_h)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)
```

- `make_block_ptr(base, shape, strides, offsets, block_shape, order)`：
  - `p_q`：从已偏移的 `q` 起点出发，逻辑形状 `(T, K)`、stride `(Hg*K, 1)`，定位到第 `i_t` 个 chunk × 第 `i_k` 个 K-tile，块大小 `(BT, BK)`，行优先 (`order=(1,0)`)。
  - `p_k`：注意这里逻辑形状写成 `(K, T)`、stride `(1, Hg*K)`、`order=(0,1)` —— 这是**通过 stride 翻转来"在线转置"**，使加载下来的 `b_k` 直接是 `[BK, BT]`，省掉一次 `tl.trans`。`b_q @ b_k` 直接得到 `[BT, BT]` 的 QKᵀ 分块。
  - `p_h`：在 `h_base` 中按 `(i_k*BK, i_v*BV)` 选 `[BK, BV]` 子块，对应 SSM 隐藏状态 `H_t` 的 (K-tile, V-tile)。

- `boundary_check=(0, 1)`：两个轴都打开越界检查，越界部分自动当 0；这能正确处理 `T` 不是 `BT` 整数倍、`K`/`V` 不是 `BK`/`BV` 整数倍的尾块。

- 两条 `tl.dot`：
  - `b_o += b_q @ b_h`：逐 K-tile 累加 `Q_t @ H_t` —— **inter-chunk 项**；
  - `b_A += b_q @ b_k`：逐 K-tile 累加 `Q_t @ K_tᵀ` —— **intra-chunk QK 分数**。
  - 两者**共享同一个 K-tile 的 `b_q` 加载**，是这段代码最关键的复用：每读一次 `b_q` 就同时给两个 MMA 用，能把寄存器压力下的访存量减半。

### 6.9 USE_G：施加门控衰减

```89:96:vllm_ascend/ops/triton/fla/chunk_o.py
        if USE_G:
            offs_t = i_t * BT + tl.arange(0, BT)
            mask_t = offs_t < T
            g_ptr = g + bos + i_h * T_max
            b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])
```

- `offs_t`：本 chunk 内 `BT` 个 token 在序列内的全局位置；`mask_t` 用于过滤越界。
- `g_ptr = g + bos + i_h * T_max`：注意 `g` 已经在 host 端做了 `transpose(1,2).contiguous()` → `[B, H, T]`，所以同一 head 内 token 沿 stride=1 连续，跨 head 的 stride = `T_max`（等于 padded 维度上的 `T`）。这就是为什么必须用 `T_max` 而不是被 varlen 覆盖后的 `T`。
- `b_o * exp(b_g)[:,None]`：把第一项里"chunk 起点到当前 token 的累计衰减" `e^{g_i}` 乘到 inter-chunk 输出上。
- `b_A * safe_exp(b_g[:,None] - b_g[None,:])`：把"j → i 之间的累计衰减" `e^{g_i - g_j}` 乘到 QK 分数上。
- `safe_exp` 而不是 `tl.exp`：上三角 (`i<j`) 的差值可能为正且数值很大，直接 `exp` 会溢出/产生 NaN。`safe_exp` 把所有正值预先换成 `-inf`，`exp(-inf)=0`，安全地零掉这部分；下一行的因果掩码会再次确认零。

### 6.10 因果掩码、加载 V、合成最终输出

```98:109:vllm_ascend/ops/triton/fla/chunk_o.py
        o_i = tl.arange(0, BT).to(tl.float32)
        m_A = o_i[:, None] >= o_i[None, :]
        b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        # to fix mma -> mma layout conversion
        # already solved by fla v3.2 or higher
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
```

- `o_i = arange(0, BT)` 的 `to(float32)` 不是为了精度（索引本身用整型即可），而是因为某些 Triton 后端对**布尔比较的输入类型**有要求；用 fp32 比较生成的 mask 在后续 `tl.where` 里更稳定。
- `m_A`：标准下三角因果掩码，包含对角（同位置自己看自己）。
- `b_A = where(m_A, b_A, 0)`：把上三角清零。`safe_exp` 已经把上三角变成 `0×exp(...)=0` 了，但 `b_A` 在 `Q_t@K_tᵀ` 阶段是 fp32 全量计算，理论上上三角并非零；这一句是**正式且必须**的因果约束。
- `p_v` / `p_o` 用与 `v/o` 相同的 stride `(H*V, 1)` 切出 `(BT, BV)` 块。
- **关键合成行**：`b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale`
  - 先把 `b_A`（fp32）cast 回 `b_v.dtype`（一般是 bf16），是为了让 `tl.dot` 走 cube 单元的 bf16 MMA 通路。
  - 把 inter-chunk（`b_o`）和 intra-chunk（`b_A @ b_v`）都乘上 `scale = 1/√K`，相加得到最终 `O_t`。
  - 注释 `to fix mma -> mma layout conversion / already solved by fla v3.2 or higher` 是 FLA 上游的历史遗留：早期 Triton 在两次 MMA 之间需要这种显式 cast 以避免布局转换 bug；新版 FLA 已修复，但留作兼容。
- `tl.store(p_o, b_o.to(p_o.dtype.element_ty), ...)`：把 fp32 的累加结果 cast 回输出 dtype（bf16）写回 HBM。`boundary_check` 处理尾块越界。

### 6.11 Python 包装器 `chunk_fwd_o`

```112:163:vllm_ascend/ops/triton/fla/chunk_o.py
def chunk_fwd_o(
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
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N = len(cu_seqlens) - 1
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    g = g.transpose(1, 2).contiguous()
    chunk_fwd_kernel_o[grid](
        q=q, k=k, v=v, h=h, g=g, o=o,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T, H=H, Hg=Hg, K=K, V=V,
        BT=BT, BK=128, BV=128,
        num_warps=4,
        num_stages=2,
    )
    return o
```

- `B, T, Hg, K, V = *q.shape, v.shape[-1]`：用 unpack 一次性取出形状，然后用 `v.shape[-2]` 拿到输出 head 数 `H`（区别于 `q` 的 `Hg`）。
- `scale = k.shape[-1] ** -0.5`：默认 `1/√K`，与 Transformer 注意力惯例一致。
- `o = torch.empty_like(v)`：**直接按 `v` 的 shape/dtype 分配**，与内核里的 `(H*V, 1)` stride 假设一致。
- `cu_seqlens` 处理：fixed-len 时 `N=B`、不需要 `chunk_offsets`；varlen 时 `N = len(cu_seqlens)-1`，并在没有传入 `chunk_offsets` 时即时构造（`prepare_chunk_offsets(cu_seqlens, BT)` 实质是 `cumsum(cdiv(seqlens, BT))`）。
  - **优化要点**：当上层（`gdn_chunk_meta`）已经预构建了 `chunk_offsets`，直接传入可避免每个请求都做一次 host 上的 cumsum + H2D 拷贝（这是 PR #7487/PR #6830 的关键路径优化）。
- `grid` 闭包用 `meta["BV"]` 而不是常数 128，是为了如果将来通过 `autotune` 改 `BV`，grid 自动跟随。
- `g = g.transpose(1, 2).contiguous()`：把 `[B, T, H]` 重排为 `[B, H, T]`，**让同一 head 内的 token 在内存上连续**，对应内核里 `g_ptr = g + bos + i_h * T_max` 的偏移计算。`.contiguous()` 保证物理 stride 与逻辑一致。
- 启动参数：
  - `BK=128, BV=128`：与 Ascend cube 单元 16×16×16 / 128×128 tile 大小匹配；
  - `num_warps=4`：每个 program 4 个 warp，平衡寄存器与并行度；
  - `num_stages=2`：双缓冲流水线，让 K-tile 加载与计算重叠。

---

## 7. 性能与实现要点小结

1. **K/V tile 复用**：同一 program 在一个 chunk 内，每读一次 `b_q` 就同时驱动两次 MMA（`b_o` 与 `b_A`），是计算密度的核心。
2. **`b_k` 的 stride trick**：通过 `make_block_ptr` 用 `(1, Hg*K)` + `order=(0,1)` 直接读到 `[BK, BT]`，省去 `tl.trans`。
3. **fp32 累加 + bf16 存储**：避免长累加链上的精度损失，又保证带宽与显存友好。
4. **`do_not_specialize` 形参**：避免 vLLM 不同请求长度导致 Triton 反复重编译，是首请求时延的关键优化。
5. **heuristics 派生 constexpr 分支**：将 `USE_G`、`IS_VARLEN` 编进二进制，运行期无分支开销。
6. **`chunk_offsets` 上层预构建**：在 `gdn_chunk_meta` 阶段一次性算好后透传，避免每层、每请求重复构造。
7. **`safe_exp` 双保险**：既防 `exp` 上三角溢出，又配合显式因果掩码确保正确性。

---

## 8. 配套流程图

可直接在 https://excalidraw.com 上 *Open → Load* 加载下面的文件，得到本算子完整执行流的可编辑流程图：

`docs/source/developer_guide/Design_Documents/assets/chunk_o_flow.excalidraw`

### 8.1 流程图结构说明

整张图按从上到下、从左到右组织为四个层次（与本文档第 1、4、5、6 节一一对应）：

1. **Caller 上下文层（最上）**
   描绘 `chunk_fwd_o` 在整条 GDN 流水中的位置：从 `chunk_local_cumsum` 一路到 `chunk_gated_delta_rule_fwd_h`，最后产出 `q, k, v_new, h, g` 五个张量喂入本算子。

2. **Host 包装器层**
   展示 `chunk_fwd_o(q,k,v,h,g,...)` 在 Python 侧做的 4 件事：解析形状/scale、决定 N 与 `chunk_offsets`、`g.transpose(1,2).contiguous()`、按 grid `(cdiv(V,BV), N*H)` 启动 Triton 内核。

3. **Kernel 控制流层（核心）**
   - 入口节点：`program_id = (i_v, i_nh)`，分解出 `i_n, i_h`、备份 `T_max`；
   - **菱形决策** `IS_VARLEN`，分出"varlen"与"fixed-len"两条小分支并合并；
   - 指针偏移节点：把 `q/k/v/o` 推进到本序列+head 起点（GQA 映射 `i_h // (H//Hg)`）；
   - 外层时间循环 `for i_t in range(NT)` 矩形容器，内部嵌套：
     - 累加器初始化 `b_o, b_A`；
     - 内层 K-tile 循环 `for i_k in range(K/BK)`：3 次 load + 2 次 MMA（其中 `Q_t@H_t` 与 `Q_t@K_tᵀ` 共享 `b_q`，用一条粗箭头示意复用）；
     - **菱形决策** `USE_G`，分出"加载 g、施加 exp(g) 与 safe_exp"分支；
     - 因果掩码节点；
     - 加载 `V_t`、合成 `b_o = scale·b_o + scale·(b_A·V_t)`；
     - 写回 `o[BT,BV]`。

4. **数学公式与张量形状层（最右侧）**
   将
   `O_t = scale·exp(g_t)⊙Q_tH_t + scale·(M⊙exp(g_i-g_j)⊙Q_tK_tᵀ)V_t`
   及关键张量 shape (`q [B,T,Hg,K]` / `v,o [B,T,H,V]` / `h [B,NT,H,K,V]` / `g [B,H,T]`) 单独成块，放在右侧作为对照表。

### 8.2 图例约定

- **矩形（白底）**：顺序计算节点；
- **菱形（白底）**：分支判断（`IS_VARLEN`、`USE_G`）；
- **圆角矩形（浅黄背景）**：循环容器（外层 chunk 循环、内层 K-tile 循环）；
- **平行四边形（浅蓝背景）**：HBM 读 / 写节点（load/store）；
- **粗箭头**：数据流（张量在节点间的传递）；
- **细箭头**：控制流（执行顺序）；
- **虚线箭头**：跨层引用（例如"上层 `gdn_chunk_meta` 提前构建 `chunk_offsets`"）。

通过这张图，可以一眼看出：

- 本算子的两次 MMA 是如何**共享同一个 `b_q` 加载**的；
- `g` 的内存布局是怎么从 `[B,T,H]` 转成 `[B,H,T]` 并被 `g_ptr = g + bos + i_h*T_max` 索引的；
- varlen 与 fixed-len 路径的差异只在 `bos/eos/boh` 的来源；
- 因果性是通过 `safe_exp + 显式 mask` 双重保证的。

---

## 9. 参考

- [vLLM Ascend PR #4070] Triton chunk_gated_delta_rule ops for Qwen3-Next（首次落地）
- [vLLM Ascend PR #7487 / #6830] 通过预构建 chunk metadata 减少 host-device 同步
- [vLLM Ascend PR #7482 / #7483 / #7481] kernel 重编译优化（`do_not_specialize`）
- [Flash Linear Attention 上游] [`fla/ops/gated_delta_rule`](https://github.com/sustcsonglin/flash-linear-attention)
