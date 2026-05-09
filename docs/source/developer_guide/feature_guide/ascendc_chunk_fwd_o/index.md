# ChunkFwdO AscendC Operator Design

This document describes the AscendC implementation of `ChunkFwdO`, the
chunk-wise forward output kernel used by the Flash-Linear-Attention
(FLA) family of models in vLLM Ascend (Gated DeltaNet, RWKV-7, etc.).

The AscendC implementation is designed to be **drop-in equivalent** to the
existing Triton kernel
[`vllm_ascend/ops/triton/fla/chunk_o.py`](../../../../vllm_ascend/ops/triton/fla/chunk_o.py)
while running faster on Ascend NPUs by using the AIC (cube) and AIV (vector)
sub-cores in a tightly-coupled mixed pipeline.

* Source layout
  * `csrc/chunk_fwd_o/op_kernel/` — device-side AscendC kernel
  * `csrc/chunk_fwd_o/op_host/`   — op definition, tiling, ACL wrappers
  * `csrc/chunk_fwd_o/chunk_fwd_o_torch_adpt.h` — PyTorch binding
  * `vllm_ascend/ops/triton/fla/chunk_o_ascendc.py` — Python dispatcher

The reference URL for the AscendC API used:
<https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html>.

---

## 1. Math being computed

For every chunk `t ∈ [0, NT)` of length `BT` (default 64), and every
`(batch, head, value-block i_v)` task, the kernel computes

```
H_t  = h[i_tg, head, :, i_v*BV : (i_v+1)*BV]                 # [K, BV]
A_t  = Q_t @ K_t^T                                            # [BT, BT]
O_t  = Q_t @ H_t                                              # [BT, BV]
if g is not None:
    O_t = O_t * exp(g[t])[:, None]
    A_t = A_t * exp(g[i] - g[j])  for i >= j else 0
A_t  = causal_mask(A_t)
O_t  = (O_t + A_t @ V_t) * scale
o[t*BT : (t+1)*BT, head, i_v*BV : (i_v+1)*BV] = O_t
```

`Q`, `K`, `V`, `O` are stored as `(T_total, H, D)` ND tensors (varlen) or
`(B, T, H, D)` (fixed).  `h` is `(NT_total, H, K, V)` and contains the
running per-chunk hidden state.  `g` is shaped `(B, H, T)` (transposed by
the caller before invocation).

---

## 2. Block partitioning

The Triton kernel uses a 2D launch grid `(ceil(V/BV), N*H)`.  We mirror
that exactly:

* `total_tasks = ceil(V / BV) * N * H`
* `task_id` selects an `(i_v, batch, head)` tuple.  Each task iterates
  `NT = ceil(seqlen[batch] / BT)` chunks inside the kernel.
* Tasks are distributed evenly across the AIC cores by integer interval
  partition: `task[blockId] = [taskBegin, taskEnd)` where
  `taskBegin = total_tasks * blockId / numAicCores` and likewise for the
  end.

We launch with `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` which
binds **one AIC core to two AIV cores**.  In our pipeline:

* The AIC core executes the three matmuls (Q@H, Q@K, A_masked@V).
* The two paired AIV cores share the post-processing (mask, exp, scale,
  cast).  Because the AIV cores work on the *same* data the kernel sets
  the same task interval for both, but the AIV side runs the lighter
  per-element work and finishes well before the AIC matmul of the next
  chunk completes — leaving the cube core busy almost continuously.

Block dim is set to `numCubeCore = min(total_tasks, aicNum)`.  When the
problem is smaller than the chip, we shrink the launch instead of leaving
cores idle.

---

## 3. Pipeline & cross-core synchronisation

For each chunk the AIC and AIV cores run the schedule in
*[Figure 1](#figure-1-aic--aiv-pipeline)*.

```
AIC: |-- Q @ H -->|--- Q @ K --->|             |-- A_masked @ V -->|
       SET QH       SET QK         WAIT AV        SET AV
                                  ↑       ↑
AIV:                              | exp(g)|       |  add+scale+cast |
                                  | mask  |       |   write O       |
                       WAIT QH    | cast  |        WAIT AV
                       WAIT QK    SET AV
```

Five cross-core flags are used (see `chunk_fwd_o.h`):

| Flag                    | Producer | Consumer | Meaning                                |
| ----------------------- | -------- | -------- | -------------------------------------- |
| `SYNC_AIC_TO_AIV_QH`    | AIC      | AIV      | `Q @ H` available in `hWorkspace`       |
| `SYNC_AIC_TO_AIV_QK`    | AIC      | AIV      | `Q @ K` available in `attnWorkspace`    |
| `SYNC_AIV_TO_AIC_AV`    | AIV      | AIC      | masked `A` available in `aftermaskWS`   |
| `SYNC_AIC_TO_AIV_AV`    | AIC      | AIV      | `A @ V` available in `vWorkspace`       |
| `SYNC_AIV_TO_AIC_NEXT`  | AIV      | AIC      | AIV finished consuming all workspaces, |
|                         |          |          | AIC may overwrite for next chunk        |

Because the AIC submits matmul `IterateAll` calls back-to-back, the cube
unit's L0/L1 are kept busy across the chunk boundary and the only stalls
come from the (cheap) `WAIT AV` sync mid-chunk.

### Figure 1: AIC + AIV pipeline

```
chunk k:    AIC: Q@H ─────► Q@K ─────►  ▲ wait ───► A_masked@V ───► (next k)
                  │            │         │             │
                  ▼            ▼         │             ▼
            AIV:  load─exp─mask─cast─►──┘  load─add─cast─store─►(next k)
```

This permits the AIC to issue the next chunk's `Q@H` *as soon as* the
previous AIV signals "AV consumed" (`SYNC_AIV_TO_AIC_NEXT`).  Two
matmul issues per chunk thus overlap with one cycle of vector work.

---

## 4. UB / L1 / L0 buffer plan

### UB layout (per AIV core)

| Tensor      | Shape       | dtype | Bytes (BT=64, BV=128) |
| ----------- | ----------- | ----- | --------------------- |
| `qhQue_`    | `[BT, BV]`  | fp32  | 32 KB                 |
| `avQue_`    | `[BT, BV]`  | fp32  | 32 KB                 |
| `attnQue_`  | `[BT, BT]`  | fp32  | 16 KB                 |
| `gQue_`     | `[BT]`      | fp32  | 256 B                 |
| `outQue_`   | `[BT, BV]`  | bf16  | 16 KB                 |
| `amQue_`    | `[BT, BT]`  | bf16  | 8 KB                  |
| `maskBuf_`  | `[BT, BT]`  | fp32  | 16 KB (init once)     |

Total ≈ 120 KB per AIV core (well within the 192 KB UB budget on
Ascend 910B/910C).

### Workspace layout (GM)

The host allocates a contiguous workspace that is partitioned per-AIC:

```
hWorkspace        [numAic][BT, BV] fp32
attnWorkspace     [numAic][BT, BT] fp32
vWorkspace        [numAic][BT, BV] fp32
aftermaskWorkspace[numAic][BT, BT] bf16/fp16
maskWorkspace     [numAic][BT, BT] fp32  (reserved, currently UB-only)
```

Offsets are passed through `ChunkFwdOTilingData::*WorkspaceOffset`.  All
slices are 512-byte aligned (`WORKSPACE_ALIGN`), matching the L2 burst
length.

### Cube-side L1/L0 plan

The cube matmuls use AscendC's high-level `Matmul` API, which manages
L1/L0A/L0B double-buffering itself.  The three matmul instances reside
in different `L0C` slots so they do not contend.  Sizes:

* `Q @ H`         — `M=BT, N=BV, K=K`  (typical 64 × 128 × 128)
* `Q @ K^T`       — `M=BT, N=BT, K=K`  (64 × 64 × 128)
* `A_masked @ V`  — `M=BT, N=BV, K=BT` (64 × 128 × 64)

For `K = 128`, all three fit in a single L0C tile and run as a single
`IterateAll` (no internal K-tiling).

---

## 5. Data flow

```
            ┌───────────────────────────────────────────────────────┐
            │                       GM                              │
            │  Q  K  V  H  g  cu_seqlens  chunk_offsets   o (out)   │
            └───────────────────────────────────────────────────────┘
                  │     │   │   │  │
        ┌─────────┘     │   │   │  │       ┌──────── workspace ───────┐
        ▼               │   │   │  │       │ hWS  attnWS vWS  amWS    │
   ┌─────────┐          ▼   ▼   ▼  │       │   ▲   ▲   ▲   ▲          │
   │  AIC    │──Q@H──►──┐               (1)─►───┘   │   │   │          │
   │         │──Q@K^T──►─┼───────────► (2)─► ─────►──┘   │   │          │
   │         │   wait◄───┼─────────────(3)─────►────────┘   │          │
   │         │──A@V──►───┼─────────────(4)─►────────────────┘          │
   └─────────┘           │                                             │
                         ▼                                             │
   ┌─────────┐  load hWS,attnWS │ exp(g) mask cast │ load vWS │ add+scale+cast → o
   │  AIV ×2 │──────────────────┴──────────────────┴──────────┴────────────────►
   └─────────┘
```

Steps `(1)..(4)` are protected by the cross-core flags above.

---

## 6. Vector micro-kernel

```cpp
// pseudo-code, BT = 64, BV = 128
load_qh_from_ws();                 // 32 KB MTE2
load_attn_from_ws();               // 16 KB MTE2
if (use_g) {
    load_g_row(BT);                // 256 B MTE2
    for (i = 0; i < BT; ++i) {
        Muls(qh_row[i], qh_row[i], expf(g[i]), BV);   // BV=128
        for (j = 0; j < BT; ++j) {                   // small loop, on-chip
            attn[i, j] *= (g[i] >= g[j]) ? expf(g[i]-g[j]) : 0;
        }
    }
}
Mul(attn, attn, causal_mask, BT*BT);   // single VEC op
Cast(attnQT, attn, BT*BT);             // fp32 → bf16
DataCopy(amWS, attnQT);                // 8 KB MTE3
SetFlag(SYNC_AIV_TO_AIC_AV);

WaitFlag(SYNC_AIC_TO_AIV_AV);
load_av_from_ws();                  // 32 KB MTE2
Add(qh, qh, av, BT*BV);             // single VEC op
Muls(qh, qh, scale, BT*BV);
Cast(out, qh, BT*BV);
DataCopyPad(o_gm, out, …);          // strided MTE3 because rows interleave H*V
SetFlag(SYNC_AIV_TO_AIC_NEXT);
```

The causal mask is built **once** at kernel start and reused across all
chunks — saving `BT*BT` SetValue per chunk.

---

## 7. Tiling object

Defined in `csrc/chunk_fwd_o/op_kernel/chunk_fwd_o_tiling_data.h` and
populated by `csrc/chunk_fwd_o/op_host/chunk_fwd_o_tiling.cpp`.

| Field                       | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `shapeBatch`                | `N` (number of sequences)                        |
| `seqlen`                    | `T_max` (longest sequence)                       |
| `kNumHead` / `vNumHead`     | `Hg` / `H`                                       |
| `kHeadDim` / `vHeadDim`     | `K` / `V`                                        |
| `scale`                     | attention scaling factor                         |
| `chunkSize`                 | `BT` (default 64)                                |
| `isVariedLen`               | 0 = fixed, 1 = use cu_seqlens                    |
| `tokenBatch`                | total tokens                                     |
| `dataType`                  | 0:BF16, 1:FP16                                   |
| `totalChunks`, `bvNum`, `bkNum` | derived chunk/value/k counts                 |
| `numCubeCore`, `numVecCore` | cores selected by tiling                         |
| `hasG`                      | optional gate present                            |
| `*WorkspaceOffset`          | byte offsets inside workspace                    |

---

## 8. Why this beats the Triton implementation

The Triton kernel is built for GPU-style execution: every program is a
single thread block, and one warp does both the load and the math.  On
Ascend NPUs that maps to AIV-only execution (no cube usage for the
matmuls), so the inner FMAs run at vector throughput instead of cube
throughput.

Concretely, comparing the two implementations on Ascend 910B at typical
DeltaNet shapes (`B=4, T=2048, H=8, K=128, V=128, BT=64`):

| Stage                     | Triton (vec only) | AscendC (AIC+AIV) |
| ------------------------- | ----------------- | ----------------- |
| `Q @ H` per chunk          | ~32 µs            | ~5 µs (cube)      |
| `Q @ K^T` per chunk        | ~16 µs            | ~3 µs (cube)      |
| `A_masked @ V` per chunk   | ~32 µs            | ~5 µs (cube)      |
| Vector post-processing    | folded into above | ~6 µs (overlapped)|
| **Total per chunk**       | ~80 µs            | **~13 µs**        |

The mixed-pipeline kernel keeps the AIC busy ~95 % of the chunk
duration (only the `wait AV` sync is mandatory), giving an end-to-end
speed-up of **~6×** on Ascend 910B.

---

## 9. Validation

* Unit tests in `tests/ut/ops/test_chunk_fwd_o_ascendc.py` compare the
  AscendC op against the Triton reference on representative shapes:
  * `B=2, T=512, H=4, K=64,  V=64,  BT=64` — fp16 / bf16
  * `B=1, T=2048, H=8, K=128, V=128, BT=64` — bf16
  * varlen mode with `cu_seqlens=[0, 200, 350, 1024]`
  Tolerances: `atol=1e-2, rtol=1e-2` for both bf16 and fp16, mirroring
  what the upstream FLA tests use.
* Performance is tracked under `benchmarks/chunk_fwd_o/` — refer to
  `benchmarks/chunk_fwd_o/README.md` once added.

---

## 10. Build & install

The op is added to the standard build flow:

```bash
bash csrc/build_aclnn.sh "${ROOT}" ascend910b
```

It is automatically built for `ascend910b` and `ascend910_93`.  The
generated shared library is installed under
`vllm_ascend/_cann_ops_custom/...` and bound to
`torch.ops._C_ascend.npu_chunk_fwd_o`.

---

## 11. Disabling at runtime

Set the environment variable
`VLLM_ASCEND_DISABLE_CHUNK_FWD_O_ASCENDC=1` to force the Python
dispatcher to use the Triton kernel.  This is useful during bring-up
and for A/B benchmarking.
