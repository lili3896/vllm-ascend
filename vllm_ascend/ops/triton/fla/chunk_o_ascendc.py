# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-Linear-Attention chunk 前向输出的 AscendC 实现入口。

本模块对外暴露 :func:`chunk_fwd_o_ascendc`，其语义与
:mod:`vllm_ascend.ops.triton.fla.chunk_o` 中的 triton kernel 完全等价，
但底层会优先调用通过 ``torch.ops._C_ascend`` 注册的 AscendC 自定义算子
``npu_chunk_fwd_o``。该自定义算子在 Ascend NPU 上比 triton 版本显著更快，
原因是它把三个 GEMM 下放到 AIC（Cube）核，并在 1 AIC + 2 AIV 的 CV 融合
ping-pong 流水中让 AIV（Vector）核同时完成 exp(g)、causal mask、缩放和
回写 cast 等向量计算。

签名注意点：本入口与 triton 版本签名相同，多出来的 ``chunk_offsets``
参数仅在 triton 路径使用；走 AscendC 路径时，本入口会在内部把 triton
风格的 ``cu_seqlens`` 转成 AscendC kernel 要求的 ``cu_seqlens`` 与
``chunk_indices``（shape ``[NT, 2]``）。

在以下情况下，本入口会自动回落到 triton 实现：

* vllm-ascend 构建未启用自定义算子；
* ``npu_chunk_fwd_o`` 未在 ``torch.ops._C_ascend`` 中注册；
* 当前输入 shape 不在 AscendC kernel 的支持范围内（当前要求 ``K``、``V``
  对齐到 16，并且 ``chunk_size == 64``）。
"""
from __future__ import annotations

import os

import torch

from .chunk_o import chunk_fwd_o as _triton_chunk_fwd_o
from .utils import prepare_chunk_indices

_ASCENDC_OP_AVAILABLE: bool | None = None


def _ascendc_chunk_fwd_o_available() -> bool:
    """判断 AscendC 自定义算子在当前进程中是否可用。结果会被缓存。"""
    global _ASCENDC_OP_AVAILABLE
    if _ASCENDC_OP_AVAILABLE is not None:
        return _ASCENDC_OP_AVAILABLE
    if os.environ.get("VLLM_ASCEND_DISABLE_CHUNK_FWD_O_ASCENDC", "0") == "1":
        _ASCENDC_OP_AVAILABLE = False
        return False
    try:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            _ASCENDC_OP_AVAILABLE = False
            return False
    except Exception:
        _ASCENDC_OP_AVAILABLE = False
        return False
    op_set = getattr(torch.ops, "_C_ascend", None)
    _ASCENDC_OP_AVAILABLE = op_set is not None and hasattr(op_set, "npu_chunk_fwd_o")
    return _ASCENDC_OP_AVAILABLE


def _supports_shape(q: torch.Tensor, v: torch.Tensor, chunk_size: int) -> bool:
    """判断当前 (q, v, chunk_size) 是否落在 AscendC kernel 支持范围内。"""
    if chunk_size != 64:
        return False
    K = q.shape[-1]
    V = v.shape[-1]
    if K % 16 != 0 or V % 16 != 0:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    return True


def _maybe_build_cu_seqlens(q: torch.Tensor,
                            cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    """非 varlen 模式下根据 q 的形状合成 cu_seqlens=[0, T, 2T, ..., B*T]。"""
    if cu_seqlens is not None:
        return cu_seqlens.to(torch.int64)
    B, T = q.shape[0], q.shape[1]
    return torch.arange(0, (B + 1) * T, T,
                        device=q.device, dtype=torch.int64)


def _build_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """构造 [NT, 2] 的 chunk_indices 表。

    第一列：chunk 所属序列 id；第二列：chunk 在序列内的 id。
    与 :func:`prepare_chunk_indices` 等价，但保证 dtype/device 与
    AscendC kernel 期望一致。
    """
    try:
        ci = prepare_chunk_indices(cu_seqlens, chunk_size)
        return ci.to(dtype=torch.int64, device=cu_seqlens.device).contiguous()
    except Exception:
        # 兜底实现
        cu = cu_seqlens.tolist()
        rows = []
        for i in range(len(cu) - 1):
            n = (cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
            for j in range(n):
                rows.append([i, j])
        if not rows:
            rows = [[0, 0]]
        return torch.tensor(rows, dtype=torch.int64, device=cu_seqlens.device)


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
    """计算 FLA chunk 前向输出 ``o``。

    优先调用 AscendC 自定义算子；不可用或 shape 不支持时自动回落到 triton
    实现，因此函数签名与 :func:`vllm_ascend.ops.triton.fla.chunk_o.chunk_fwd_o`
    完全保持一致。
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if not _ascendc_chunk_fwd_o_available() or not _supports_shape(q, v, chunk_size):
        return _triton_chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale,
                                   cu_seqlens=cu_seqlens,
                                   chunk_size=chunk_size,
                                   chunk_offsets=chunk_offsets)

    cu_seqlens_t = _maybe_build_cu_seqlens(q, cu_seqlens).contiguous()
    chunk_indices_t = _build_chunk_indices(cu_seqlens_t, chunk_size)

    # AscendC kernel 期望 g 是 (B, H, T) 布局，与 triton 实现转置后的输入一致。
    g_in: torch.Tensor | None = None
    if g is not None:
        g_in = g.transpose(-2, -1).contiguous()

    o = torch.ops._C_ascend.npu_chunk_fwd_o(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        h.contiguous(),
        g_in,
        cu_seqlens_t,
        chunk_indices_t,
        float(scale),
        int(chunk_size),
    )
    return o
