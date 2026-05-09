# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC implementation of the Flash-Linear-Attention chunk forward output.

This module exposes :func:`chunk_fwd_o_ascendc`, which is functionally
equivalent to the Triton kernel in :mod:`vllm_ascend.ops.triton.fla.chunk_o`
but executes a custom AscendC kernel (``npu_chunk_fwd_o``) registered through
``torch.ops._C_ascend``.  The custom kernel is significantly faster on
Ascend NPUs because it dispatches the three matmuls onto the AIC (cube) cores
while the AIV (vector) cores fuse exp(g), causal masking, scaling and the cast
in a 1 AIC + 2 AIV mixed pipeline.

The Python entry point falls back automatically to the Triton implementation
when:

* ``vllm_ascend`` is built without custom ops, or
* ``npu_chunk_fwd_o`` is not present in ``torch.ops._C_ascend``, or
* the input shape is unsupported by the AscendC kernel (currently we require
  ``K`` and ``V`` to be a multiple of 16 and ``chunk_size`` to be 64).
"""
from __future__ import annotations

import os

import torch

from .chunk_o import chunk_fwd_o as _triton_chunk_fwd_o
from .utils import prepare_chunk_offsets


_ASCENDC_OP_AVAILABLE: bool | None = None


def _ascendc_chunk_fwd_o_available() -> bool:
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
    if cu_seqlens is not None:
        return cu_seqlens.to(torch.int64)
    # Fixed-shape mode: synthesize cu_seqlens = [0, T, 2T, ..., B*T].
    B, T = q.shape[0], q.shape[1]
    return torch.arange(0, (B + 1) * T, T,
                        device=q.device, dtype=torch.int64)


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
    """Compute FLA chunk-wise output ``o`` using the AscendC custom op when
    available, otherwise fall back to the Triton implementation.

    Signature matches :func:`vllm_ascend.ops.triton.fla.chunk_o.chunk_fwd_o`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if not _ascendc_chunk_fwd_o_available() or not _supports_shape(q, v, chunk_size):
        return _triton_chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale,
                                   cu_seqlens=cu_seqlens,
                                   chunk_size=chunk_size,
                                   chunk_offsets=chunk_offsets)

    cu_seqlens_t = _maybe_build_cu_seqlens(q, cu_seqlens)
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens_t, chunk_size)
    chunk_offsets_t = chunk_offsets.to(torch.int64)

    # The AscendC kernel expects g in (T, H) layout already (matching the
    # triton kernel's transpose).  Mirror the conversion done there.
    g_in: torch.Tensor | None = None
    if g is not None:
        g_in = g.transpose(-2, -1).contiguous()

    o = torch.ops._C_ascend.npu_chunk_fwd_o(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        h.contiguous(),
        g_in,
        cu_seqlens_t.contiguous(),
        chunk_offsets_t.contiguous(),
        float(scale),
        int(chunk_size),
    )
    return o
