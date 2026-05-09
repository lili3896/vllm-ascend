# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for the AscendC implementation of FLA chunk_fwd_o.

When the custom AscendC op (`torch.ops._C_ascend.npu_chunk_fwd_o`) is not
available the dispatcher must transparently fall back to the Triton kernel.
This is the only behaviour we can verify in CI environments without an NPU
runtime, so most tests are dispatcher-shape tests.

A numerical comparison test is included but skipped automatically when no
NPU device is visible / the custom op is not registered.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest
import torch


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _has_ascendc_op() -> bool:
    op_set = getattr(torch.ops, "_C_ascend", None)
    return op_set is not None and hasattr(op_set, "npu_chunk_fwd_o")


class TestChunkFwdODispatcher(unittest.TestCase):
    """Exercises the python-level dispatcher logic without launching a kernel."""

    def test_dispatcher_falls_back_when_op_missing(self):
        """When the custom op is unavailable the dispatcher must call triton."""
        from vllm_ascend.ops.triton.fla import chunk_o_ascendc as mod

        with mock.patch.object(mod, "_ascendc_chunk_fwd_o_available",
                               return_value=False), \
             mock.patch.object(mod, "_triton_chunk_fwd_o") as triton_fn:
            triton_fn.return_value = torch.zeros(1)
            q = torch.zeros((1, 64, 1, 64))
            k = torch.zeros((1, 64, 1, 64))
            v = torch.zeros((1, 64, 1, 64))
            h = torch.zeros((1, 1, 64, 64))
            out = mod.chunk_fwd_o_ascendc(q, k, v, h)
            triton_fn.assert_called_once()
            self.assertIs(out, triton_fn.return_value)

    def test_unsupported_chunk_size_falls_back(self):
        from vllm_ascend.ops.triton.fla import chunk_o_ascendc as mod
        with mock.patch.object(mod, "_ascendc_chunk_fwd_o_available",
                               return_value=True), \
             mock.patch.object(mod, "_triton_chunk_fwd_o") as triton_fn:
            triton_fn.return_value = torch.zeros(1)
            q = torch.zeros((1, 32, 1, 64))
            k = torch.zeros_like(q)
            v = torch.zeros((1, 32, 1, 64))
            h = torch.zeros((1, 1, 64, 64))
            mod.chunk_fwd_o_ascendc(q, k, v, h, chunk_size=32)
            triton_fn.assert_called_once()

    def test_env_disable_forces_fallback(self):
        from vllm_ascend.ops.triton.fla import chunk_o_ascendc as mod
        with mock.patch.dict(os.environ,
                             {"VLLM_ASCEND_DISABLE_CHUNK_FWD_O_ASCENDC": "1"}):
            mod._ASCENDC_OP_AVAILABLE = None  # force re-evaluation
            self.assertFalse(mod._ascendc_chunk_fwd_o_available())
        mod._ASCENDC_OP_AVAILABLE = None  # reset for other tests


@pytest.mark.skipif(not (_has_npu() and _has_ascendc_op()),
                    reason="ChunkFwdO AscendC op or NPU not available")
class TestChunkFwdONumerical(unittest.TestCase):
    """Numerical comparison vs. the Triton reference. Run only on NPU."""

    def _run_one(self, B, T, H, K, V, dtype, varlen=False):
        from vllm_ascend.ops.triton.fla.chunk_o import \
            chunk_fwd_o as triton_chunk_fwd_o
        from vllm_ascend.ops.triton.fla.chunk_o_ascendc import \
            chunk_fwd_o_ascendc

        device = "npu"
        torch.manual_seed(0)
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        BT = 64
        NT = (T + BT - 1) // BT
        h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
        g = torch.randn(B, H, T, dtype=torch.float32, device=device) * -0.1

        cu_seqlens = None
        if varlen:
            cu_seqlens = torch.tensor([0, T, 2 * T], device=device,
                                      dtype=torch.int64)
        scale = K ** -0.5
        ref = triton_chunk_fwd_o(q.clone(), k.clone(), v.clone(),
                                 h.clone(), g.clone(), scale=scale,
                                 cu_seqlens=cu_seqlens, chunk_size=BT)
        out = chunk_fwd_o_ascendc(q, k, v, h, g, scale=scale,
                                  cu_seqlens=cu_seqlens, chunk_size=BT)
        torch.testing.assert_close(out.float(), ref.float(),
                                   atol=1e-2, rtol=1e-2)

    def test_bf16_basic(self):
        self._run_one(2, 512, 4, 64, 64, torch.bfloat16)

    def test_bf16_large(self):
        self._run_one(1, 2048, 8, 128, 128, torch.bfloat16)

    def test_fp16_basic(self):
        self._run_one(2, 256, 4, 64, 64, torch.float16)


if __name__ == "__main__":
    unittest.main()
