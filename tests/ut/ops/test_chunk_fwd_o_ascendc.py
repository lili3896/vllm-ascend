# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# 本文件是 vllm-ascend 项目的一部分。
#
# 在 Apache License 2.0 协议下发布。
"""FLA ChunkFwdO 算子 AscendC 实现的单元测试。

当 AscendC 自定义算子（``torch.ops._C_ascend.npu_chunk_fwd_o``）不可用时，
分发器必须能够无感地回落到 triton kernel。这是 CI 环境（无 NPU 运行时）
中唯一可以被验证的行为，因此大部分用例只测试 Python 分发器逻辑。

数值对比用例需要 NPU 设备且自定义算子已注册，否则会自动跳过。
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
    """验证 Python 分发器的回落与开关行为，无需启动 device kernel。"""

    def test_dispatcher_falls_back_when_op_missing(self):
        """当 AscendC 算子不可用时，分发器必须调用 triton 实现。"""
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
        """非 64 的 chunk_size 必须走 triton 路径。"""
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
        """环境变量 VLLM_ASCEND_DISABLE_CHUNK_FWD_O_ASCENDC=1 应强制使用 triton。"""
        from vllm_ascend.ops.triton.fla import chunk_o_ascendc as mod
        with mock.patch.dict(os.environ,
                             {"VLLM_ASCEND_DISABLE_CHUNK_FWD_O_ASCENDC": "1"}):
            mod._ASCENDC_OP_AVAILABLE = None  # 强制重新评估
            self.assertFalse(mod._ascendc_chunk_fwd_o_available())
        mod._ASCENDC_OP_AVAILABLE = None  # 复位以免影响其它用例


@pytest.mark.skipif(not (_has_npu() and _has_ascendc_op()),
                    reason="ChunkFwdO AscendC 算子或 NPU 设备不可用")
class TestChunkFwdONumerical(unittest.TestCase):
    """与 triton 参考实现的数值对比，仅在 NPU 上执行。"""

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
