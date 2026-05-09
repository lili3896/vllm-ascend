/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_tiling_data.h
 * \brief ChunkFwdO 算子在 host 与 device 端共享的 tiling 数据结构定义。
 *
 *  ChunkFwdO 计算 FLA（Flash-Linear-Attention）前向 chunk 切分的输出，
 *  数学行为与 vllm_ascend 中 triton 版本
 *  vllm_ascend/ops/triton/fla/chunk_o.py 完全一致：
 *
 *      o = (Q @ H) * exp(g) * scale + (causal_mask(Q @ K) * exp(g_i - g_j)) @ V * scale
 *
 *  完整的设计文档参见：
 *  docs/source/developer_guide/feature_guide/ascendc_chunk_fwd_o
 */

#ifndef CHUNK_FWD_O_TILING_DATA_H
#define CHUNK_FWD_O_TILING_DATA_H

#include <cstdint>

namespace ChunkFwdO {

// 算子使用的 chunk 大小及 K/V 方向的内层块大小（与 triton 版默认一致）。
constexpr uint32_t CHUNK_FWD_O_BT = 64;          // 单个 chunk 包含的 token 数 BT
constexpr uint32_t CHUNK_FWD_O_BK = 128;         // K 方向的内层块大小
constexpr uint32_t CHUNK_FWD_O_BV = 128;         // V 方向的内层块大小

// dataType 字段编码（由 host 端 tiling 写入，device 端 dispatch 时使用）
constexpr uint32_t CHUNK_FWD_O_DTYPE_BF16 = 0;
constexpr uint32_t CHUNK_FWD_O_DTYPE_FP16 = 1;

#pragma pack(push, 8)
struct alignas(8) ChunkFwdOTilingData {
    int64_t shapeBatch;                 // N，输入序列条数
    int64_t seqlen;                     // T，固定 shape 模式下的序列最大长度
    int64_t kNumHead;                   // Hg，q/k 的 KV 分组头数
    int64_t vNumHead;                   // H，v/o 的完整头数
    int64_t kHeadDim;                   // K，每个头的 K 维度
    int64_t vHeadDim;                   // V，每个头的 V 维度
    float   scale;                      // attention 缩放系数
    int64_t chunkSize;                  // BT，chunk 大小
    int64_t isVariedLen;                // 0 = 固定长度，1 = 启用 cu_seqlens
    int64_t tokenBatch;                 // varlen 模式下的总 token 数；否则为 N*T
    int64_t dataType;                   // 0:BF16，1:FP16
    int64_t totalChunks;                // sum(ceil(seqlen[i]/BT))，对应 h 工作区下标基准
    int64_t bvNum;                      // ceil(V / BV)
    int64_t bkNum;                      // ceil(K / BK)
    int64_t numCubeCore;                // 实际选用的 AIC 核数
    int64_t numVecCore;                 // 实际选用的 AIV 核数
    int64_t hasG;                       // 1 表示存在可选输入 g
    // workspace 内部各缓冲区相对于工作区起始地址的字节偏移。
    int64_t vWorkspaceOffset;           // (A_masked @ V) 中间结果，fp32
    int64_t hWorkspaceOffset;           // (Q @ H)        中间结果，fp32
    int64_t attnWorkspaceOffset;        // (Q @ K) 注意力分数，    fp32
    int64_t aftermaskWorkspaceOffset;   // 经过 mask 与 g 缩放后的注意力分数，dtype 与输入一致
    int64_t maskWorkspaceOffset;        // 因果 mask 的设备端缓存（当前实现只在 UB 中保留），fp32
};
#pragma pack(pop)

}  // namespace ChunkFwdO

#endif  // CHUNK_FWD_O_TILING_DATA_H
