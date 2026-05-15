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
 * \brief ChunkFwdO 算子在 host 与 device 端共享的 tiling 数据结构。
 *
 *  本实现采用 1 AIC + 2 AIV 的 CV 融合模式，配合 PING_PONG_STAGES = 2 的
 *  双缓冲软件流水：Cube 调度器与 Vec 调度器各持一份 offsets[2]，分别在
 *  `currStage-1` 与 `currStage-2` 上读取，实现 Cube1/Cube23 与 Vec1/Vec2
 *  的 4 段交错。
 */

#ifndef CHUNK_FWD_O_TILING_DATA_H
#define CHUNK_FWD_O_TILING_DATA_H

#include <cstdint>

namespace ChunkFwdO {

// chunk 算法常量（与 triton 参考实现一致）
constexpr uint32_t CHUNK_FWD_O_MAX_BT = 64;      // 单 chunk token 数的当前实现上限
constexpr uint32_t CHUNK_FWD_O_BV = 128;         // V 方向块大小上限

// 数据类型编码
constexpr uint32_t CHUNK_FWD_O_DTYPE_BF16 = 0;
constexpr uint32_t CHUNK_FWD_O_DTYPE_FP16 = 1;

// 双缓冲阶段数（PING / PONG）
constexpr uint32_t PING_PONG_STAGES = 2;

#pragma pack(push, 8)
struct alignas(8) ChunkFwdOTilingData {
    int64_t shapeBatch;                 // B：原始 batch 数
    int64_t seqlen;                     // T：q/k/v/g 中的 T 维（varlen 即 token 总数）
    int64_t kNumHead;                   // Hg：q/k 头数
    int64_t vNumHead;                   // H ：v/o/h/g 头数
    int64_t kHeadDim;                   // D ：q/k head dim（= K）
    int64_t vHeadDim;                   // D ：v/o head dim（= V）
    float   scale;                      // 注意力缩放，默认 1/sqrt(D)
    int64_t chunkSize;                  // BT，运行时 chunk 大小，当前要求 16 对齐且不超过 MAX_BT
    int64_t isVariedLen;                // 0=固定 shape，1=cu_seqlens 变长
    int64_t totalChunks;                // NT：chunk_indices 的行数
    int64_t numChunks;                  // 单 batch 的最大 chunk 数（h 的第 3 维）
    int64_t vLoops;                     // ceil(V / BV)
    int64_t taskNum;                    // vLoops × shapeBatch × numChunks × vNumHead
    int64_t numCubeCore;                // 实际使用的 AIC 数（= blockDim）
    int64_t numVecCore;                 // 实际使用的 AIV 数（= 2 × numCubeCore）
    int64_t dataType;                   // 0:BF16, 1:FP16
    int64_t hasG;                       // 1=携带 gate g
    // 各 workspace 段相对工作区起始地址的字节偏移，每段为
    // numCubeCore × PING_PONG_STAGES × slot_size。
    int64_t hWorkspaceOffset;           // (Q @ H) 中间结果 fp32
    int64_t attnWorkspaceOffset;        // (Q @ K^T) 注意力分数 fp32
    int64_t vWorkspaceOffset;           // (A_masked @ V) 中间结果 fp32
    int64_t aftermaskWorkspaceOffset;   // 经 mask + g 衰减后的 A，dtype 与输入相同
};
#pragma pack(pop)

}  // namespace ChunkFwdO

#endif  // CHUNK_FWD_O_TILING_DATA_H
