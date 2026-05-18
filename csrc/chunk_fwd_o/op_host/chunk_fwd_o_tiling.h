/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_tiling.h
 * \brief ChunkFwdO 算子的 host tiling 头文件。
 */

#ifndef __OP_HOST_CHUNK_FWD_O_TILING_H__
#define __OP_HOST_CHUNK_FWD_O_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../tiling_base/tiling_base.h"
#include "../op_kernel/chunk_fwd_o_tiling_data.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkFwdOTilingData)
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(float,   scale);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(int64_t, totalChunks);
TILING_DATA_FIELD_DEF(int64_t, numChunks);
TILING_DATA_FIELD_DEF(int64_t, vLoops);
TILING_DATA_FIELD_DEF(int64_t, taskNum);
TILING_DATA_FIELD_DEF(int64_t, numCubeCore);
TILING_DATA_FIELD_DEF(int64_t, numVecCore);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, hasG);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, attnWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, aftermaskWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, maskWorkspaceOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkFwdO, ChunkFwdOTilingData)

struct ChunkFwdOCompileInfo {
    uint64_t aicNum {0};
    uint64_t aivNum {0};
    uint64_t ubSize {0};
    uint64_t l1Size {0};
};

struct ChunkFwdOInfo {
    const char* opName = "ChunkFwdO";
};

class ChunkFwdOTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ChunkFwdOTiling(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    }
    ~ChunkFwdOTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t        GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void InitCompileInfo();
    void PrintTilingData();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus AnalyzeAttrs();
    ge::graphStatus PlanWorkspaces();
    ge::graphStatus PlanCorePartition();

    ChunkFwdOCompileInfo                     compileInfo_;
    ::optiling::ChunkFwdOTilingData          tilingData_;
    ChunkFwdOInfo                            inputParams_;

    int64_t totalWorkspaceBytes_ {0};
    int64_t systemWorkspaceBytes_ {16 * 1024 * 1024};
};

}  // namespace optiling

#endif  // __OP_HOST_CHUNK_FWD_O_TILING_H__
