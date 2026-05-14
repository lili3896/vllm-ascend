# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# 本文件是 vllm-ascend 项目的一部分。
# 在遵守相应许可协议的前提下，您可以重新分发或修改本程序。
# 详细条款请参阅项目根目录下的 LICENSE 文件。

# ChunkFwdO 算子的编译选项：开启 cce 自动同步、屏蔽过时声明告警。
add_ops_compile_options(
    OP_NAME ChunkFwdO
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
)

# 把 OpDef 注册源文件加入 op_host_aclnnExc 目标。
target_sources(op_host_aclnnExc PRIVATE chunk_fwd_o_def.cpp)

# 把 L0 op 与 aclnn 入口源文件加入 opapi 目标。
target_sources(opapi PRIVATE
    chunk_fwd_o.cpp
    aclnn_chunk_fwd_o.cpp
)

# 非开源构建模式下，同时把源文件加入 train/infer 目标。
if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        chunk_fwd_o.cpp
        aclnn_chunk_fwd_o.cpp
    )
    target_sources(aclnn_ops_infer PRIVATE
        chunk_fwd_o.cpp
        aclnn_chunk_fwd_o.cpp
    )
endif ()

# tiling/proto 源文件分别注册到 optiling/opsproto 目标。
target_sources(optiling PRIVATE chunk_fwd_o_tiling.cpp)
target_sources(opsproto PRIVATE chunk_fwd_o_infershape.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 安装 aclnn 头文件。
file(GLOB _CFO_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_chunk_fwd_o.h")

install(FILES ${_CFO_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
