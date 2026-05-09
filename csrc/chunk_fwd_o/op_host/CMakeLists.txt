# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.

add_ops_compile_options(
    OP_NAME ChunkFwdO
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
)

target_sources(op_host_aclnnExc PRIVATE chunk_fwd_o_def.cpp)

target_sources(opapi PRIVATE
    chunk_fwd_o.cpp
    aclnn_chunk_fwd_o.cpp
)

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

target_sources(optiling PRIVATE chunk_fwd_o_tiling.cpp)
target_sources(opsproto PRIVATE chunk_fwd_o_infershape.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB _CFO_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_chunk_fwd_o.h")

install(FILES ${_CFO_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
