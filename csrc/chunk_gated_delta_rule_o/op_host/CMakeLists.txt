# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Build registration for the ChunkGatedDeltaRuleO operator. Mirrors the
# pattern used by csrc/recurrent_gated_delta_rule.

add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleO
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnExc PRIVATE
    chunk_gated_delta_rule_o_def.cpp
)

target_sources(opapi PRIVATE
    aclnn_chunk_gated_delta_rule_o.cpp
    chunk_gated_delta_rule_o.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        chunk_gated_delta_rule_o.cpp
        aclnn_chunk_gated_delta_rule_o.cpp
    )
    target_sources(aclnn_ops_infer PRIVATE
        chunk_gated_delta_rule_o.cpp
        aclnn_chunk_gated_delta_rule_o.cpp
    )
endif ()

target_sources(optiling PRIVATE
    chunk_gated_delta_rule_o_tiling.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE)

file(GLOB _CGDRO_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_chunk_gated_delta_rule_o.h")

install(FILES ${_CGDRO_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
