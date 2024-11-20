cmake_minimum_required(VERSION 3.10)

project(LLM_NPU_EXAMPLE VERSION 1.0.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenVINO REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
add_library(tokenizers_cpp STATIC IMPORTED)
set_target_properties(tokenizers_cpp PROPERTIES IMPORTED_LOCATION ${LIBRARY_DIR}/tokenizers_cpp.lib)
add_library(npu_llm STATIC IMPORTED)
set_target_properties(npu_llm PROPERTIES IMPORTED_LOCATION ${LIBRARY_DIR}/npu_llm.lib)

set(TARGET llm-npu-cli)
add_executable(${TARGET} llm-npu-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE tokenizers_cpp npu_llm openvino::runtime)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_custom_command(TARGET llm-npu-cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIBRARY_DIR}/npu_llm.dll
        ${CMAKE_BINARY_DIR}/Release/
    COMMENT "Copying npu_llm.dll to build/Release"
)