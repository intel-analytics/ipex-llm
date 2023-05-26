cmake_minimum_required(VERSION 3.4...3.22)

project(bigdl_llm)

option(FORCE_CMAKE "Force CMake build of Related CPP project" OFF)

set(FORCE_CMAKE $ENV{FORCE_CMAKE})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/libllama.so ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/quantize
    COMMAND make libllama.so quantize
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp/libgptneox.so ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp/quantize-gptneox
    COMMAND make libgptneox.so quantize-gptneox
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp/libbloom.so ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp/quantize
    COMMAND make libbloom.so quantize
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp
)
add_custom_target(
    run ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/libllama.so
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/quantize
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp/libgptneox.so
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp/quantize-gptneox
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp/libbloom.so
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp/quantize
)
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/libllama.so
    DESTINATION src/bigdl/llm/lib
)
install(
    PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/quantize
    RENAME quantize-llama
    DESTINATION src/bigdl/llm/bin
)
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp/libgptneox.so
    DESTINATION src/bigdl/llm/lib
)
install(
    PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/redpajama.cpp/quantize-gptneox
    DESTINATION src/bigdl/llm/bin
)
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp/libbloom.so
    DESTINATION src/bigdl/llm/lib
)
install(
    PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/bloomz.cpp/quantize
    RENAME quantize-bloomz
    DESTINATION src/bigdl/llm/bin
)
