cmake_minimum_required(VERSION 3.20)
project(bpe_consumer_smoke LANGUAGES CXX)

if(NOT DEFINED BPE_SOURCE_DIR)
  message(FATAL_ERROR "Set BPE_SOURCE_DIR to the cpp-bpe-tokenizer repository root.")
endif()

get_filename_component(BPE_SOURCE_DIR "${BPE_SOURCE_DIR}" ABSOLUTE)
if(NOT EXISTS "${BPE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "BPE_SOURCE_DIR does not point to a CMake project: ${BPE_SOURCE_DIR}")
endif()

add_subdirectory("${BPE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_bpe")

if(TARGET bpe_bench)
  message(FATAL_ERROR "bpe_bench must not be created by default for add_subdirectory consumers.")
endif()

add_executable(bpe_consumer_smoke main.cpp)
target_link_libraries(bpe_consumer_smoke PRIVATE bpe_tokenizer)
