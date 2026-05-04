cmake_minimum_required(VERSION 3.20)
project(cpp-bpe-tokenizer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(BPE_TOP_LEVEL ON)
else()
  set(BPE_TOP_LEVEL OFF)
endif()

# ── Options ──────────────────────────────────────────────────────────
option(BPE_BUILD_TESTS      "Build unit tests"              ${BPE_TOP_LEVEL})
option(BPE_BUILD_BENCHMARKS "Build performance benchmarks" ${BPE_TOP_LEVEL})
option(BPE_ENABLE_ASAN      "Enable AddressSanitizer"       OFF)
option(BPE_ENABLE_TSAN      "Enable ThreadSanitizer"        OFF)
option(BPE_NATIVE_ARCH      "Build benchmarks with -march=native when supported" OFF)

if(BPE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

if(BPE_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# ── Core library ─────────────────────────────────────────────────────
add_library(bpe_tokenizer
  src/tokenizer.cpp
  src/pretokenizer.cpp
  src/trainer.cpp
  src/codec.cpp
  src/utils.cpp
)
target_compile_features(bpe_tokenizer PUBLIC cxx_std_20)
target_include_directories(bpe_tokenizer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(bpe_tokenizer PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
)

# ── Dependencies via FetchContent ────────────────────────────────────
include(FetchContent)

# nlohmann/json for vocab serialization
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# utf8proc for HF-compatible NFC normalization and Unicode categories
set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  utf8proc
  GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
  GIT_TAG        v2.11.3
)
FetchContent_MakeAvailable(utf8proc)

target_link_libraries(bpe_tokenizer PRIVATE
  nlohmann_json::nlohmann_json
  utf8proc
)

# ── Tests ────────────────────────────────────────────────────────────
if(BPE_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Unit tests
  add_executable(bpe_tests
    tests/test_train.cpp
    tests/test_encode_decode.cpp
    tests/test_pretokenizer.cpp
    tests/test_special_tokens.cpp
    tests/test_roundtrip.cpp
    tests/test_edge_cases.cpp
    tests/test_correctness.cpp
    tests/test_load_validation.cpp
    tests/test_public_api.cpp
    tests/test_thread_safety.cpp
    tests/test_main.cpp
  )
  target_link_libraries(bpe_tests PRIVATE
    bpe_tokenizer
    nlohmann_json::nlohmann_json
    GTest::gtest
  )
  target_include_directories(bpe_tests PRIVATE tests)
  target_compile_definitions(bpe_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
    FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
  )
  include(GoogleTest)
  gtest_discover_tests(bpe_tests)
endif()

# ── Benchmarks ───────────────────────────────────────────────────────
if(BPE_BUILD_BENCHMARKS)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(bpe_bench
    bench/bench_train.cpp
    bench/bench_encode.cpp
    bench/bench_decode.cpp
    bench/bench_pretokenizer.cpp
    bench/bench_main.cpp
  )
  target_link_libraries(bpe_bench PRIVATE bpe_tokenizer benchmark::benchmark)
  target_compile_definitions(bpe_bench PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
    FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
  )
  target_compile_options(bpe_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
    $<$<AND:$<BOOL:${BPE_NATIVE_ARCH}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-march=native>
  )
endif()
