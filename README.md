# cpp-bpe-tokenizer

C++20 implementation of a GPT-2 style byte-level BPE tokenizer.

The project includes:

- byte-level encode/decode helpers
- GPT-2 style pre-tokenization
- BPE training primitives and training loop
- special-token handling
- save/load support for the native format and generated GPT-2 fixtures
- GoogleTest correctness tests and Google Benchmark performance tests

This is an experimental library, but the current tree builds and passes the
full local test suite, including tiktoken-based GPT-2 compatibility fixtures
when those fixtures have been generated.

## Requirements

- CMake 3.20+
- C++20 compiler
- Python 3, only for fixture generation and Python comparison scripts

CMake fetches C++ test and benchmark dependencies when those targets are
enabled.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Test

Generate the reference fixtures first if you want the GPT-2/tiktoken
compatibility tests to run instead of being skipped:

```bash
python3 -m pip install tiktoken
python3 scripts/gen_fixtures.py
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

You can also run the test binary directly:

```bash
./build/bpe_tests
```

## Benchmarks

Benchmarks are built by default only when this project is configured as the
top-level CMake project. When included from another project with
`add_subdirectory`, `BPE_BUILD_BENCHMARKS` defaults to `OFF`. Run:

```bash
./build/bpe_bench --benchmark_format=console
```

For a faster smoke benchmark:

```bash
./build/bpe_bench --benchmark_min_time=0.03s --benchmark_format=console
```

## Usage

```cpp
#include <bpe/tokenizer.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    bpe::Tokenizer tokenizer;
    tokenizer.train("the cat sat on the mat. the cat ate the rat.", 300,
                    {"<|endoftext|>"});

    std::vector<uint32_t> ids = tokenizer.encode("the cat<|endoftext|>");
    std::string text = tokenizer.decode(ids);

    std::cout << text << "\n";
}
```

## Saving And Loading

```cpp
tokenizer.save("tokenizer-model");

auto loaded = bpe::Tokenizer::load("tokenizer-model");
auto ids = loaded.encode("hello world");
```

`Tokenizer::load()` also supports the hex-encoded GPT-2 fixture files produced
by `scripts/gen_fixtures.py`.

## Repository Layout

- `include/bpe/` - public API headers
- `src/` - implementation
- `tests/` - GoogleTest unit and compatibility tests
- `bench/` - Google Benchmark performance tests
- `scripts/` - fixture generation and comparison helpers

## Public Release Notes

Generated fixtures, local API keys, editor settings, research run output, and
vendored reference repositories are intentionally ignored. Generate fixtures
locally or in CI instead of committing them.

## License

MIT. See `LICENSE`.
