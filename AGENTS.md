# cpp-bpe-tokenizer

C++20 implementation of GPT-2 style byte-level BPE tokenizer.
Port of Stanford CS336 Assignment 1 with performance benchmarking.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Test

```bash
# Generate reference fixtures first
pip install tiktoken
python3 scripts/gen_fixtures.py

# Run tests
./build/bpe_tests

# Run benchmarks
./build/bpe_bench --benchmark_format=console

# Compare against tiktoken/HF
python3 scripts/compare_perf.py
```

## Architecture

- `include/bpe/` — Public API headers
  - `tokenizer.h` — Main Tokenizer class
  - `pretokenizer.h` — GPT-2 regex pre-tokenization
  - `trainer.h` — BPE training primitives (count_pairs, find_best_pair, apply_merge)
  - `codec.h` — Encoding/decoding helpers (bytes_to_ids, apply_merges, ids_to_bytes)
- `src/` — Implementation
  - Current implementation is consolidated in `src/codec.cpp`
  - `src/pretokenizer.cpp`, `src/tokenizer.cpp`, and `src/trainer.cpp` are shims
- `tests/` — GTest unit tests
- `bench/` — Google Benchmark performance tests
- `scripts/` — Fixture generation and comparison tools

## Historical implementation order

1. `bytes_to_ids()` and `ids_to_bytes()` — trivial, gets tests green fast
2. `pretokenize()` — GPT-2 regex splitting
3. `count_pairs()`, `find_best_pair()`, `apply_merge()` — training primitives
4. `train_bpe()` — full training loop
5. `Tokenizer::train()` — wire it all together
6. `Tokenizer::encode()` / `decode()` — encoding pipeline
7. `Tokenizer::save()` / `load()` — persistence
8. Optimize: incremental pair counts, better data structures
