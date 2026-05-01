#include <benchmark/benchmark.h>
#include "bpe/tokenizer.h"
#include "bench_helpers.h"

// ═══════════════════════════════════════════════════════════════════
// Encoding benchmarks
//
// Key metrics:
//   - Throughput (MB/s) — compare against:
//       tiktoken:           ~3-10 MB/s
//       GitHub bpe crate:   ~30-80 MB/s
//       HuggingFace:        ~5-15 MB/s
//   - Latency per encode call
// ═══════════════════════════════════════════════════════════════════

static void BM_EncodeShort(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    std::string text = "Hello, world! This is a short sentence.";
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeShort);

static void BM_EncodeMedium(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    std::string text = bench_helpers::medium_corpus(1); // ~600 bytes
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeMedium);

static void BM_EncodeLarge(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_large();
    std::string text = bench_helpers::medium_corpus(100); // ~60KB
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel(std::to_string(text.size() / 1024) + " KB");
}
BENCHMARK(BM_EncodeLarge);

static void BM_EncodeVaryingLength(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    size_t len = static_cast<size_t>(state.range(0));
    std::string text = bench_helpers::medium_corpus(10).substr(0, len);
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeVaryingLength)
    ->Arg(10)->Arg(100)->Arg(1000)->Arg(10000)->Arg(50000);

// ═══════════════════════════════════════════════════════════════════
// Encoding with special tokens
// ═══════════════════════════════════════════════════════════════════

static void BM_EncodeWithSpecials(benchmark::State& state) {
    static bpe::Tokenizer tok = [] {
        bpe::Tokenizer t;
        t.train(bench_helpers::medium_corpus(10), 500, {"<|endoftext|>"});
        return t;
    }();

    // Text with special tokens interspersed.
    std::string text;
    auto base = bench_helpers::medium_corpus(1);
    for (int i = 0; i < 10; ++i) {
        text += base + "<|endoftext|>";
    }

    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeWithSpecials);

// ═══════════════════════════════════════════════════════════════════
// Unicode-heavy encoding
// ═══════════════════════════════════════════════════════════════════

static void BM_EncodeUnicode(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    // Mix of multi-byte UTF-8 characters.
    std::string text;
    for (int i = 0; i < 100; ++i) {
        text += "café résumé naïve 你好世界 🌍🚀 ";
    }
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeUnicode);

static void BM_EncodeHoldoutCode(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    std::string text = bench_helpers::holdout_code_corpus();
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeHoldoutCode);

static void BM_EncodeHoldoutPunct(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    std::string text = bench_helpers::holdout_punctuation_corpus();
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeHoldoutPunct);

static void BM_EncodeHoldoutMultilingual(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    std::string text = bench_helpers::holdout_multilingual_corpus();
    for (auto _ : state) {
        auto ids = tok.encode(text);
        benchmark::DoNotOptimize(ids.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_EncodeHoldoutMultilingual);
