#include <benchmark/benchmark.h>
#include "bpe/pretokenizer.h"
#include "bench_helpers.h"

// ═══════════════════════════════════════════════════════════════════
// Pre-tokenizer benchmarks
//
// Pre-tokenization is often a bottleneck — regex matching can dominate
// the encoding pipeline. These benchmarks help you compare:
//   - std::regex vs hand-rolled state machine vs RE2
// ═══════════════════════════════════════════════════════════════════

static void BM_PretokenizeShort(benchmark::State& state) {
    std::string text = "Hello, world! I'm fine. You're great.";
    for (auto _ : state) {
        auto result = bpe::pretokenize(text);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_PretokenizeShort);

static void BM_PretokenizeMedium(benchmark::State& state) {
    auto text = bench_helpers::medium_corpus(1);
    for (auto _ : state) {
        auto result = bpe::pretokenize(text);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_PretokenizeMedium);

static void BM_PretokenizeLarge(benchmark::State& state) {
    auto text = bench_helpers::medium_corpus(100);
    for (auto _ : state) {
        auto result = bpe::pretokenize(text);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel(std::to_string(text.size() / 1024) + " KB");
}
BENCHMARK(BM_PretokenizeLarge);

static void BM_PretokenizeWithSpecials(benchmark::State& state) {
    auto text = bench_helpers::medium_corpus(10);
    // Insert special tokens throughout.
    std::string modified;
    size_t chunk_size = text.size() / 20;
    for (size_t i = 0; i < text.size(); i += chunk_size) {
        modified += text.substr(i, chunk_size) + "<|endoftext|>";
    }
    std::vector<std::string> specials = {"<|endoftext|>"};

    for (auto _ : state) {
        auto result = bpe::pretokenize_with_specials(modified, specials);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * modified.size());
}
BENCHMARK(BM_PretokenizeWithSpecials);

// ═══════════════════════════════════════════════════════════════════
// Adversarial inputs for pre-tokenizer
// ═══════════════════════════════════════════════════════════════════

static void BM_PretokenizeAllPunctuation(benchmark::State& state) {
    // Worst case: every character is a separate chunk.
    std::string text(10000, '!');
    for (auto _ : state) {
        auto result = bpe::pretokenize(text);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_PretokenizeAllPunctuation);

static void BM_PretokenizeAllWhitespace(benchmark::State& state) {
    std::string text(10000, ' ');
    for (auto _ : state) {
        auto result = bpe::pretokenize(text);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_PretokenizeAllWhitespace);

static void BM_PretokenizeAllDigits(benchmark::State& state) {
    std::string text;
    for (int i = 0; i < 10000; ++i) {
        text += std::to_string(i % 10);
    }
    for (auto _ : state) {
        auto result = bpe::pretokenize(text);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_PretokenizeAllDigits);
