#include <benchmark/benchmark.h>
#include "bpe/tokenizer.h"
#include "bench_helpers.h"

// ═══════════════════════════════════════════════════════════════════
// Decoding benchmarks
// ═══════════════════════════════════════════════════════════════════

static void BM_DecodeShort(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    auto ids = tok.encode("Hello, world! This is a short sentence.");
    for (auto _ : state) {
        auto text = tok.decode(ids);
        benchmark::DoNotOptimize(text.data());
    }
    state.SetItemsProcessed(state.iterations() * ids.size());
}
BENCHMARK(BM_DecodeShort);

static void BM_DecodeLarge(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_large();
    auto ids = tok.encode(bench_helpers::medium_corpus(100));
    for (auto _ : state) {
        auto text = tok.decode(ids);
        benchmark::DoNotOptimize(text.data());
    }
    state.SetItemsProcessed(state.iterations() * ids.size());
    state.SetLabel(std::to_string(ids.size()) + " tokens");
}
BENCHMARK(BM_DecodeLarge);

static void BM_DecodeVaryingLength(benchmark::State& state) {
    auto& tok = bench_helpers::cached_tokenizer_small();
    size_t len = static_cast<size_t>(state.range(0));
    auto full_text = bench_helpers::medium_corpus(10);
    auto ids = tok.encode(full_text.substr(0, len));
    for (auto _ : state) {
        auto text = tok.decode(ids);
        benchmark::DoNotOptimize(text.data());
    }
    state.SetItemsProcessed(state.iterations() * ids.size());
}
BENCHMARK(BM_DecodeVaryingLength)
    ->Arg(10)->Arg(100)->Arg(1000)->Arg(10000)->Arg(50000);
