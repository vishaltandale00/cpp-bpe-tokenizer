#include <benchmark/benchmark.h>
#include "bpe/tokenizer.h"
#include "bpe/trainer.h"
#include "bench_helpers.h"

// ═══════════════════════════════════════════════════════════════════
// Training benchmarks
//
// These measure the full training pipeline: pretokenize + BPE merge loop.
// Compare your results against:
//   - CS336 requirement: TinyStories in < 1.5s
//   - tiktoken training speed
//   - HuggingFace tokenizers training speed
// ═══════════════════════════════════════════════════════════════════

static void BM_TrainTiny(benchmark::State& state) {
    auto corpus = bench_helpers::tiny_corpus();
    uint32_t vocab_size = static_cast<uint32_t>(state.range(0));
    for (auto _ : state) {
        bpe::Tokenizer tok;
        tok.train(corpus, vocab_size);
        benchmark::DoNotOptimize(tok.vocab_size());
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
    state.SetBytesProcessed(state.iterations() * corpus.size());
}
BENCHMARK(BM_TrainTiny)->Arg(270)->Arg(280)->Arg(300);

static void BM_TrainMedium(benchmark::State& state) {
    auto corpus = bench_helpers::medium_corpus(100);
    uint32_t vocab_size = static_cast<uint32_t>(state.range(0));
    for (auto _ : state) {
        bpe::Tokenizer tok;
        tok.train(corpus, vocab_size);
        benchmark::DoNotOptimize(tok.vocab_size());
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
    state.SetBytesProcessed(state.iterations() * corpus.size());
    state.SetLabel(std::to_string(corpus.size() / 1024) + " KB corpus");
}
BENCHMARK(BM_TrainMedium)->Arg(500)->Arg(1000)->Arg(2000);

static void BM_TrainLarge(benchmark::State& state) {
    auto corpus = bench_helpers::large_corpus();
    uint32_t vocab_size = static_cast<uint32_t>(state.range(0));
    for (auto _ : state) {
        bpe::Tokenizer tok;
        tok.train(corpus, vocab_size);
        benchmark::DoNotOptimize(tok.vocab_size());
    }
    state.SetItemsProcessed(state.iterations() * corpus.size());
    state.SetBytesProcessed(state.iterations() * corpus.size());
    state.SetLabel(std::to_string(corpus.size() / (1024 * 1024)) + " MB corpus");
}
BENCHMARK(BM_TrainLarge)->Arg(1000)->Arg(5000)->Arg(10000);

// ═══════════════════════════════════════════════════════════════════
// Training sub-step benchmarks (for profiling bottlenecks)
// ═══════════════════════════════════════════════════════════════════

static void BM_CountPairs(benchmark::State& state) {
    // Pre-tokenized chunks from a medium corpus.
    auto corpus = bench_helpers::medium_corpus(50);
    // Simulate: each "word" is a chunk of byte IDs.
    std::vector<std::vector<uint32_t>> chunks;
    std::vector<uint32_t> current;
    for (char c : corpus) {
        if (c == ' ' && !current.empty()) {
            chunks.push_back(std::move(current));
            current.clear();
        } else {
            current.push_back(static_cast<uint8_t>(c));
        }
    }
    if (!current.empty()) chunks.push_back(std::move(current));

    for (auto _ : state) {
        auto counts = bpe::count_pairs(chunks);
        benchmark::DoNotOptimize(counts.size());
    }
}
BENCHMARK(BM_CountPairs);

static void BM_ApplyMerge(benchmark::State& state) {
    auto corpus = bench_helpers::medium_corpus(50);
    std::vector<std::vector<uint32_t>> chunks;
    std::vector<uint32_t> current;
    for (char c : corpus) {
        if (c == ' ' && !current.empty()) {
            chunks.push_back(std::move(current));
            current.clear();
        } else {
            current.push_back(static_cast<uint8_t>(c));
        }
    }
    if (!current.empty()) chunks.push_back(std::move(current));

    // Find the most common pair to merge.
    auto counts = bpe::count_pairs(chunks);
    auto best = bpe::find_best_pair(counts);

    // Benchmark applying this merge repeatedly (reset chunks each time).
    auto chunks_copy = chunks;
    for (auto _ : state) {
        state.PauseTiming();
        chunks = chunks_copy;
        state.ResumeTiming();
        bpe::apply_merge(chunks, best.pair.left, best.pair.right, 256);
    }
}
BENCHMARK(BM_ApplyMerge);
