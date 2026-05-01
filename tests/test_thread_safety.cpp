#include <gtest/gtest.h>
#include "bpe/tokenizer.h"
#include "test_helpers.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace {

void run_concurrent_encode_check(const bpe::Tokenizer& tok) {
    const std::vector<std::string> inputs = {
        "the cat sat on the mat",
        "hello<|endoftext|>world",
        "unseen text with 123 numbers and !@# symbols",
        test_helpers::medium_corpus(2),
    };

    std::vector<std::vector<uint32_t>> expected;
    expected.reserve(inputs.size());
    for (const auto& input : inputs) {
        expected.push_back(tok.encode(input));
    }

    constexpr int kThreads = 12;
    constexpr int kIterations = 200;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < kIterations && !failed.load(std::memory_order_relaxed); ++i) {
                const size_t idx = static_cast<size_t>(t + i) % inputs.size();
                if (tok.encode(inputs[idx]) != expected[idx]) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kThreads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_FALSE(failed.load());
}

} // namespace

TEST(ThreadSafety, ConcurrentEncodeSharedTrainedTokenizer) {
    bpe::Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 300, {"<|endoftext|>"});
    run_concurrent_encode_check(tok);
}

TEST(ThreadSafety, ConcurrentEncodeSharedLoadedTokenizer) {
    auto dir = test_helpers::temp_dir();
    bpe::Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 300, {"<|endoftext|>"});
    tok.save(dir);

    auto loaded = bpe::Tokenizer::load(dir);
    run_concurrent_encode_check(loaded);
    test_helpers::cleanup_temp_dir();
}
