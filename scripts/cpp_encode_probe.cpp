#include "bpe/tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t hash_ids(const std::vector<uint32_t>& ids) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t id : ids) {
        for (int shift = 0; shift < 32; shift += 8) {
            h ^= static_cast<uint8_t>((id >> shift) & 0xffU);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return 0.0;
    }
    const double pos = q * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<size_t>(pos);
    const auto hi = std::min(lo + 1, values.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

uint64_t peak_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#else
    return 0;
#endif
}

void keep_ids_live(const std::vector<uint32_t>& ids) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(ids.data()), "g"(ids.size()) : "memory");
#else
    static volatile size_t sink = 0;
    sink += ids.size();
#endif
}

size_t encode_batch_single_thread(
    const bpe::Tokenizer& tokenizer,
    const std::string& text,
    size_t batch) {
    size_t local_tokens = 0;
    for (size_t i = 0; i < batch; ++i) {
        auto ids = tokenizer.encode(text);
        local_tokens += ids.size();
        keep_ids_live(ids);
    }
    return local_tokens;
}

size_t encode_batch_multi_thread(
    const bpe::Tokenizer& tokenizer,
    const std::string& text,
    size_t batch,
    size_t thread_count) {
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<size_t> token_counts(thread_count, 0);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            token_counts[t] = encode_batch_single_thread(tokenizer, text, batch);
        });
    }

    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    size_t total_tokens = 0;
    for (auto& thread : threads) {
        thread.join();
    }
    for (size_t count : token_counts) {
        total_tokens += count;
    }
    return total_tokens;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: cpp_encode_probe MODEL_DIR REPEATS TARGET_BYTES THREADS CORPUS_FILE...\n";
        return 2;
    }

    const std::filesystem::path model_dir = argv[1];
    const int repeats = std::stoi(argv[2]);
    const size_t target_bytes = static_cast<size_t>(std::stoull(argv[3]));
    const size_t thread_count = std::max<size_t>(1, static_cast<size_t>(std::stoull(argv[4])));
    const bpe::Tokenizer tokenizer = bpe::Tokenizer::load(model_dir);

    for (int argi = 5; argi < argc; ++argi) {
        const std::filesystem::path corpus_path = argv[argi];
        const std::string text = read_file(corpus_path);
        const size_t n_bytes = text.size();
        const size_t bytes_per_service_batch = std::max<size_t>(n_bytes * thread_count, 1);
        const size_t batch = std::max<size_t>(1, target_bytes / bytes_per_service_batch);

        for (size_t i = 0; i < std::min<size_t>(batch, 8); ++i) {
            auto ids = tokenizer.encode(text);
            keep_ids_live(ids);
        }

        std::vector<double> durations;
        durations.reserve(static_cast<size_t>(repeats));
        size_t token_count = 0;
        for (int r = 0; r < repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            const size_t local_tokens = thread_count == 1
                ? encode_batch_single_thread(tokenizer, text, batch)
                : encode_batch_multi_thread(tokenizer, text, batch, thread_count);
            const auto end = std::chrono::steady_clock::now();
            durations.push_back(std::chrono::duration<double>(end - start).count());
            token_count = local_tokens / (batch * thread_count);
        }

        const auto ids = tokenizer.encode(text);
        std::vector<double> per_call_seconds;
        per_call_seconds.reserve(durations.size());
        const double calls_per_repeat = static_cast<double>(batch * thread_count);
        for (double duration : durations) {
            per_call_seconds.push_back(duration / calls_per_repeat);
        }
        const double seconds = percentile(durations, 0.50);
        const double mib_s = (static_cast<double>(n_bytes) * calls_per_repeat) /
            seconds / (1024.0 * 1024.0);

        std::cout << corpus_path.stem().string() << '\t'
                  << mib_s << '\t'
                  << token_count << '\t'
                  << hash_ids(ids) << '\t'
                  << batch << '\t'
                  << repeats << '\t'
                  << thread_count << '\t'
                  << percentile(per_call_seconds, 0.50) << '\t'
                  << percentile(per_call_seconds, 0.95) << '\t'
                  << percentile(per_call_seconds, 0.99) << '\t'
                  << peak_rss_bytes() << '\n';
    }
}
