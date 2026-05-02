#include "bpe/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void keep_ids_live(const std::vector<uint32_t>& ids) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(ids.data()), "g"(ids.size()) : "memory");
#else
    static volatile size_t sink = 0;
    sink += ids.size();
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: cpp_encode_probe MODEL_DIR REPEATS TARGET_BYTES CORPUS_FILE...\n";
        return 2;
    }

    const std::filesystem::path model_dir = argv[1];
    const int repeats = std::stoi(argv[2]);
    const size_t target_bytes = static_cast<size_t>(std::stoull(argv[3]));
    const bpe::Tokenizer tokenizer = bpe::Tokenizer::load(model_dir);

    for (int argi = 4; argi < argc; ++argi) {
        const std::filesystem::path corpus_path = argv[argi];
        const std::string text = read_file(corpus_path);
        const size_t n_bytes = text.size();
        const size_t batch = std::max<size_t>(1, target_bytes / std::max<size_t>(n_bytes, 1));

        for (size_t i = 0; i < std::min<size_t>(batch, 8); ++i) {
            auto ids = tokenizer.encode(text);
            keep_ids_live(ids);
        }

        std::vector<double> durations;
        durations.reserve(static_cast<size_t>(repeats));
        size_t token_count = 0;
        for (int r = 0; r < repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            size_t local_tokens = 0;
            for (size_t i = 0; i < batch; ++i) {
                auto ids = tokenizer.encode(text);
                local_tokens += ids.size();
                keep_ids_live(ids);
            }
            const auto end = std::chrono::steady_clock::now();
            durations.push_back(std::chrono::duration<double>(end - start).count());
            token_count = local_tokens / batch;
        }

        const auto ids = tokenizer.encode(text);
        const double seconds = median(std::move(durations));
        const double mib_s = (static_cast<double>(n_bytes) * static_cast<double>(batch)) /
            seconds / (1024.0 * 1024.0);

        std::cout << corpus_path.stem().string() << '\t'
                  << mib_s << '\t'
                  << token_count << '\t'
                  << hash_ids(ids) << '\n';
    }
}
