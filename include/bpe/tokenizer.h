#pragma once

#include "bpe/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bpe {

// The core BPE tokenizer.
//
// Algorithm overview (GPT-2 style byte-level BPE):
//   1. Pre-tokenize text with regex into coarse chunks
//   2. Convert each chunk to a sequence of byte tokens (IDs 0-255)
//   3. Iteratively find the most frequent adjacent pair across all chunks,
//      merge it into a new token, repeat until vocab_size reached
//   4. Encoding: apply learned merges in priority order
//   5. Decoding: map token IDs back to byte sequences, decode as UTF-8
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    Tokenizer(const Tokenizer& other);
    Tokenizer& operator=(const Tokenizer& other);
    Tokenizer(Tokenizer&& other) noexcept;
    Tokenizer& operator=(Tokenizer&& other) noexcept;

    // ── Training ────────────────────────────────────────────────────
    // Train BPE on `text` until vocabulary reaches `vocab_size`.
    // `special_tokens` are reserved (e.g. "<|endoftext|>") and must not
    // participate in merges. They get IDs after the learned vocab.
    //
    // Training mutates the tokenizer and must not run concurrently with
    // encode/decode calls on the same instance.
    void train(const std::string& text,
               uint32_t vocab_size,
               const std::vector<std::string>& special_tokens = {});

    // ── Encoding / Decoding ─────────────────────────────────────────
    // Encode text to token IDs. Special tokens in the text are matched
    // literally and encoded to their reserved IDs.
    //
    // Concurrent encode calls are safe after train() or load() returns.
    std::vector<uint32_t> encode(const std::string& text) const;

    // Decode token IDs back to a UTF-8 string.
    std::string decode(const std::vector<uint32_t>& ids) const;

    // ── Persistence ─────────────────────────────────────────────────
    // Save vocab.json + merges.txt + special_tokens.txt to `dir`.
    void save(const std::filesystem::path& dir) const;

    // Load a previously saved tokenizer from `dir`.
    // Throws std::runtime_error when required files are missing, unreadable,
    // malformed, or semantically invalid.
    static Tokenizer load(const std::filesystem::path& dir);

    // ── Accessors (used by tests) ───────────────────────────────────
    uint32_t vocab_size() const;
    const std::vector<MergeRule>& merges() const;
    const std::unordered_map<uint32_t, std::vector<uint8_t>>& vocab() const;
    const std::unordered_map<std::string, uint32_t>& special_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bpe
