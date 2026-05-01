#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {

// Pair of adjacent token IDs.
struct TokenPair {
    uint32_t left;
    uint32_t right;

    bool operator==(const TokenPair& o) const {
        return left == o.left && right == o.right;
    }
};

struct TokenPairHash {
    size_t operator()(const TokenPair& p) const {
        // Combine two 32-bit IDs into one 64-bit hash.
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(p.left) << 32) | p.right);
    }
};

// Result of one training step: the most frequent pair and its count.
struct MergeCandidate {
    TokenPair pair;
    uint64_t count;
};

// ── Training helpers ────────────────────────────────────────────────
// Core primitives used by the BPE training loop.

// Count all adjacent pairs across pre-tokenized chunks.
// Each chunk is a sequence of token IDs.
// Returns a map from pair → total frequency.
std::unordered_map<TokenPair, uint64_t, TokenPairHash>
count_pairs(const std::vector<std::vector<uint32_t>>& chunks);

// Find the most frequent pair. Ties broken by larger pair value
// (i.e. lexicographically greater when comparing (left, right)).
MergeCandidate find_best_pair(
    const std::unordered_map<TokenPair, uint64_t, TokenPairHash>& counts);

// Apply a merge: replace all occurrences of (left, right) with `new_id`
// in every chunk. Modifies chunks in-place.
void apply_merge(std::vector<std::vector<uint32_t>>& chunks,
                 uint32_t left, uint32_t right, uint32_t new_id);

// Full training loop. Returns the learned merge rules.
// `pre_tokenized` is the text already split into chunks, each chunk
// converted to byte-token IDs (0-255).
// Trains until vocab reaches `target_vocab_size` (256 + num_merges).
std::vector<std::pair<TokenPair, uint32_t>>
train_bpe(std::vector<std::vector<uint32_t>>& chunks,
          uint32_t target_vocab_size);

} // namespace bpe
