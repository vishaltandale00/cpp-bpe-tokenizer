#pragma once

#include "bpe/types.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {

// ── Encoding helpers ────────────────────────────────────────────────

// Convert a pre-tokenized chunk (raw bytes) into byte-level token IDs.
std::vector<uint32_t> bytes_to_ids(std::string_view chunk);

// Apply learned merges to a sequence of token IDs (in merge priority order).
// This is the core of BPE encoding.
std::vector<uint32_t> apply_merges(
    std::vector<uint32_t> ids,
    const std::vector<MergeRule>& merges);

// ── Decoding helpers ────────────────────────────────────────────────

// Convert token IDs back to a byte string using the vocab.
// Throws std::out_of_range when an ID is not present in the vocab.
std::string ids_to_bytes(
    const std::vector<uint32_t>& ids,
    const std::unordered_map<uint32_t, std::vector<uint8_t>>& vocab);

} // namespace bpe
