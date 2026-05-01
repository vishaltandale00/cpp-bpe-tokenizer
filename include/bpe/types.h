#pragma once

#include <cstdint>

namespace bpe {

// A merge rule: two token IDs that merge into a new one.
struct MergeRule {
    uint32_t left;
    uint32_t right;
    uint32_t merged; // the resulting token ID
};

} // namespace bpe
