#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace bpe {

// GPT-2 style regex pre-tokenizer.
//
// Splits text into coarse chunks before BPE merging. The regex pattern
// matches (in order of priority):
//   - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
//   - Words (with optional leading space)
//   - Digit runs
//   - Punctuation/symbol runs
//   - Whitespace runs (non-newline)
//
// This prevents BPE from merging across word/punctuation boundaries.
//
// The GPT-2 pattern is:
//   R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
//
// For C++, a reasonable ASCII approximation (sufficient for this assignment):
//   R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)"

// Returns the pre-tokenized chunks of `text`.
std::vector<std::string> pretokenize(std::string_view text);

// Variant that also strips special tokens first, returning:
//   { {chunk, is_special}, ... }
// where is_special=true means the chunk is a special token literal.
struct PreToken {
    std::string text;
    bool is_special;
};

std::vector<PreToken> pretokenize_with_specials(
    std::string_view text,
    const std::vector<std::string>& special_tokens);

} // namespace bpe
