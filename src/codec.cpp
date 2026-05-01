// Consolidated implementation for the tokenizer hot path. The public API
// signatures live in include/bpe/.
#include "bpe/codec.h"
#include "bpe/pretokenizer.h"
#include "bpe/tokenizer.h"
#include "bpe/trainer.h"
#include "bpe/utils.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <cctype>
#include <filesystem>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <array>
#include <queue>
#include <functional>
#include <system_error>

namespace bpe {

namespace {

struct StringViewHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        for (char c : sv) {
            h ^= static_cast<size_t>(static_cast<unsigned char>(c));
            h *= 0x100000001b3ULL;
        }
        return h;
    }
    size_t operator()(const std::string& s) const noexcept {
        return operator()(std::string_view(s));
    }
};

struct StringViewEqual {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept {
        return a == b;
    }
};

using RanksMap = std::unordered_map<std::string, uint32_t, StringViewHash, StringViewEqual>;

struct MergeStep {
    uint32_t rank = UINT32_MAX;
    uint32_t merged = UINT32_MAX;
};

enum class EncodeMode {
    PairRules,
    ByteSpanRanks,
};

std::vector<uint32_t> encode_pair_rules_for_cache(
    std::string_view piece,
    const std::array<uint32_t, 256>& byte_lookup,
    const std::array<MergeStep, 65536>& base_pair_lookup,
    const std::unordered_map<TokenPair, MergeStep, TokenPairHash>& merge_lookup)
{
    std::vector<uint32_t> ids;
    ids.reserve(piece.size());
    for (unsigned char byte : piece) {
        ids.push_back(byte_lookup[byte]);
    }
    if (ids.size() < 2 || merge_lookup.empty()) {
        return ids;
    }

    auto get_pair_merge = [&](uint32_t left, uint32_t right) -> MergeStep {
        if (left < 256 && right < 256) {
            return base_pair_lookup[(left << 8) | right];
        }
        auto it = merge_lookup.find(TokenPair{left, right});
        return it != merge_lookup.end() ? it->second : MergeStep{};
    };

    while (ids.size() > 1) {
        uint32_t best_rank = UINT32_MAX;
        size_t best_pos = ids.size();
        uint32_t best_merged = UINT32_MAX;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            const MergeStep merge = get_pair_merge(ids[i], ids[i + 1]);
            if (merge.rank < best_rank) {
                best_rank = merge.rank;
                best_pos = i;
                best_merged = merge.merged;
            }
        }
        if (best_rank == UINT32_MAX) {
            break;
        }
        ids[best_pos] = best_merged;
        ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best_pos + 1));
    }
    return ids;
}

bool parse_u32(std::string_view text, uint32_t& out)
{
    if (text.empty()) {
        return false;
    }
    uint32_t value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) {
        return false;
    }
    out = value;
    return true;
}

std::runtime_error load_error(const std::filesystem::path& path, const std::string& message)
{
    return std::runtime_error("Invalid tokenizer model at " + path.string() + ": " + message);
}

std::runtime_error save_error(const std::filesystem::path& path, const std::string& message)
{
    return std::runtime_error("Failed to save tokenizer model at " + path.string() + ": " + message);
}

std::runtime_error train_error(const std::string& message)
{
    return std::runtime_error("Invalid special token set: " + message);
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        out.push_back(kHex[byte >> 4]);
        out.push_back(kHex[byte & 0x0f]);
    }
    return out;
}

uint32_t checked_special_token_count(size_t count)
{
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw train_error("too many special tokens");
    }
    return static_cast<uint32_t>(count);
}

void validate_special_token_text(std::string_view token)
{
    if (token.empty()) {
        throw train_error("special tokens must not be empty");
    }
    if (token.front() == ' ' || token.front() == '\t') {
        throw train_error("special tokens must not start with space or tab");
    }
    if (token.find('\n') != std::string_view::npos ||
        token.find('\r') != std::string_view::npos)
    {
        throw train_error("special tokens must not contain CR or LF");
    }
}

void validate_special_tokens(const std::vector<std::string>& special_tokens)
{
    std::unordered_set<std::string> seen;
    seen.reserve(special_tokens.size());
    for (const auto& token : special_tokens) {
        validate_special_token_text(token);
        if (!seen.insert(token).second) {
            throw train_error("duplicate special token");
        }
    }
}

void validate_special_token_vocab_collisions(
    const std::vector<std::string>& special_tokens,
    const std::unordered_map<uint32_t, std::vector<uint8_t>>& vocab)
{
    std::unordered_set<std::string> vocab_tokens;
    vocab_tokens.reserve(vocab.size());
    for (const auto& [_, bytes] : vocab) {
        vocab_tokens.emplace(bytes.begin(), bytes.end());
    }

    for (const auto& token : special_tokens) {
        if (vocab_tokens.find(token) != vocab_tokens.end()) {
            throw train_error("special token collides with normal vocab byte string");
        }
    }
}

uint32_t checked_vocab_count_for_specials(
    const std::unordered_map<uint32_t, std::vector<uint8_t>>& vocab,
    size_t special_count)
{
    if (vocab.size() > std::numeric_limits<uint32_t>::max()) {
        throw train_error("normal vocab is too large to assign special token IDs");
    }

    const uint32_t vocab_count = static_cast<uint32_t>(vocab.size());
    if (special_count > std::numeric_limits<uint32_t>::max() - vocab_count) {
        throw train_error("too many special tokens to assign IDs");
    }
    return vocab_count;
}

template <typename Writer>
void write_checked_file(const std::filesystem::path& path, Writer writer)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw save_error(path, "failed to open file");
    }

    writer(file);
    if (!file) {
        throw save_error(path, "failed to write file");
    }

    file.flush();
    if (!file) {
        throw save_error(path, "failed to flush file");
    }

    file.close();
    if (!file) {
        throw save_error(path, "failed to close file");
    }
}

uint32_t json_u32(const nlohmann::json& value, const std::filesystem::path& path, const std::string& field)
{
    uint64_t parsed = 0;
    if (value.is_number_unsigned()) {
        parsed = value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value < 0) {
            throw load_error(path, field + " must be non-negative");
        }
        parsed = static_cast<uint64_t>(signed_value);
    } else {
        throw load_error(path, field + " must be an integer");
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw load_error(path, field + " exceeds uint32_t");
    }
    return static_cast<uint32_t>(parsed);
}

std::vector<std::string_view> split_ws(std::string_view line)
{
    std::vector<std::string_view> parts;
    size_t pos = 0;
    while (pos < line.size()) {
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        const size_t start = pos;
        while (pos < line.size() && !std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        if (pos > start) {
            parts.emplace_back(line.substr(start, pos - start));
        }
    }
    return parts;
}

std::string trim_line(std::string line)
{
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

} // namespace

class TokenTrie {
public:
    static constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();

    TokenTrie() {
        clear();
    }

    void clear() {
        build_nodes_.clear();
        build_nodes_.push_back(BuildNode{});
        nodes_.clear();
        edges_.clear();
        prefix2_.fill(kInvalid);
        finalized_ = false;
    }

    void reserve(size_t n) {
        if (n > build_nodes_.capacity()) {
            build_nodes_.reserve(n);
        }
        if (n > nodes_.capacity()) {
            nodes_.reserve(n);
        }
    }

    void insert(std::string_view key, uint32_t value) {
        if (key.empty()) {
            return;
        }
        finalized_ = false;

        uint32_t node = 0;
        size_t pos = 0;
        if (key.size() >= 2) {
            const uint32_t prefix =
                (static_cast<uint32_t>(static_cast<uint8_t>(key[0])) << 8) |
                static_cast<uint8_t>(key[1]);
            node = prefix2_[prefix];
            if (node == kInvalid) {
                const uint32_t first = ensure_child(0, static_cast<uint8_t>(key[0]));
                node = ensure_child(first, static_cast<uint8_t>(key[1]));
                prefix2_[prefix] = node;
            }
            pos = 2;
        } else {
            node = ensure_child(0, static_cast<uint8_t>(key[0]));
            pos = 1;
        }

        for (; pos < key.size(); ++pos) {
            node = ensure_child(node, static_cast<uint8_t>(key[pos]));
        }

        if (build_nodes_[node].value == kInvalid || value < build_nodes_[node].value) {
            build_nodes_[node].value = value;
        }
    }

    void finalize() {
        if (finalized_) {
            return;
        }

        nodes_.clear();
        edges_.clear();
        nodes_.resize(build_nodes_.size());

        size_t total_edges = 0;
        for (const auto& node : build_nodes_) {
            total_edges += node.children.size();
        }
        edges_.reserve(total_edges);

        for (size_t i = 0; i < build_nodes_.size(); ++i) {
            auto& children = build_nodes_[i].children;
            std::sort(children.begin(), children.end(), [](const BuildChild& a, const BuildChild& b) {
                return a.byte < b.byte;
            });

            nodes_[i].value = build_nodes_[i].value;
            nodes_[i].first_edge = static_cast<uint32_t>(edges_.size());
            nodes_[i].edge_count = static_cast<uint32_t>(children.size());
            for (const auto& child : children) {
                edges_.push_back(Edge{child.byte, child.child});
            }
        }

        finalized_ = true;
    }

    const uint32_t* find(std::string_view key) const {
        const uint32_t node = find_node(key);
        return node != kInvalid && nodes_[node].value != kInvalid ? &nodes_[node].value : nullptr;
    }

private:
    struct Edge {
        uint8_t byte = 0;
        uint32_t child = kInvalid;
    };

    struct Node {
        uint32_t value = kInvalid;
        uint32_t first_edge = 0;
        uint32_t edge_count = 0;
    };

    struct BuildChild {
        uint8_t byte = 0;
        uint32_t child = kInvalid;
    };

    struct BuildNode {
        uint32_t value = kInvalid;
        std::vector<BuildChild> children;
    };

    uint32_t find_node(std::string_view key) const {
        if (key.empty() || nodes_.empty()) {
            return kInvalid;
        }

        uint32_t node = 0;
        size_t pos = 0;
        if (key.size() >= 2) {
            const uint32_t prefix =
                (static_cast<uint32_t>(static_cast<uint8_t>(key[0])) << 8) |
                static_cast<uint8_t>(key[1]);
            node = prefix2_[prefix];
            if (node == kInvalid) {
                return kInvalid;
            }
            pos = 2;
        } else {
            node = find_child(0, static_cast<uint8_t>(key[0]));
            if (node == kInvalid) {
                return kInvalid;
            }
            pos = 1;
        }

        for (; pos < key.size(); ++pos) {
            node = find_child(node, static_cast<uint8_t>(key[pos]));
            if (node == kInvalid) {
                return kInvalid;
            }
        }

        return node;
    }

    uint32_t find_child(uint32_t parent, uint8_t byte) const {
        const Node& node = nodes_[parent];
        if (node.edge_count == 0) {
            return kInvalid;
        }

        const Edge* base = edges_.data() + node.first_edge;
        if (node.edge_count <= 8) {
            for (uint32_t i = 0; i < node.edge_count; ++i) {
                if (base[i].byte == byte) {
                    return base[i].child;
                }
            }
            return kInvalid;
        }

        uint32_t lo = 0;
        uint32_t hi = node.edge_count;
        while (lo < hi) {
            const uint32_t mid = lo + ((hi - lo) >> 1);
            if (base[mid].byte < byte) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < node.edge_count && base[lo].byte == byte) {
            return base[lo].child;
        }
        return kInvalid;
    }

    uint32_t ensure_child(uint32_t parent, uint8_t byte) {
        for (const auto& child : build_nodes_[parent].children) {
            if (child.byte == byte) {
                return child.child;
            }
        }

        const uint32_t child = static_cast<uint32_t>(build_nodes_.size());
        build_nodes_.push_back(BuildNode{});
        build_nodes_[parent].children.push_back(BuildChild{byte, child});
        return child;
    }

    std::vector<BuildNode> build_nodes_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    bool finalized_ = false;
    std::array<uint32_t, 65536> prefix2_{};
};

struct Tokenizer::Impl {
    Impl() {
        rebuild_encode_caches();
    }

    void rebuild_encode_caches() {
        std::string key(1, '\0');
        for (size_t i = 0; i < byte_lookup_.size(); ++i) {
            key[0] = static_cast<char>(i);
            auto it = token_to_id_.find(key);
            byte_lookup_[i] = it != token_to_id_.end()
                ? it->second
                : static_cast<uint32_t>(i);
        }

        merge_lookup_.clear();
        merge_lookup_.reserve(merges_.size());
        base_pair_lookup_.fill(MergeStep{});
        for (size_t rank = 0; rank < merges_.size(); ++rank) {
            const auto& merge = merges_[rank];
            const MergeStep step{
                static_cast<uint32_t>(rank),
                merge.merged,
            };
            merge_lookup_[TokenPair{merge.left, merge.right}] = step;
            if (merge.left < 256 && merge.right < 256) {
                base_pair_lookup_[(merge.left << 8) | merge.right] = step;
            }
        }

        token_trie_.clear();
        size_t approx_nodes = 1;
        for (const auto& [key_bytes, _] : token_to_id_) {
            approx_nodes += key_bytes.size();
        }
        token_trie_.reserve(approx_nodes);
        for (const auto& [key_bytes, id] : token_to_id_) {
            token_trie_.insert(std::string_view(key_bytes), id);
        }
        token_trie_.finalize();

        safe_token_trie_.clear();
        if (encode_mode_ == EncodeMode::PairRules) {
            approx_nodes = 1;
            for (const auto& [_, bytes] : vocab_) {
                approx_nodes += bytes.size();
            }
            safe_token_trie_.reserve(approx_nodes);
            for (const auto& [id, bytes] : vocab_) {
                std::string key(bytes.begin(), bytes.end());
                // Native vocab entries are only safe as whole-piece shortcuts if
                // the learned pair rules would produce exactly that token.
                auto encoded = encode_pair_rules_for_cache(
                    key, byte_lookup_, base_pair_lookup_, merge_lookup_);
                if (encoded.size() == 1 && encoded[0] == id) {
                    safe_token_trie_.insert(std::string_view(key), id);
                }
            }
        }
        safe_token_trie_.finalize();
    }

    std::unordered_map<uint32_t, std::vector<uint8_t>> vocab_;
    RanksMap token_to_id_;
    TokenTrie token_trie_;
    TokenTrie safe_token_trie_;
    std::unordered_map<TokenPair, MergeStep, TokenPairHash> merge_lookup_;
    std::vector<MergeRule> merges_;
    std::unordered_map<std::string, uint32_t> special_tokens_;
    EncodeMode encode_mode_ = EncodeMode::PairRules;
    std::array<uint32_t, 256> byte_lookup_{};
    std::array<MergeStep, 65536> base_pair_lookup_{};
};

// ─────────────────────────────────────────────────────────────────────
// Byte-level codec, GPT-2 style pre-tokenization, BPE training, and
// tokenizer persistence.
// ─────────────────────────────────────────────────────────────────────

// ---- codec.h: byte-level helpers -----------------------------------

std::vector<uint32_t> bytes_to_ids(std::string_view chunk)
{
    std::vector<uint32_t> ids{};
    ids.reserve(chunk.size());
    for (auto ch : chunk)
    {
        ids.emplace_back(static_cast<uint8_t>(ch));
    }
    return ids;
}

std::vector<uint32_t> apply_merges(
    std::vector<uint32_t> ids,
    const std::vector<MergeRule>& merges)
{
    for (const auto& m : merges)
    {
        std::vector<uint32_t> out;
        out.reserve(ids.size());
        for (size_t i = 0; i < ids.size();)
        {
            if (i + 1 < ids.size() && ids[i] == m.left && ids[i + 1] == m.right)
            {
                out.push_back(m.merged);
                i += 2;
            }
            else
            {
                out.push_back(ids[i]);
                ++i;
            }
        }
        ids.swap(out);
    }
    return ids;
}

std::string ids_to_bytes(
    const std::vector<uint32_t>& ids,
    const std::unordered_map<uint32_t, std::vector<uint8_t>>& vocab)
{
    std::string result;
    for (auto id : ids)
    {
        if (auto it = vocab.find(id); it != vocab.end())
        {
            result.append(it->second.begin(), it->second.end());
            continue;
        }
        throw std::out_of_range("Unknown token id: " + std::to_string(id));
    }
    return result;
}

// ---- pretokenizer.h: GPT-2 regex splitting -------------------------

// UTF-8 helpers
static inline size_t utf8_byte_len(unsigned char c)
{
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static inline uint32_t utf8_codepoint(const char* s, size_t len)
{
    auto u = [](char c) { return static_cast<unsigned char>(c); };
    if (len == 1) return u(s[0]);
    if (len == 2) return ((u(s[0]) & 0x1F) << 6) | (u(s[1]) & 0x3F);
    if (len == 3) return ((u(s[0]) & 0x0F) << 12) | ((u(s[1]) & 0x3F) << 6) | (u(s[2]) & 0x3F);
    return ((u(s[0]) & 0x07) << 18) | ((u(s[1]) & 0x3F) << 12) | ((u(s[2]) & 0x3F) << 6) | (u(s[3]) & 0x3F);
}

// Approximate Unicode \p{L} (letters)
static bool is_unicode_letter(uint32_t cp)
{
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x80) return false;
    // Latin-1 Supplement letters
    if ((cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0xFF)) return true;
    if (cp < 0x100) return false;
    // Latin Extended A/B, IPA, Spacing Modifiers
    if (cp <= 0x02FF) return true;
    // Combining Diacritical Marks
    if (cp >= 0x0300 && cp <= 0x036F) return true;
    // Greek, Cyrillic, Armenian, Hebrew, Arabic, etc.
    if (cp >= 0x0370 && cp <= 0x1FFF) return true;
    // CJK
    if (cp >= 0x3040 && cp <= 0x9FFF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true; // Hangul
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // General: non-space non-digit non-punctuation above BMP
    if (cp >= 0x10000) return true;
    return false;
}

// Approximate Unicode \p{N} (numbers)
static bool is_unicode_digit(uint32_t cp)
{
    return (cp >= '0' && cp <= '9');
}

// Unicode \s (whitespace)
static bool is_unicode_space(uint32_t cp)
{
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v') return true;
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    return false;
}

static inline bool is_ascii_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static inline bool is_ascii_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}
static inline bool is_ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Advance one UTF-8 character, return its codepoint and byte length.
static inline uint32_t next_cp(std::string_view text, size_t pos, size_t& len)
{
    len = utf8_byte_len(static_cast<unsigned char>(text[pos]));
    if (pos + len > text.size()) len = 1;
    return utf8_codepoint(&text[pos], len);
}

// Internal pretokenize returning string_views (no allocations).
// Caller must ensure the backing string outlives the views.
static void pretokenize_sv(std::string_view text,
                           std::vector<std::string_view>& out)
{
    size_t i = 0;
    const size_t n = text.size();

    while (i < n)
    {
        // Alt 1: contraction  '(?:[sdmt]|ll|ve|re)
        if (text[i] == '\'')
        {
            if (i + 1 < n)
            {
                char c = text[i + 1];
                if (c == 's' || c == 't' || c == 'm' || c == 'd')
                { out.emplace_back(text.substr(i, 2)); i += 2; continue; }
            }
            if (i + 2 < n)
            {
                char c1 = text[i + 1], c2 = text[i + 2];
                if ((c1 == 'l' && c2 == 'l') || (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e'))
                { out.emplace_back(text.substr(i, 3)); i += 3; continue; }
            }
        }

        // Alt 2:  ?\p{L}++
        {
            size_t j = i;
            if (j < n && text[j] == ' ') j++;
            size_t lstart = j;
            while (j < n) {
                unsigned char c = static_cast<unsigned char>(text[j]);
                if (c < 0x80) {
                    if (is_ascii_letter(c)) { ++j; continue; }
                    break;
                }
                size_t cl; uint32_t cp = next_cp(text, j, cl);
                if (!is_unicode_letter(cp)) break;
                j += cl;
            }
            if (j > lstart)
            { out.emplace_back(text.substr(i, j - i)); i = j; continue; }
        }

        // Alt 3:  ?\p{N}++
        {
            size_t j = i;
            if (j < n && text[j] == ' ') j++;
            size_t dstart = j;
            while (j < n) {
                unsigned char c = static_cast<unsigned char>(text[j]);
                if (is_ascii_digit(c)) { ++j; continue; }
                if (c >= 0x80) {
                    size_t cl; uint32_t cp = next_cp(text, j, cl);
                    if (!is_unicode_digit(cp)) break;
                    j += cl;
                    continue;
                }
                break;
            }
            if (j > dstart)
            { out.emplace_back(text.substr(i, j - i)); i = j; continue; }
        }

        // Alt 4:  ?[^\s\p{L}\p{N}]++
        {
            size_t j = i;
            if (j < n && text[j] == ' ') j++;
            size_t ostart = j;
            while (j < n)
            {
                unsigned char c = static_cast<unsigned char>(text[j]);
                if (c < 0x80) {
                    if (is_ascii_letter(c) || is_ascii_digit(c) || is_ascii_space(c))
                        break;
                    ++j;
                    continue;
                }
                size_t cl; uint32_t cp = next_cp(text, j, cl);
                if (is_unicode_space(cp) || is_unicode_letter(cp) || is_unicode_digit(cp)) break;
                j += cl;
            }
            if (j > ostart)
            { out.emplace_back(text.substr(i, j - i)); i = j; continue; }
        }

        // Alt 5: \s++$
        {
            size_t j = i;
            while (j < n)
            {
                size_t cl; uint32_t cp = next_cp(text, j, cl);
                if (!is_unicode_space(cp)) break;
                j += cl;
            }
            if (j > i && j == n)
            { out.emplace_back(text.substr(i, j - i)); i = j; continue; }
        }

        // Alt 6: \s+(?!\S)
        {
            size_t j = i;
            size_t prev = i;
            while (j < n)
            {
                size_t cl; uint32_t cp = next_cp(text, j, cl);
                if (!is_unicode_space(cp)) break;
                prev = j;
                j += cl;
            }
            if (j > i && prev > i)
            { out.emplace_back(text.substr(i, prev - i)); i = prev; continue; }
        }

        // Alt 7: \s
        {
            size_t cl; uint32_t cp = next_cp(text, i, cl);
            if (is_unicode_space(cp))
            { out.emplace_back(text.substr(i, cl)); i += cl; continue; }
        }

        // Fallback
        {
            size_t cl = utf8_byte_len(static_cast<unsigned char>(text[i]));
            if (i + cl > n) cl = 1;
            out.emplace_back(text.substr(i, cl));
            i += cl;
        }
    }
}

std::vector<std::string> pretokenize(std::string_view text)
{
    std::vector<std::string_view> svs;
    svs.reserve(text.size() / 4);
    pretokenize_sv(text, svs);
    std::vector<std::string> out;
    out.reserve(svs.size());
    for (auto sv : svs) out.emplace_back(sv);
    return out;
}

std::vector<PreToken> pretokenize_with_specials(
    std::string_view text,
    const std::vector<std::string>& special_tokens)
{
    std::vector<std::string> specs = special_tokens;
    for (const auto& tok : specs) {
        if (tok.empty()) {
            throw std::invalid_argument("Special tokens must not be empty");
        }
    }
    std::sort(specs.begin(), specs.end(), [](const std::string& a, const std::string& b) {
        return a.size() > b.size();
    });

    std::vector<PreToken> out;
    size_t pos = 0;
    while (pos < text.size())
    {
        size_t best_pos = std::string_view::npos;
        const std::string* best_tok = nullptr;
        for (const auto& tok : specs)
        {
            size_t p = text.find(tok, pos);
            if (p != std::string_view::npos && (best_tok == nullptr || p < best_pos))
            {
                best_pos = p;
                best_tok = &tok;
                if (p == pos) break;
            }
        }

        if (!best_tok)
        {
            for (auto& t : pretokenize(text.substr(pos))) out.push_back({t, false});
            break;
        }

        if (best_pos > pos)
        {
            for (auto& t : pretokenize(text.substr(pos, best_pos - pos))) out.push_back({t, false});
        }
        out.push_back({*best_tok, true});
        pos = best_pos + best_tok->size();
    }
    return out;
}

// ---- trainer.h: BPE training primitives ----------------------------

std::unordered_map<TokenPair, uint64_t, TokenPairHash>
count_pairs(const std::vector<std::vector<uint32_t>>& chunks)
{
    std::unordered_map<TokenPair, uint64_t, TokenPairHash> counts;
    for (const auto& chunk : chunks)
    {
        for (size_t i = 0; i + 1 < chunk.size(); ++i)
        {
            ++counts[TokenPair{chunk[i], chunk[i + 1]}];
        }
    }
    return counts;
}

MergeCandidate find_best_pair(
    const std::unordered_map<TokenPair, uint64_t, TokenPairHash>& counts)
{
    MergeCandidate best{};
    best.count = 0;
    for (const auto& [pair, count] : counts)
    {
        if (count > best.count ||
            (count == best.count &&
             (pair.left > best.pair.left ||
              (pair.left == best.pair.left && pair.right > best.pair.right))))
        {
            best.pair = pair;
            best.count = count;
        }
    }
    return best;
}

void apply_merge(std::vector<std::vector<uint32_t>>& chunks,
                 uint32_t left, uint32_t right, uint32_t new_id)
{
    for (auto& chunk : chunks)
    {
        std::vector<uint32_t> out;
        out.reserve(chunk.size());
        for (size_t i = 0; i < chunk.size();)
        {
            if (i + 1 < chunk.size() && chunk[i] == left && chunk[i + 1] == right)
            {
                out.push_back(new_id);
                i += 2;
            }
            else
            {
                out.push_back(chunk[i]);
                ++i;
            }
        }
        chunk.swap(out);
    }
}

std::vector<std::pair<TokenPair, uint32_t>>
train_bpe(std::vector<std::vector<uint32_t>>& chunks,
          uint32_t target_vocab_size)
{
    std::vector<std::pair<TokenPair, uint32_t>> learned;
    uint32_t next_id = 256;
    while (next_id < target_vocab_size)
    {
        auto counts = count_pairs(chunks);
        auto best = find_best_pair(counts);
        if (best.count == 0) break;
        apply_merge(chunks, best.pair.left, best.pair.right, next_id);
        learned.push_back({best.pair, next_id});
        ++next_id;
    }
    return learned;
}

// ---- tokenizer.h: Tokenizer class ----------------------------------

Tokenizer::Tokenizer()
    : impl_(std::make_unique<Impl>())
{
}

Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(const Tokenizer& other)
    : impl_(std::make_unique<Impl>(*other.impl_))
{
}

Tokenizer& Tokenizer::operator=(const Tokenizer& other)
{
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

Tokenizer::Tokenizer(Tokenizer&& other) noexcept = default;

Tokenizer& Tokenizer::operator=(Tokenizer&& other) noexcept = default;

void Tokenizer::train(const std::string& text,
                      uint32_t vocab_size,
                      const std::vector<std::string>& special_tokens)
{
    validate_special_tokens(special_tokens);
    const uint32_t special_count = checked_special_token_count(special_tokens.size());

    std::unordered_map<uint32_t, std::vector<uint8_t>> vocab;
    std::vector<MergeRule> merges;
    std::unordered_map<std::string, uint32_t> trained_special_tokens;
    RanksMap token_to_id;

    for (uint32_t i = 0; i < 256; ++i) vocab[i] = {static_cast<uint8_t>(i)};

    std::vector<std::vector<uint32_t>> chunks;
    for (const auto& pt : pretokenize_with_specials(text, special_tokens))
    {
        if (!pt.is_special) chunks.push_back(bytes_to_ids(pt.text));
    }

    uint32_t train_target = vocab_size > special_count
        ? vocab_size - special_count
        : 256;

    auto learned = train_bpe(chunks, std::max<uint32_t>(256, train_target));
    for (const auto& [pair, id] : learned)
    {
        merges.push_back(MergeRule{pair.left, pair.right, id});
        std::vector<uint8_t> merged = vocab[pair.left];
        const auto& rhs = vocab[pair.right];
        merged.insert(merged.end(), rhs.begin(), rhs.end());
        vocab[id] = std::move(merged);
    }

    validate_special_token_vocab_collisions(special_tokens, vocab);

    uint32_t next_special = checked_vocab_count_for_specials(vocab, special_tokens.size());
    for (const auto& s : special_tokens) {
        if (vocab.find(next_special) != vocab.end()) {
            throw train_error("special token ID collides with normal vocab token ID");
        }
        trained_special_tokens.emplace(s, next_special++);
    }

    for (const auto& [id, bytes] : vocab)
    {
        token_to_id[std::string(bytes.begin(), bytes.end())] = id;
    }
    for (const auto& [s, id] : trained_special_tokens) token_to_id[s] = id;

    impl_->vocab_ = std::move(vocab);
    impl_->merges_ = std::move(merges);
    impl_->special_tokens_ = std::move(trained_special_tokens);
    impl_->token_to_id_ = std::move(token_to_id);
    impl_->encode_mode_ = EncodeMode::PairRules;
    impl_->rebuild_encode_caches();
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const
{
    const auto& byte_lookup_ = impl_->byte_lookup_;
    const auto& base_pair_lookup_ = impl_->base_pair_lookup_;
    const auto& merge_lookup_ = impl_->merge_lookup_;
    const auto& token_trie_ = impl_->token_trie_;
    const auto& safe_token_trie_ = impl_->safe_token_trie_;
    const auto& special_tokens_ = impl_->special_tokens_;
    const EncodeMode encode_mode = impl_->encode_mode_;

    std::vector<uint32_t> out;
    out.reserve(text.size());

    auto get_pair_merge = [&](uint32_t left, uint32_t right) -> MergeStep {
        if (left < 256 && right < 256) {
            return base_pair_lookup_[(left << 8) | right];
        }
        auto it = merge_lookup_.find(TokenPair{left, right});
        return it != merge_lookup_.end() ? it->second : MergeStep{};
    };

    auto encode_piece = [&](std::string_view piece) {
        const size_t pn = piece.size();
        if (pn == 0) return;
        if (pn > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::length_error("Tokenization chunk is too large");
        }
        if (encode_mode == EncodeMode::ByteSpanRanks) {
            if (const uint32_t* found = token_trie_.find(piece); found != nullptr) {
                out.push_back(*found);
                return;
            }
        } else if (const uint32_t* found = safe_token_trie_.find(piece); found != nullptr) {
            out.push_back(*found);
            return;
        }
        if (pn == 1 || merge_lookup_.empty()) {
            for (unsigned char byte : piece) {
                out.push_back(byte_lookup_[byte]);
            }
            return;
        }
        if (encode_mode == EncodeMode::PairRules && pn == 2) {
            const uint32_t left = byte_lookup_[static_cast<unsigned char>(piece[0])];
            const uint32_t right = byte_lookup_[static_cast<unsigned char>(piece[1])];
            const MergeStep merge = get_pair_merge(left, right);
            if (merge.rank != UINT32_MAX) {
                out.push_back(merge.merged);
            } else {
                out.push_back(left);
                out.push_back(right);
            }
            return;
        }
        if (encode_mode == EncodeMode::PairRules && pn == 3) {
            const uint32_t t0 = byte_lookup_[static_cast<unsigned char>(piece[0])];
            const uint32_t t1 = byte_lookup_[static_cast<unsigned char>(piece[1])];
            const uint32_t t2 = byte_lookup_[static_cast<unsigned char>(piece[2])];
            const MergeStep left_merge = get_pair_merge(t0, t1);
            const MergeStep right_merge = get_pair_merge(t1, t2);
            if (left_merge.rank == UINT32_MAX && right_merge.rank == UINT32_MAX) {
                out.push_back(t0);
                out.push_back(t1);
                out.push_back(t2);
            } else if (left_merge.rank <= right_merge.rank) {
                const MergeStep chained = get_pair_merge(left_merge.merged, t2);
                if (chained.rank != UINT32_MAX) {
                    out.push_back(chained.merged);
                } else {
                    out.push_back(left_merge.merged);
                    out.push_back(t2);
                }
            } else {
                const MergeStep chained = get_pair_merge(t0, right_merge.merged);
                if (chained.rank != UINT32_MAX) {
                    out.push_back(chained.merged);
                } else {
                    out.push_back(t0);
                    out.push_back(right_merge.merged);
                }
            }
            return;
        }

        struct LLNode {
            uint32_t token;
            uint32_t start;
            uint32_t end;
            uint32_t rank;
            uint32_t merged;
            uint32_t version;
            int32_t prev;
            int32_t next;
            bool alive;
        };
        struct HeapEntry {
            uint32_t rank;
            uint32_t start;
            uint32_t version;
            uint32_t merged;
            int32_t idx;
        };
        thread_local std::vector<LLNode> ll;
        thread_local std::vector<HeapEntry> heap;
        ll.resize(pn);
        heap.clear();
        heap.reserve(pn * 2);
        for (size_t i = 0; i < pn; ++i)
        {
            ll[i].token = byte_lookup_[static_cast<unsigned char>(piece[i])];
            ll[i].start = static_cast<uint32_t>(i);
            ll[i].end = static_cast<uint32_t>(i + 1);
            ll[i].rank = UINT32_MAX;
            ll[i].merged = UINT32_MAX;
            ll[i].version = 0;
            ll[i].prev = static_cast<int32_t>(i) - 1;
            ll[i].next = (i + 1 < pn) ? static_cast<int32_t>(i + 1) : -1;
            ll[i].alive = true;
        }

        auto get_merge = [&](int32_t idx) -> MergeStep {
            if (idx < 0 || !ll[idx].alive) return MergeStep{};
            int32_t n1 = ll[idx].next;
            if (n1 < 0 || !ll[n1].alive) return MergeStep{};
            if (encode_mode == EncodeMode::ByteSpanRanks) {
                const size_t start = ll[idx].start;
                const size_t end = ll[n1].end;
                if (const uint32_t* found = token_trie_.find(piece.substr(start, end - start)); found != nullptr) {
                    return MergeStep{*found, *found};
                }
                return MergeStep{};
            }
            return get_pair_merge(ll[idx].token, ll[n1].token);
        };

        auto heap_less = [](const HeapEntry& a, const HeapEntry& b) {
            if (a.rank != b.rank) {
                return a.rank > b.rank;
            }
            return a.start > b.start;
        };

        auto push_rank = [&](int32_t idx) {
            if (idx < 0 || !ll[idx].alive || ll[idx].rank == UINT32_MAX) {
                return;
            }
            heap.push_back(HeapEntry{ll[idx].rank, ll[idx].start, ll[idx].version, ll[idx].merged, idx});
            std::push_heap(heap.begin(), heap.end(), heap_less);
        };

        auto refresh_rank = [&](int32_t idx) {
            if (idx < 0 || !ll[idx].alive) {
                return;
            }
            ++ll[idx].version;
            const MergeStep merge = get_merge(idx);
            ll[idx].rank = merge.rank;
            ll[idx].merged = merge.merged;
            push_rank(idx);
        };

        for (size_t i = 0; i + 1 < pn; ++i)
        {
            refresh_rank(static_cast<int32_t>(i));
        }

        while (!heap.empty())
        {
            std::pop_heap(heap.begin(), heap.end(), heap_less);
            HeapEntry next = heap.back();
            heap.pop_back();

            const int32_t min_idx = next.idx;
            if (!ll[min_idx].alive ||
                ll[min_idx].version != next.version ||
                ll[min_idx].rank != next.rank ||
                ll[min_idx].merged != next.merged)
            {
                continue;
            }

            int32_t to_remove = ll[min_idx].next;
            if (to_remove < 0 || !ll[to_remove].alive) {
                continue;
            }

            int32_t after = ll[to_remove].next;
            ll[min_idx].token = next.merged;
            ll[min_idx].end = ll[to_remove].end;
            ll[min_idx].next = after;
            ll[min_idx].rank = UINT32_MAX;
            ll[min_idx].merged = UINT32_MAX;
            ++ll[min_idx].version;
            if (after >= 0) ll[after].prev = min_idx;
            ll[to_remove].alive = false;
            ll[to_remove].rank = UINT32_MAX;
            ll[to_remove].merged = UINT32_MAX;
            ++ll[to_remove].version;

            int32_t pred = ll[min_idx].prev;
            refresh_rank(min_idx);
            refresh_rank(pred);
        }

        for (int32_t cur = 0; cur >= 0; cur = ll[cur].next) {
            out.push_back(ll[cur].token);
        }
    };

    if (special_tokens_.empty())
    {
        thread_local std::vector<std::string_view> sv_chunks;
        sv_chunks.clear();
        pretokenize_sv(text, sv_chunks);
        for (const auto& piece : sv_chunks)
        {
            encode_piece(piece);
        }
    }
    else
    {
        std::vector<std::string> specials;
        specials.reserve(special_tokens_.size());
        for (const auto& [s, _] : special_tokens_) specials.push_back(s);
        auto pretokens = pretokenize_with_specials(text, specials);
        for (const auto& pt : pretokens)
        {
            if (pt.is_special)
            {
                auto it = special_tokens_.find(pt.text);
                if (it != special_tokens_.end()) out.push_back(it->second);
            }
            else
            {
                encode_piece(pt.text);
            }
        }
    }

    return out;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const
{
    const auto& vocab_ = impl_->vocab_;
    const auto& special_tokens_ = impl_->special_tokens_;

    std::string out;
    for (uint32_t id : ids)
    {
        if (auto it = vocab_.find(id); it != vocab_.end())
        {
            out.append(it->second.begin(), it->second.end());
            continue;
        }
        bool found_special = false;
        for (const auto& [s, sid] : special_tokens_)
        {
            if (sid == id)
            {
                out += s;
                found_special = true;
                break;
            }
        }
        if (!found_special)
        {
            throw std::out_of_range("Unknown token id: " + std::to_string(id));
        }
    }
    return out;
}

void Tokenizer::save(const std::filesystem::path& dir) const
{
    const auto& vocab_ = impl_->vocab_;
    const auto& merges_ = impl_->merges_;
    const auto& special_tokens_ = impl_->special_tokens_;
    const EncodeMode encode_mode = impl_->encode_mode_;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw save_error(dir, "failed to create directory: " + ec.message());
    }
    if (!std::filesystem::is_directory(dir, ec)) {
        if (ec) {
            throw save_error(dir, "failed to inspect directory: " + ec.message());
        }
        throw save_error(dir, "path is not a directory");
    }

    nlohmann::json j = nlohmann::json::object();
    if (encode_mode == EncodeMode::ByteSpanRanks)
    {
        for (const auto& [id, bytes] : vocab_) {
            j[bytes_to_hex(bytes)] = id;
        }
    }
    else
    {
        // Native format: binary-safe byte arrays keyed by token ID.
        for (const auto& [id, bytes] : vocab_)
        {
            nlohmann::json arr = nlohmann::json::array();
            for (auto b : bytes) arr.push_back(static_cast<int>(b));
            j[std::to_string(id)] = arr;
        }
    }
    write_checked_file(dir / "vocab.json", [&](std::ostream& out) {
        out << j.dump();
    });

    write_checked_file(dir / "merges.txt", [&](std::ostream& out) {
        for (const auto& m : merges_)
        {
            if (encode_mode == EncodeMode::ByteSpanRanks) {
                auto left = vocab_.find(m.left);
                auto right = vocab_.find(m.right);
                if (left == vocab_.end() || right == vocab_.end()) {
                    throw save_error(dir / "merges.txt", "merge rule references unknown token id");
                }
                out << bytes_to_hex(left->second) << ' ' << bytes_to_hex(right->second) << '\n';
            } else {
                out << m.left << ' ' << m.right << ' ' << m.merged << '\n';
            }
        }
    });

    write_checked_file(dir / "special_tokens.txt", [&](std::ostream& out) {
        for (const auto& [s, id] : special_tokens_)
        {
            out << id << ' ' << s << '\n';
        }
    });
}

// Helper: decode a hex string like "2074" to bytes {0x20, 0x74}.
static std::vector<uint8_t> parse_hex_bytes(
    std::string_view hex,
    const std::filesystem::path& path,
    const std::string& field)
{
    std::vector<uint8_t> out;
    if (hex.empty()) {
        throw load_error(path, field + " must not be empty");
    }
    if ((hex.size() % 2) != 0) {
        throw load_error(path, field + " has odd length");
    }
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2)
    {
        auto hi = hex[i], lo = hex[i + 1];
        auto nibble = [&](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
            throw load_error(path, field + " contains non-hex character");
        };
        out.push_back(static_cast<uint8_t>((nibble(hi) << 4) | nibble(lo)));
    }
    return out;
}

enum class VocabFormat {
    Gpt2Hex,
    Native
};

static VocabFormat classify_vocab_format(
    const nlohmann::json& j,
    const std::filesystem::path& path)
{
    if (!j.is_object()) {
        throw load_error(path, "vocab.json must be a JSON object");
    }
    if (j.empty()) {
        throw load_error(path, "vocab.json must not be empty");
    }

    bool saw_gpt2 = false;
    bool saw_native = false;
    for (auto it = j.begin(); it != j.end(); ++it) {
        const auto& value = it.value();
        if (value.is_number_integer() || value.is_number_unsigned()) {
            saw_gpt2 = true;
        } else if (value.is_array() || value.is_string()) {
            saw_native = true;
        } else {
            throw load_error(path, "vocab entry has unsupported value type");
        }
    }
    if (saw_gpt2 == saw_native) {
        throw load_error(path, "vocab.json mixes GPT-2 and native formats");
    }
    return saw_gpt2 ? VocabFormat::Gpt2Hex : VocabFormat::Native;
}

Tokenizer Tokenizer::load(const std::filesystem::path& dir)
{
    Tokenizer tok;
    auto& vocab_ = tok.impl_->vocab_;
    auto& token_to_id_ = tok.impl_->token_to_id_;
    auto& merges_ = tok.impl_->merges_;
    auto& special_tokens_ = tok.impl_->special_tokens_;

    // ── Load vocab.json (two formats) ──
    {
        const auto vocab_path = dir / "vocab.json";
        std::ifstream f(vocab_path);
        if (!f)
        {
            throw load_error(vocab_path, "required file is missing or unreadable");
        }

        nlohmann::json j;
        try {
            f >> j;
        } catch (const nlohmann::json::exception& e) {
            throw load_error(vocab_path, std::string("invalid JSON: ") + e.what());
        }

        std::unordered_set<uint32_t> seen_ids;
        std::unordered_set<std::string> seen_bytes;
        std::array<bool, 256> single_bytes_seen{};
        single_bytes_seen.fill(false);
        const VocabFormat vocab_format = classify_vocab_format(j, vocab_path);
        tok.impl_->encode_mode_ = vocab_format == VocabFormat::Gpt2Hex
            ? EncodeMode::ByteSpanRanks
            : EncodeMode::PairRules;

        const auto remember_vocab_entry = [&](uint32_t id, std::vector<uint8_t> bytes) {
            if (bytes.empty()) {
                throw load_error(vocab_path, "vocab byte sequence must not be empty");
            }
            if (bytes.size() == 1) {
                if (id >= 256) {
                    throw load_error(vocab_path, "single-byte token id must be in the base token range");
                }
                if (vocab_format == VocabFormat::Native && id != bytes[0]) {
                    throw load_error(vocab_path, "native single-byte token id must equal its byte value");
                }
            } else if (id < 256) {
                throw load_error(vocab_path, "base token ids 0..255 must be single-byte tokens");
            }
            if (!seen_ids.insert(id).second) {
                throw load_error(vocab_path, "duplicate token id " + std::to_string(id));
            }
            std::string key(bytes.begin(), bytes.end());
            if (!seen_bytes.insert(key).second) {
                throw load_error(vocab_path, "duplicate token byte sequence");
            }
            if (bytes.size() == 1) {
                single_bytes_seen[bytes[0]] = true;
            }
            vocab_[id] = std::move(bytes);
        };

        if (vocab_format == VocabFormat::Gpt2Hex)
        {
            // GPT-2 format: {hex_encoded_bytes: token_id}
            for (auto it = j.begin(); it != j.end(); ++it)
            {
                auto bytes = parse_hex_bytes(it.key(), vocab_path, "vocab key");
                uint32_t id = json_u32(it.value(), vocab_path, "token id");
                remember_vocab_entry(id, std::move(bytes));
            }
        }
        else
        {
            // Native format: {token_id_string: [byte_array]}
            for (auto it = j.begin(); it != j.end(); ++it)
            {
                uint32_t id = 0;
                if (!parse_u32(it.key(), id)) {
                    throw load_error(vocab_path, "native vocab key is not a uint32 token id");
                }
                std::vector<uint8_t> bytes;
                if (it.value().is_array())
                {
                    for (const auto& b : it.value()) {
                        const uint32_t byte = json_u32(b, vocab_path, "byte value");
                        if (byte > 255) {
                            throw load_error(vocab_path, "byte value exceeds 255");
                        }
                        bytes.push_back(static_cast<uint8_t>(byte));
                    }
                }
                else if (it.value().is_string())
                {
                    std::string s = it.value().get<std::string>();
                    bytes.assign(s.begin(), s.end());
                }
                remember_vocab_entry(id, std::move(bytes));
            }
        }

        for (size_t i = 0; i < single_bytes_seen.size(); ++i) {
            if (!single_bytes_seen[i]) {
                throw load_error(vocab_path, "missing single-byte token " + std::to_string(i));
            }
        }

        std::vector<bool> contiguous_ids(vocab_.size(), false);
        for (const auto& [id, _] : vocab_) {
            if (id >= vocab_.size()) {
                throw load_error(vocab_path, "vocab token ids must be contiguous from 0");
            }
            contiguous_ids[id] = true;
        }
        for (size_t id = 0; id < contiguous_ids.size(); ++id) {
            if (!contiguous_ids[id]) {
                throw load_error(vocab_path, "missing vocab token id " + std::to_string(id));
            }
        }
    }

    // Build a reverse lookup: byte-sequence-string → token_id.
    // Needed to resolve merges.txt in GPT-2 hex format.
    for (const auto& [id, bytes] : vocab_)
        token_to_id_[std::string(bytes.begin(), bytes.end())] = id;

    // ── Load merges.txt (two formats) ──
    {
        const auto merges_path = dir / "merges.txt";
        std::ifstream f(merges_path);
        if (!f)
        {
            throw load_error(merges_path, "required file is missing or unreadable");
        }

        enum class MergeFormat { Unknown, Native, Gpt2Hex };
        MergeFormat format = MergeFormat::Unknown;
        uint32_t expected_native_merged_id = 256;
        uint32_t expected_gpt2_merged_id = 256;
        std::string line;
        size_t line_number = 0;
        while (std::getline(f, line))
        {
            ++line_number;
            line = trim_line(std::move(line));
            auto parts = split_ws(line);
            if (parts.empty() || (!parts[0].empty() && parts[0][0] == '#')) {
                continue;
            }

            auto merge_context = [&] {
                return "merges.txt line " + std::to_string(line_number);
            };

            if (parts.size() == 3)
            {
                if (format == MergeFormat::Gpt2Hex) {
                    throw load_error(merges_path, merge_context() + " mixes merge formats");
                }
                format = MergeFormat::Native;

                uint32_t l = 0, r = 0, m = 0;
                if (!parse_u32(parts[0], l) || !parse_u32(parts[1], r) || !parse_u32(parts[2], m)) {
                    throw load_error(merges_path, merge_context() + " has invalid native token id");
                }
                auto it_l = vocab_.find(l);
                auto it_r = vocab_.find(r);
                auto it_m = vocab_.find(m);
                if (it_l == vocab_.end() || it_r == vocab_.end() || it_m == vocab_.end()) {
                    throw load_error(merges_path, merge_context() + " references unknown token id");
                }
                if (m != expected_native_merged_id) {
                    throw load_error(merges_path, merge_context() + " native merged token id is not in merge-priority order");
                }

                std::vector<uint8_t> expected = it_l->second;
                expected.insert(expected.end(), it_r->second.begin(), it_r->second.end());
                if (expected != it_m->second) {
                    throw load_error(merges_path, merge_context() + " merged token bytes do not match left+right");
                }
                merges_.push_back(MergeRule{l, r, m});
                ++expected_native_merged_id;
                continue;
            }

            if (parts.size() == 2)
            {
                if (format == MergeFormat::Native) {
                    throw load_error(merges_path, merge_context() + " mixes merge formats");
                }
                format = MergeFormat::Gpt2Hex;

                auto bytes_l = parse_hex_bytes(parts[0], merges_path, merge_context() + " left token");
                auto bytes_r = parse_hex_bytes(parts[1], merges_path, merge_context() + " right token");
                std::string key_l(bytes_l.begin(), bytes_l.end());
                std::string key_r(bytes_r.begin(), bytes_r.end());

                auto it_l = token_to_id_.find(key_l);
                auto it_r = token_to_id_.find(key_r);
                if (it_l == token_to_id_.end() || it_r == token_to_id_.end())
                {
                    throw load_error(merges_path, merge_context() + " references unknown token bytes");
                }

                std::vector<uint8_t> merged_bytes = std::move(bytes_l);
                merged_bytes.insert(merged_bytes.end(), bytes_r.begin(), bytes_r.end());
                std::string merged_key(merged_bytes.begin(), merged_bytes.end());
                auto it_m = token_to_id_.find(merged_key);
                if (it_m == token_to_id_.end()) {
                    throw load_error(merges_path, merge_context() + " merged token bytes are missing from vocab");
                }
                if (it_m->second != expected_gpt2_merged_id) {
                    throw load_error(merges_path, merge_context() + " GPT-2 merged token id is not in merge-priority order");
                }
                merges_.push_back(MergeRule{it_l->second, it_r->second, it_m->second});
                ++expected_gpt2_merged_id;
                continue;
            }

            throw load_error(merges_path, merge_context() + " is not a valid merge rule");
        }

        std::vector<bool> merged_ids(vocab_.size(), false);
        for (const auto& merge : merges_) {
            if (merge.merged >= vocab_.size()) {
                throw load_error(merges_path, "merge rule produced token id outside vocab");
            }
            if (merge.merged < 256) {
                throw load_error(merges_path, "merge rule produced base token id");
            }
            if (merged_ids[merge.merged]) {
                throw load_error(merges_path, "duplicate merged token id " + std::to_string(merge.merged));
            }
            merged_ids[merge.merged] = true;
        }
        for (uint32_t id = 256; id < vocab_.size(); ++id) {
            if (!merged_ids[id]) {
                throw load_error(merges_path, "vocab token id " + std::to_string(id) + " has no merge rule");
            }
        }
    }

    // ── Load special_tokens.txt ──
    {
        const auto specials_path = dir / "special_tokens.txt";
        if (std::filesystem::exists(specials_path)) {
            std::ifstream f(specials_path);
            if (!f) {
                throw load_error(specials_path, "file exists but is unreadable");
            }

            std::unordered_set<uint32_t> seen_special_ids;
            std::string line;
            size_t line_number = 0;
            while (std::getline(f, line))
            {
                ++line_number;
                line = trim_line(std::move(line));
                if (line.empty()) {
                    continue;
                }

                const size_t first_ws = line.find_first_of(" \t");
                if (first_ws == std::string::npos) {
                    throw load_error(specials_path, "line " + std::to_string(line_number) + " is missing token text");
                }

                uint32_t id = 0;
                if (!parse_u32(std::string_view(line).substr(0, first_ws), id)) {
                    throw load_error(specials_path, "line " + std::to_string(line_number) + " has invalid token id");
                }

                size_t text_pos = first_ws;
                while (text_pos < line.size() && (line[text_pos] == ' ' || line[text_pos] == '\t')) {
                    ++text_pos;
                }
                if (text_pos >= line.size()) {
                    throw load_error(specials_path, "line " + std::to_string(line_number) + " has empty token text");
                }

                std::string s = line.substr(text_pos);
                if (vocab_.find(id) != vocab_.end()) {
                    throw load_error(specials_path, "special token id collides with vocab id " + std::to_string(id));
                }
                try {
                    validate_special_token_text(s);
                } catch (const std::runtime_error& e) {
                    throw load_error(specials_path, "line " + std::to_string(line_number) + " has invalid token text: " + e.what());
                }
                if (!seen_special_ids.insert(id).second) {
                    throw load_error(specials_path, "duplicate special token id " + std::to_string(id));
                }
                if (!special_tokens_.emplace(s, id).second) {
                    throw load_error(specials_path, "duplicate special token string");
                }
                if (token_to_id_.find(s) != token_to_id_.end()) {
                    throw load_error(specials_path, "special token collides with vocab byte sequence");
                }
            }

            const uint32_t vocab_count = static_cast<uint32_t>(vocab_.size());
            std::vector<bool> special_id_slots(special_tokens_.size(), false);
            for (const auto& [_, id] : special_tokens_) {
                if (id < vocab_count || id >= vocab_count + special_tokens_.size()) {
                    throw load_error(specials_path, "special token ids must be contiguous after vocab");
                }
                special_id_slots[id - vocab_count] = true;
            }
            for (size_t i = 0; i < special_id_slots.size(); ++i) {
                if (!special_id_slots[i]) {
                    throw load_error(specials_path, "missing special token id " + std::to_string(vocab_count + i));
                }
            }
        }
    }

    // Rebuild token_to_id_ with special tokens included.
    token_to_id_.clear();
    for (const auto& [id, bytes] : vocab_)
        token_to_id_[std::string(bytes.begin(), bytes.end())] = id;
    for (const auto& [s, id] : special_tokens_)
        token_to_id_[s] = id;
    tok.impl_->rebuild_encode_caches();

    return tok;
}

uint32_t Tokenizer::vocab_size() const {
    return static_cast<uint32_t>(impl_->vocab_.size() + impl_->special_tokens_.size());
}

const std::vector<MergeRule>& Tokenizer::merges() const {
    return impl_->merges_;
}

const std::unordered_map<uint32_t, std::vector<uint8_t>>& Tokenizer::vocab() const {
    return impl_->vocab_;
}

const std::unordered_map<std::string, uint32_t>& Tokenizer::special_tokens() const {
    return impl_->special_tokens_;
}

} // namespace bpe
