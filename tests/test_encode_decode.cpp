#include <gtest/gtest.h>
#include "bpe/codec.h"
#include "bpe/tokenizer.h"
#include "test_helpers.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

using namespace bpe;

namespace {

std::filesystem::path unique_encode_temp_dir(const std::string& name) {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    auto dir = std::filesystem::temp_directory_path() /
        ("bpe_encode_decode_" + name + "_" + std::to_string(suffix));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void write_text(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << content;
}

std::filesystem::path write_native_merge_identity_model() {
    auto dir = unique_encode_temp_dir("merge_identity");

    nlohmann::json vocab = nlohmann::json::object();
    for (uint32_t i = 0; i < 256; ++i) {
        vocab[std::to_string(i)] = {i};
    }
    vocab["256"] = {98, 99};      // "bc"
    vocab["257"] = {97, 98};      // "ab"
    vocab["258"] = {97, 98, 99};  // "abc"

    write_text(dir / "vocab.json", vocab.dump());
    write_text(dir / "merges.txt",
               "98 99 256\n"
               "97 98 257\n"
               "257 99 258\n");
    return dir;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════
// Low-level codec tests
// ═══════════════════════════════════════════════════════════════════

TEST(BytesToIds, EmptyString) {
    auto ids = bytes_to_ids("");
    EXPECT_TRUE(ids.empty());
}

TEST(BytesToIds, ASCIIString) {
    auto ids = bytes_to_ids("abc");
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 97u);  // 'a'
    EXPECT_EQ(ids[1], 98u);  // 'b'
    EXPECT_EQ(ids[2], 99u);  // 'c'
}

TEST(BytesToIds, HighBytes) {
    // UTF-8 multi-byte: é = 0xC3 0xA9
    auto ids = bytes_to_ids("é");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 0xC3u);
    EXPECT_EQ(ids[1], 0xA9u);
}

TEST(BytesToIds, NullByte) {
    std::string s(1, '\0');
    auto ids = bytes_to_ids(s);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 0u);
}

TEST(ApplyMergesCodec, NoMerges) {
    std::vector<uint32_t> ids = {97, 98, 99};
    std::vector<MergeRule> merges;
    auto result = apply_merges(ids, merges);
    EXPECT_EQ(result, ids);
}

TEST(ApplyMergesCodec, SingleMerge) {
    std::vector<uint32_t> ids = {97, 98, 99};
    std::vector<MergeRule> merges = {{97, 98, 256}};
    auto result = apply_merges(ids, merges);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 256u);
    EXPECT_EQ(result[1], 99u);
}

TEST(ApplyMergesCodec, ChainedMerges) {
    // "abcd" → merge (a,b)→256, then (256,c)→257
    std::vector<uint32_t> ids = {97, 98, 99, 100};
    std::vector<MergeRule> merges = {
        {97, 98, 256},
        {256, 99, 257},
    };
    auto result = apply_merges(ids, merges);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 257u);
    EXPECT_EQ(result[1], 100u);
}

TEST(ApplyMergesCodec, MergeOrderMatters) {
    // "abc" with merges: (b,c)→256 first, (a,256)→257 second
    std::vector<uint32_t> ids = {97, 98, 99};
    std::vector<MergeRule> merges = {
        {98, 99, 256},
        {97, 256, 257},
    };
    auto result = apply_merges(ids, merges);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 257u);
}

TEST(IdsToBytes, EmptyIds) {
    std::unordered_map<uint32_t, std::vector<uint8_t>> vocab;
    auto result = ids_to_bytes({}, vocab);
    EXPECT_TRUE(result.empty());
}

TEST(IdsToBytes, ByteTokens) {
    std::unordered_map<uint32_t, std::vector<uint8_t>> vocab = {
        {97, {97}}, {98, {98}}, {99, {99}}
    };
    auto result = ids_to_bytes({97, 98, 99}, vocab);
    EXPECT_EQ(result, "abc");
}

TEST(IdsToBytes, MergedToken) {
    std::unordered_map<uint32_t, std::vector<uint8_t>> vocab = {
        {97, {97}}, {98, {98}}, {99, {99}}, {256, {97, 98}}
    };
    auto result = ids_to_bytes({256, 99}, vocab);
    EXPECT_EQ(result, "abc");
}

TEST(IdsToBytes, UnknownIdThrows) {
    std::unordered_map<uint32_t, std::vector<uint8_t>> vocab = {
        {97, {97}}, {98, {98}}
    };
    EXPECT_THROW(ids_to_bytes({97, 99}, vocab), std::out_of_range);
}

// ═══════════════════════════════════════════════════════════════════
// Tokenizer encode/decode
// ═══════════════════════════════════════════════════════════════════

class TokenizerEncodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        tok = test_helpers::train_tiny(280);
    }
    bpe::Tokenizer tok;
};

TEST_F(TokenizerEncodeTest, EncodesNonEmpty) {
    auto ids = tok.encode("the cat");
    EXPECT_FALSE(ids.empty());
}

TEST_F(TokenizerEncodeTest, AllIdsInVocab) {
    auto ids = tok.encode("the cat sat on the mat");
    for (auto id : ids) {
        EXPECT_LT(id, tok.vocab_size())
            << "Token ID " << id << " exceeds vocab size " << tok.vocab_size();
    }
}

TEST_F(TokenizerEncodeTest, DecodesBackToOriginal) {
    std::string text = "the cat sat on the mat";
    auto ids = tok.encode(text);
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerEncodeTest, ShorterThanBytes) {
    // After merges, encoding should produce fewer tokens than bytes.
    std::string text = "the the the the";
    auto ids = tok.encode(text);
    EXPECT_LT(ids.size(), text.size());
}

TEST_F(TokenizerEncodeTest, EmptyString) {
    auto ids = tok.encode("");
    EXPECT_TRUE(ids.empty());
}

TEST_F(TokenizerEncodeTest, SingleChar) {
    auto ids = tok.encode("a");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], static_cast<uint32_t>('a'));
}

TEST_F(TokenizerEncodeTest, UnseenText) {
    // Text not in training corpus — should still encode (byte fallback).
    auto ids = tok.encode("xyz123!@#");
    EXPECT_FALSE(ids.empty());
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, "xyz123!@#");
}

TEST(TokenizerEncodeRegression, RespectsMergePairIdentityOverVocabSpan) {
    auto dir = write_native_merge_identity_model();
    auto saved = unique_encode_temp_dir("merge_identity_saved");
    auto tok = bpe::Tokenizer::load(dir);

    // The first learned merge is b+c -> 256, leaving a+256. Even though the
    // vocab contains "abc", there is no a+256 merge rule that can produce it.
    const std::vector<uint32_t> expected = {97, 256};
    EXPECT_EQ(tok.encode("abc"), expected);

    tok.save(saved);
    auto reloaded = bpe::Tokenizer::load(saved);
    EXPECT_EQ(reloaded.encode("abc"), expected);

    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(saved);
}
