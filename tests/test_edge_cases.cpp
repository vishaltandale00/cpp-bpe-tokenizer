#include <gtest/gtest.h>
#include "bpe/tokenizer.h"
#include "bpe/trainer.h"
#include "test_helpers.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using namespace bpe;

namespace {

std::filesystem::path save_error_dir(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / ("bpe_save_error_" + name);
    std::filesystem::remove_all(dir);
    return dir;
}

Tokenizer trained_save_tokenizer() {
    Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 280, {"<|endoftext|>"});
    return tok;
}

std::string hex_byte(uint32_t byte) {
    static constexpr char kHex[] = "0123456789abcdef";
    return {
        kHex[(byte >> 4) & 0x0f],
        kHex[byte & 0x0f],
    };
}

void write_text(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << content;
}

std::filesystem::path write_gpt2_hex_save_model() {
    auto dir = save_error_dir("gpt2_hex_source");
    std::filesystem::create_directories(dir);

    nlohmann::json vocab = nlohmann::json::object();
    for (uint32_t i = 0; i < 256; ++i) {
        const uint32_t token_id = i == 0 ? 1 : i == 1 ? 0 : i;
        vocab[hex_byte(i)] = token_id;
    }
    vocab["6263"] = 256;    // "bc"
    vocab["6162"] = 257;    // "ab"
    vocab["616263"] = 258;  // "abc"

    write_text(dir / "vocab.json", vocab.dump());
    write_text(dir / "merges.txt",
               "62 63\n"
               "61 62\n"
               "6162 63\n");
    return dir;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════
// Edge cases and adversarial inputs
// ═══════════════════════════════════════════════════════════════════

TEST(EdgeCases, TrainOnEmptyString) {
    Tokenizer tok;
    tok.train("", 256); // no merges possible
    EXPECT_EQ(tok.vocab_size(), 256u);
    EXPECT_TRUE(tok.merges().empty());
}

TEST(EdgeCases, TrainOnSingleByte) {
    Tokenizer tok;
    tok.train("a", 256);
    EXPECT_TRUE(tok.merges().empty());
}

TEST(EdgeCases, TrainOnRepeatedSingleByte) {
    Tokenizer tok;
    // "aaaa" — the only pair is (a,a), which can merge.
    tok.train("aaaa", 257);
    EXPECT_EQ(tok.merges().size(), 1u);
}

TEST(EdgeCases, VocabSizeEqualsBytes) {
    // vocab_size == 256 means no merges needed.
    Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 256);
    EXPECT_TRUE(tok.merges().empty());
}

TEST(EdgeCases, VocabSizeLargerThanPossibleMerges) {
    // Very small corpus — might run out of pairs before reaching target.
    Tokenizer tok;
    tok.train("ab", 300);
    // Should not crash; just trains as many merges as possible.
    EXPECT_LE(tok.vocab_size(), 300u);
}

TEST(EdgeCases, EncodeSingleByteNotInTrainingData) {
    Tokenizer tok;
    tok.train("abc", 256);
    // 'z' never appeared in training, but should still encode as byte.
    auto ids = tok.encode("z");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], static_cast<uint32_t>('z'));
}

TEST(EdgeCases, DecodeInvalidId) {
    Tokenizer tok;
    tok.train("abc", 256);
    // ID 99999 doesn't exist — should throw or handle gracefully.
    EXPECT_ANY_THROW(tok.decode({99999}));
}

TEST(EdgeCases, NullBytesInText) {
    Tokenizer tok;
    tok.train("abc", 256);
    std::string text_with_null = std::string("abc") + '\0' + "def";
    auto ids = tok.encode(text_with_null);
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, text_with_null);
}

TEST(EdgeCases, VeryLongSingleChunk) {
    Tokenizer tok;
    // A single "word" of 10k characters.
    std::string long_word(10000, 'a');
    tok.train(long_word, 257);
    auto ids = tok.encode(long_word);
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, long_word);
}

TEST(EdgeCases, ManyUniqueBytes) {
    // Text with all 256 byte values.
    std::string all_bytes;
    for (int i = 0; i < 256; ++i) {
        all_bytes += static_cast<char>(i);
    }
    Tokenizer tok;
    tok.train(all_bytes, 260);
    auto ids = tok.encode(all_bytes);
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, all_bytes);
}

// ═══════════════════════════════════════════════════════════════════
// Save / Load roundtrip
// ═══════════════════════════════════════════════════════════════════

TEST(SaveLoad, RoundtripPersistence) {
    auto dir = test_helpers::temp_dir();

    // Train and save.
    Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 280, {"<|endoftext|>"});
    tok.save(dir);

    // Load into a new tokenizer.
    auto tok2 = Tokenizer::load(dir);

    // Verify they produce identical results.
    std::string text = "the cat sat on the mat<|endoftext|>hello";
    EXPECT_EQ(tok.encode(text), tok2.encode(text));
    EXPECT_EQ(tok.vocab_size(), tok2.vocab_size());
    EXPECT_EQ(tok.merges().size(), tok2.merges().size());

    test_helpers::cleanup_temp_dir();
}

TEST(SaveLoad, LoadedTokenizerRoundtrips) {
    auto dir = test_helpers::temp_dir();

    Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 280);
    tok.save(dir);

    auto tok2 = Tokenizer::load(dir);

    std::string text = "unseen text with 123 numbers and !@# symbols";
    auto ids = tok2.encode(text);
    auto decoded = tok2.decode(ids);
    EXPECT_EQ(decoded, text);

    test_helpers::cleanup_temp_dir();
}

TEST(SaveLoad, Gpt2HexLoadedTokenizerSavesReloadableHexModel) {
    auto source = write_gpt2_hex_save_model();
    auto saved = save_error_dir("gpt2_hex_saved");

    auto tok = Tokenizer::load(source);
    const std::vector<uint32_t> expected = {258};
    std::string swapped_bytes;
    swapped_bytes.push_back('\0');
    swapped_bytes.push_back('\1');
    const std::vector<uint32_t> swapped_byte_ids = {1, 0};

    ASSERT_EQ(tok.encode("abc"), expected);
    ASSERT_EQ(tok.encode(swapped_bytes), swapped_byte_ids);
    ASSERT_EQ(tok.decode(swapped_byte_ids), swapped_bytes);

    tok.save(saved);
    auto reloaded = Tokenizer::load(saved);
    EXPECT_EQ(reloaded.encode("abc"), expected);
    EXPECT_EQ(reloaded.decode(expected), "abc");
    EXPECT_EQ(reloaded.encode(swapped_bytes), swapped_byte_ids);
    EXPECT_EQ(reloaded.decode(swapped_byte_ids), swapped_bytes);

    std::filesystem::remove_all(source);
    std::filesystem::remove_all(saved);
}

TEST(SaveLoad, SaveThrowsWhenDirectoryCannotBeCreated) {
    auto dir = save_error_dir("create_dir");
    {
        std::ofstream blocker(dir, std::ios::binary);
        blocker << "not a directory";
    }

    auto tok = trained_save_tokenizer();
    EXPECT_THROW(tok.save(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SaveLoad, SaveThrowsWhenVocabFileCannotBeOpened) {
    auto dir = save_error_dir("vocab_open");
    std::filesystem::create_directories(dir / "vocab.json");

    auto tok = trained_save_tokenizer();
    EXPECT_THROW(tok.save(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SaveLoad, SaveThrowsWhenMergesFileCannotBeOpened) {
    auto dir = save_error_dir("merges_open");
    std::filesystem::create_directories(dir / "merges.txt");

    auto tok = trained_save_tokenizer();
    EXPECT_THROW(tok.save(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(SaveLoad, SaveThrowsWhenSpecialTokensFileCannotBeOpened) {
    auto dir = save_error_dir("specials_open");
    std::filesystem::create_directories(dir / "special_tokens.txt");

    auto tok = trained_save_tokenizer();
    EXPECT_THROW(tok.save(dir), std::runtime_error);

    std::filesystem::remove_all(dir);
}
