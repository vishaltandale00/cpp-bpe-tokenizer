#include <gtest/gtest.h>
#include "bpe/tokenizer.h"
#include "test_helpers.h"

#include <stdexcept>
#include <vector>

using namespace bpe;

class SpecialTokenTest : public ::testing::Test {
protected:
    void SetUp() override {
        tok = test_helpers::train_with_specials(282);
    }
    Tokenizer tok;
};

TEST_F(SpecialTokenTest, SpecialTokensHaveUniqueIds) {
    auto& specials = tok.special_tokens();
    EXPECT_NE(specials.at("<|endoftext|>"), specials.at("<|pad|>"));
}

TEST_F(SpecialTokenTest, SpecialTokenIdsNotInByteRange) {
    auto& specials = tok.special_tokens();
    for (auto& [name, id] : specials) {
        EXPECT_GE(id, 256u) << name << " has ID in byte range";
    }
}

TEST_F(SpecialTokenTest, EncodeSpecialToken) {
    auto ids = tok.encode("<|endoftext|>");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], tok.special_tokens().at("<|endoftext|>"));
}

TEST_F(SpecialTokenTest, EncodeWithSpecialInMiddle) {
    auto ids = tok.encode("hello<|endoftext|>world");
    // Should contain the special token ID somewhere in the middle.
    auto eot_id = tok.special_tokens().at("<|endoftext|>");
    bool found = false;
    for (auto id : ids) {
        if (id == eot_id) found = true;
    }
    EXPECT_TRUE(found) << "Special token not found in encoding";
}

TEST_F(SpecialTokenTest, DecodeSpecialToken) {
    auto eot_id = tok.special_tokens().at("<|endoftext|>");
    auto decoded = tok.decode({eot_id});
    EXPECT_EQ(decoded, "<|endoftext|>");
}

TEST_F(SpecialTokenTest, RoundtripWithSpecials) {
    std::string text = "hello<|endoftext|>world<|pad|>foo";
    auto ids = tok.encode(text);
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(SpecialTokenTest, MultipleAdjacentSpecials) {
    std::string text = "<|endoftext|><|pad|><|endoftext|>";
    auto ids = tok.encode(text);
    auto decoded = tok.decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(SpecialTokenTest, SpecialTokenNotMergedInto) {
    // The literal text "<|endoftext|>" should not be tokenized as
    // individual bytes — it should always map to its special ID.
    auto ids = tok.encode("<|endoftext|>");
    EXPECT_EQ(ids.size(), 1u);
}

TEST_F(SpecialTokenTest, NoMergesAcrossSpecials) {
    // "ab<|endoftext|>ab" — the "ab" on each side should be tokenized
    // independently (no merge should span across the special token).
    auto ids_left = tok.encode("ab");
    auto ids_right = tok.encode("ab");
    EXPECT_EQ(ids_left, ids_right)
        << "Same text should encode the same way regardless of context";

    auto ids_full = tok.encode("ab<|endoftext|>ab");
    auto eot_id = tok.special_tokens().at("<|endoftext|>");

    // Find the special token position.
    auto it = std::find(ids_full.begin(), ids_full.end(), eot_id);
    ASSERT_NE(it, ids_full.end());

    // Tokens before and after should match independent encoding.
    std::vector<uint32_t> before(ids_full.begin(), it);
    std::vector<uint32_t> after(it + 1, ids_full.end());
    EXPECT_EQ(before, ids_left);
    EXPECT_EQ(after, ids_right);
}

TEST(SpecialTokenValidation, RejectsEmptySpecialToken) {
    Tokenizer tok;
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {""}),
        std::runtime_error);
}

TEST(SpecialTokenValidation, RejectsDuplicateSpecialTokens) {
    Tokenizer tok;
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {"<|dup|>", "<|dup|>"}),
        std::runtime_error);
}

TEST(SpecialTokenValidation, RejectsNormalVocabByteStringCollision) {
    Tokenizer tok;
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {"a"}),
        std::runtime_error);
}

TEST(SpecialTokenValidation, FailedTrainLeavesExistingTokenizerUnchanged) {
    Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 282, {"<|endoftext|>", "<|pad|>"});

    const std::string text = "the cat<|endoftext|>sat<|pad|>";
    const auto before_ids = tok.encode(text);
    const auto before_vocab = tok.vocab();
    const auto before_merges = tok.merges();
    const auto before_specials = tok.special_tokens();
    const auto before_vocab_size = tok.vocab_size();

    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {"a"}),
        std::runtime_error);

    EXPECT_EQ(tok.encode(text), before_ids);
    EXPECT_EQ(tok.vocab(), before_vocab);
    ASSERT_EQ(tok.merges().size(), before_merges.size());
    for (size_t i = 0; i < before_merges.size(); ++i) {
        EXPECT_EQ(tok.merges()[i].left, before_merges[i].left);
        EXPECT_EQ(tok.merges()[i].right, before_merges[i].right);
        EXPECT_EQ(tok.merges()[i].merged, before_merges[i].merged);
    }
    EXPECT_EQ(tok.special_tokens(), before_specials);
    EXPECT_EQ(tok.vocab_size(), before_vocab_size);
}

TEST(SpecialTokenValidation, RejectsSpecialTokensThatCannotRoundtripThroughSave) {
    Tokenizer tok;
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {" <|leading_space|>"}),
        std::runtime_error);
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {"\t<|leading_tab|>"}),
        std::runtime_error);
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {"<|line\nbreak|>"}),
        std::runtime_error);
    EXPECT_THROW(
        tok.train(test_helpers::tiny_corpus(), 280, {"<|carriage\rreturn|>"}),
        std::runtime_error);
}

TEST(SpecialTokenValidation, SpecialTokenIdsAreContiguousAfterVocab) {
    Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 282, {"<|endoftext|>", "<|pad|>"});

    const auto vocab_count = tok.vocab().size();
    const auto& specials = tok.special_tokens();
    std::vector<bool> slots(specials.size(), false);
    for (const auto& [name, id] : specials) {
        ASSERT_GE(id, vocab_count) << name;
        ASSERT_LT(id, vocab_count + specials.size()) << name;
        slots[static_cast<size_t>(id) - vocab_count] = true;
    }
    for (bool filled : slots) {
        EXPECT_TRUE(filled);
    }
}
