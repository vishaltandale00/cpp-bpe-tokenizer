#include <gtest/gtest.h>
#include "bpe/pretokenizer.h"
#include <numeric>

using namespace bpe;

// ═══════════════════════════════════════════════════════════════════
// Pre-tokenizer unit tests
// ═══════════════════════════════════════════════════════════════════

TEST(Pretokenizer, EmptyString) {
    auto result = pretokenize("");
    EXPECT_TRUE(result.empty());
}

TEST(Pretokenizer, SingleWord) {
    auto result = pretokenize("hello");
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "hello");
}

TEST(Pretokenizer, TwoWords) {
    auto result = pretokenize("hello world");
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], "hello");
    EXPECT_EQ(result[1], " world");
}

TEST(Pretokenizer, LeadingSpace) {
    auto result = pretokenize(" hello");
    // The leading space should attach to "hello"
    ASSERT_GE(result.size(), 1u);
    // Verify the text reconstructs
    std::string joined;
    for (auto& s : result) joined += s;
    EXPECT_EQ(joined, " hello");
}

TEST(Pretokenizer, Contractions) {
    auto result = pretokenize("I'm don't we're they've I'll he'd she's");
    // Each contraction suffix should be its own token
    // Check that 's, 't, 're, 've, 'll, 'd are separate chunks
    bool found_s = false, found_t = false, found_re = false;
    bool found_ve = false, found_ll = false, found_d = false;
    for (auto& s : result) {
        if (s == "'s") found_s = true;
        if (s == "'t") found_t = true;
        if (s == "'re") found_re = true;
        if (s == "'ve") found_ve = true;
        if (s == "'ll") found_ll = true;
        if (s == "'d") found_d = true;
    }
    EXPECT_TRUE(found_s) << "Missing 's contraction";
    EXPECT_TRUE(found_t) << "Missing 't contraction";
    EXPECT_TRUE(found_re) << "Missing 're contraction";
    EXPECT_TRUE(found_ve) << "Missing 've contraction";
    EXPECT_TRUE(found_ll) << "Missing 'll contraction";
    EXPECT_TRUE(found_d) << "Missing 'd contraction";
}

TEST(Pretokenizer, DigitsSeparateFromLetters) {
    auto result = pretokenize("hello123world");
    // "hello", "123", "world" should be separate chunks
    ASSERT_GE(result.size(), 3u);
}

TEST(Pretokenizer, PunctuationSeparate) {
    auto result = pretokenize("hello, world! yes.");
    // Punctuation should be in its own chunk
    bool found_comma = false, found_bang = false, found_dot = false;
    for (auto& s : result) {
        if (s.find(',') != std::string::npos) found_comma = true;
        if (s.find('!') != std::string::npos) found_bang = true;
        if (s.find('.') != std::string::npos) found_dot = true;
    }
    EXPECT_TRUE(found_comma);
    EXPECT_TRUE(found_bang);
    EXPECT_TRUE(found_dot);
}

TEST(Pretokenizer, WhitespaceRuns) {
    auto result = pretokenize("a   b");
    // Multiple spaces should be handled
    std::string joined;
    for (auto& s : result) joined += s;
    EXPECT_EQ(joined, "a   b");
}

TEST(Pretokenizer, NewlinesAndTabs) {
    auto result = pretokenize("hello\n\nworld\ttab");
    std::string joined;
    for (auto& s : result) joined += s;
    EXPECT_EQ(joined, "hello\n\nworld\ttab");
}

TEST(Pretokenizer, Reconstruction) {
    // The most important invariant: concatenating all chunks == original text.
    std::vector<std::string> texts = {
        "Hello, world!",
        "I'm happy. You're great.",
        "  leading spaces  ",
        "123abc456def",
        "a\nb\tc",
        "hello!!!???",
        "",
        "   ",
    };
    for (auto& text : texts) {
        auto result = pretokenize(text);
        std::string joined;
        for (auto& s : result) joined += s;
        EXPECT_EQ(joined, text) << "Reconstruction failed for: \"" << text << "\"";
    }
}

TEST(Pretokenizer, UnicodeBasic) {
    auto result = pretokenize("café résumé");
    std::string joined;
    for (auto& s : result) joined += s;
    EXPECT_EQ(joined, "café résumé");
}

// ═══════════════════════════════════════════════════════════════════
// Special token splitting
// ═══════════════════════════════════════════════════════════════════

TEST(PretokenizerSpecials, NoSpecialTokens) {
    auto result = pretokenize_with_specials("hello world", {});
    // Should behave like regular pretokenize
    ASSERT_FALSE(result.empty());
    for (auto& pt : result) {
        EXPECT_FALSE(pt.is_special);
    }
}

TEST(PretokenizerSpecials, RejectsEmptySpecialToken) {
    EXPECT_THROW(
        pretokenize_with_specials("hello world", {""}),
        std::invalid_argument);
}

TEST(PretokenizerSpecials, SingleSpecialToken) {
    auto result = pretokenize_with_specials(
        "hello<|endoftext|>world",
        {"<|endoftext|>"});

    bool found_special = false;
    std::string full_text;
    for (auto& pt : result) {
        full_text += pt.text;
        if (pt.text == "<|endoftext|>") {
            EXPECT_TRUE(pt.is_special);
            found_special = true;
        }
    }
    EXPECT_TRUE(found_special);
    EXPECT_EQ(full_text, "hello<|endoftext|>world");
}

TEST(PretokenizerSpecials, MultipleSpecialTokens) {
    auto result = pretokenize_with_specials(
        "<|pad|>hello<|endoftext|>world<|pad|>",
        {"<|endoftext|>", "<|pad|>"});

    int special_count = 0;
    std::string full_text;
    for (auto& pt : result) {
        full_text += pt.text;
        if (pt.is_special) special_count++;
    }
    EXPECT_EQ(special_count, 3);
    EXPECT_EQ(full_text, "<|pad|>hello<|endoftext|>world<|pad|>");
}

TEST(PretokenizerSpecials, OverlappingPrefixes) {
    // "<|end|>" and "<|endoftext|>" — longest match should win
    auto result = pretokenize_with_specials(
        "a<|endoftext|>b<|end|>c",
        {"<|end|>", "<|endoftext|>"});

    std::string full_text;
    for (auto& pt : result) full_text += pt.text;
    EXPECT_EQ(full_text, "a<|endoftext|>b<|end|>c");
}

TEST(PretokenizerSpecials, AdjacentSpecialTokens) {
    auto result = pretokenize_with_specials(
        "<|endoftext|><|endoftext|>",
        {"<|endoftext|>"});

    int special_count = 0;
    for (auto& pt : result) {
        if (pt.is_special) special_count++;
    }
    EXPECT_EQ(special_count, 2);
}

TEST(PretokenizerSpecials, SpecialTokenOnly) {
    auto result = pretokenize_with_specials(
        "<|endoftext|>",
        {"<|endoftext|>"});

    ASSERT_EQ(result.size(), 1);
    EXPECT_TRUE(result[0].is_special);
    EXPECT_EQ(result[0].text, "<|endoftext|>");
}
