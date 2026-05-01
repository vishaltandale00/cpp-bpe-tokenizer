#include <gtest/gtest.h>
#include "bpe/tokenizer.h"
#include "test_helpers.h"

using namespace bpe;

// ═══════════════════════════════════════════════════════════════════
// Roundtrip tests: encode(decode(x)) == x for all valid inputs.
// This is the most critical correctness property.
// ═══════════════════════════════════════════════════════════════════

class RoundtripTest : public ::testing::Test {
protected:
    void SetUp() override {
        tok = test_helpers::train_tiny(300);
    }
    Tokenizer tok;
};

TEST_F(RoundtripTest, EmptyString) {
    EXPECT_EQ(tok.decode(tok.encode("")), "");
}

TEST_F(RoundtripTest, SingleASCIIChar) {
    for (int c = 32; c < 127; ++c) {
        std::string s(1, static_cast<char>(c));
        EXPECT_EQ(tok.decode(tok.encode(s)), s)
            << "Failed for char: " << c << " '" << s << "'";
    }
}

TEST_F(RoundtripTest, AllSingleBytes) {
    // Every single byte should roundtrip.
    for (int b = 0; b < 256; ++b) {
        std::string s(1, static_cast<char>(b));
        auto ids = tok.encode(s);
        auto decoded = tok.decode(ids);
        EXPECT_EQ(decoded, s) << "Failed for byte: " << b;
    }
}

TEST_F(RoundtripTest, ASCIIText) {
    std::string text = "Hello, world! How are you today?";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, TrainingCorpus) {
    std::string text = test_helpers::tiny_corpus();
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, UnseenText) {
    std::string text = "This text was never in the training corpus! 12345 @#$%";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, Whitespace) {
    std::vector<std::string> texts = {
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "  hello  world  ",
        "\t\t\n\n",
    };
    for (auto& text : texts) {
        EXPECT_EQ(tok.decode(tok.encode(text)), text)
            << "Failed for whitespace text";
    }
}

TEST_F(RoundtripTest, Punctuation) {
    std::string text = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, Numbers) {
    std::string text = "0123456789 3.14159 -42 1e10 0xFF";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, UTF8TwoByteChars) {
    // Latin extended: é, ñ, ü
    std::string text = "café résumé naïve über";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, UTF8ThreeByteChars) {
    // CJK characters (3 bytes each in UTF-8)
    std::string text = "你好世界 こんにちは 안녕하세요";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, UTF8FourByteChars) {
    // Emoji (4 bytes each in UTF-8)
    std::string text = "Hello 🌍🚀💡 World";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, MixedUnicode) {
    std::string text = "Hello café 你好 🌍 naïve";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, LongRepetitiveText) {
    // Should benefit from BPE merges.
    std::string text;
    for (int i = 0; i < 100; ++i) {
        text += "the cat sat on the mat. ";
    }
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripTest, LongRandomText) {
    std::string text = test_helpers::medium_corpus(10);
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

// ═══════════════════════════════════════════════════════════════════
// Roundtrip with special tokens
// ═══════════════════════════════════════════════════════════════════

class RoundtripSpecialTest : public ::testing::Test {
protected:
    void SetUp() override {
        tok.train(test_helpers::tiny_corpus(), 300, {"<|endoftext|>"});
    }
    Tokenizer tok;
};

TEST_F(RoundtripSpecialTest, TextWithSpecialTokens) {
    std::string text = "hello<|endoftext|>world";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripSpecialTest, OnlySpecialToken) {
    std::string text = "<|endoftext|>";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST_F(RoundtripSpecialTest, RepeatedSpecialTokens) {
    std::string text = "<|endoftext|>hello<|endoftext|>world<|endoftext|>";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}
