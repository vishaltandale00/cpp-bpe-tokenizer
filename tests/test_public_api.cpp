#include <gtest/gtest.h>
#include "bpe/codec.h"
#include "bpe/tokenizer.h"
#include "test_helpers.h"

TEST(PublicApi, TokenizerHasSmallStableHandle) {
    EXPECT_LE(sizeof(bpe::Tokenizer), 64u);
}

TEST(PublicApi, CodecHeaderDoesNotRequireTokenizerInternals) {
    std::vector<uint32_t> ids = bpe::bytes_to_ids("abc");
    EXPECT_EQ(ids, std::vector<uint32_t>({97, 98, 99}));
}

TEST(PublicApi, CopyAfterCacheBuildIsIndependent) {
    bpe::Tokenizer original;
    original.train(test_helpers::tiny_corpus(), 280, {"<|endoftext|>"});

    const std::string text = "the cat<|endoftext|>sat";
    const auto expected = original.encode(text);
    bpe::Tokenizer copy = original;

    original.train("zz zz zz", 260);

    EXPECT_EQ(copy.encode(text), expected);
    EXPECT_EQ(copy.decode(expected), text);
}
