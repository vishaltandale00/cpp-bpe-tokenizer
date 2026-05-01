#include <gtest/gtest.h>
#include "bpe/trainer.h"
#include "bpe/tokenizer.h"
#include "test_helpers.h"

using namespace bpe;

// ═══════════════════════════════════════════════════════════════════
// Low-level training primitives
// ═══════════════════════════════════════════════════════════════════

TEST(CountPairs, EmptyChunks) {
    std::vector<std::vector<uint32_t>> chunks;
    auto counts = count_pairs(chunks);
    EXPECT_TRUE(counts.empty());
}

TEST(CountPairs, SingleByteChunks) {
    // Chunks of length 1 have no pairs.
    std::vector<std::vector<uint32_t>> chunks = {{65}, {66}, {67}};
    auto counts = count_pairs(chunks);
    EXPECT_TRUE(counts.empty());
}

TEST(CountPairs, SimplePair) {
    std::vector<std::vector<uint32_t>> chunks = {{1, 2, 3}};
    auto counts = count_pairs(chunks);
    TokenPair p12{1, 2}, p23{2, 3};
    EXPECT_EQ(counts[p12], 1u);
    EXPECT_EQ(counts[p23], 1u);
    EXPECT_EQ(counts.size(), 2u);
}

TEST(CountPairs, RepeatedPair) {
    // "abab" → (a,b)=2, (b,a)=1
    std::vector<std::vector<uint32_t>> chunks = {{1, 2, 1, 2}};
    auto counts = count_pairs(chunks);
    TokenPair p12{1, 2}, p21{2, 1};
    EXPECT_EQ(counts[p12], 2u);
    EXPECT_EQ(counts[p21], 1u);
}

TEST(CountPairs, AcrossChunks) {
    // Pairs do NOT span across chunks.
    std::vector<std::vector<uint32_t>> chunks = {{1, 2}, {2, 3}};
    auto counts = count_pairs(chunks);
    TokenPair p12{1, 2}, p23{2, 3}, p22{2, 2};
    EXPECT_EQ(counts[p12], 1u);
    EXPECT_EQ(counts[p23], 1u);
    // (2, 2) should NOT exist — it would require spanning chunks.
    EXPECT_EQ(counts.count(p22), 0u);
}

TEST(CountPairs, MultipleChunksSamePair) {
    std::vector<std::vector<uint32_t>> chunks = {{1, 2}, {1, 2}, {1, 2}};
    auto counts = count_pairs(chunks);
    TokenPair p12{1, 2};
    EXPECT_EQ(counts[p12], 3u);
}

// ═══════════════════════════════════════════════════════════════════
// Best pair selection
// ═══════════════════════════════════════════════════════════════════

TEST(FindBestPair, SinglePair) {
    std::unordered_map<TokenPair, uint64_t, TokenPairHash> counts = {
        {{1, 2}, 5}
    };
    auto best = find_best_pair(counts);
    EXPECT_EQ(best.pair.left, 1u);
    EXPECT_EQ(best.pair.right, 2u);
    EXPECT_EQ(best.count, 5u);
}

TEST(FindBestPair, HighestFrequency) {
    std::unordered_map<TokenPair, uint64_t, TokenPairHash> counts = {
        {{1, 2}, 5},
        {{3, 4}, 10},
        {{5, 6}, 3},
    };
    auto best = find_best_pair(counts);
    EXPECT_EQ(best.pair.left, 3u);
    EXPECT_EQ(best.pair.right, 4u);
    EXPECT_EQ(best.count, 10u);
}

TEST(FindBestPair, TieBreakByLargerPair) {
    // Same frequency — pick lexicographically greater (left, right).
    std::unordered_map<TokenPair, uint64_t, TokenPairHash> counts = {
        {{1, 2}, 5},
        {{3, 4}, 5},
        {{2, 3}, 5},
    };
    auto best = find_best_pair(counts);
    EXPECT_EQ(best.count, 5u);
    // (3, 4) > (2, 3) > (1, 2) lexicographically
    EXPECT_EQ(best.pair.left, 3u);
    EXPECT_EQ(best.pair.right, 4u);
}

TEST(FindBestPair, TieBreakByRight) {
    // Same left, different right.
    std::unordered_map<TokenPair, uint64_t, TokenPairHash> counts = {
        {{5, 2}, 10},
        {{5, 8}, 10},
    };
    auto best = find_best_pair(counts);
    EXPECT_EQ(best.pair.left, 5u);
    EXPECT_EQ(best.pair.right, 8u);
}

// ═══════════════════════════════════════════════════════════════════
// Merge application
// ═══════════════════════════════════════════════════════════════════

TEST(ApplyMerge, BasicMerge) {
    std::vector<std::vector<uint32_t>> chunks = {{1, 2, 3}};
    apply_merge(chunks, 1, 2, 256);
    ASSERT_EQ(chunks[0].size(), 2u);
    EXPECT_EQ(chunks[0][0], 256u);
    EXPECT_EQ(chunks[0][1], 3u);
}

TEST(ApplyMerge, MultipleSites) {
    std::vector<std::vector<uint32_t>> chunks = {{1, 2, 3, 1, 2}};
    apply_merge(chunks, 1, 2, 256);
    ASSERT_EQ(chunks[0].size(), 3u);
    EXPECT_EQ(chunks[0][0], 256u);
    EXPECT_EQ(chunks[0][1], 3u);
    EXPECT_EQ(chunks[0][2], 256u);
}

TEST(ApplyMerge, OverlappingTriple) {
    // [a, a, a] with merge (a, a): left-to-right, first (a,a) merges,
    // leaving [aa, a] — only one merge.
    std::vector<std::vector<uint32_t>> chunks = {{1, 1, 1}};
    apply_merge(chunks, 1, 1, 256);
    ASSERT_EQ(chunks[0].size(), 2u);
    EXPECT_EQ(chunks[0][0], 256u);
    EXPECT_EQ(chunks[0][1], 1u);
}

TEST(ApplyMerge, NoMatch) {
    std::vector<std::vector<uint32_t>> chunks = {{1, 2, 3}};
    apply_merge(chunks, 4, 5, 256);
    ASSERT_EQ(chunks[0].size(), 3u);
    EXPECT_EQ(chunks[0][0], 1u);
}

TEST(ApplyMerge, AcrossMultipleChunks) {
    std::vector<std::vector<uint32_t>> chunks = {{1, 2}, {3, 4}, {1, 2}};
    apply_merge(chunks, 1, 2, 256);
    EXPECT_EQ(chunks[0], std::vector<uint32_t>({256}));
    EXPECT_EQ(chunks[1], std::vector<uint32_t>({3, 4}));
    EXPECT_EQ(chunks[2], std::vector<uint32_t>({256}));
}

// ═══════════════════════════════════════════════════════════════════
// Full training loop
// ═══════════════════════════════════════════════════════════════════

TEST(TrainBPE, NoMergesNeeded) {
    std::vector<std::vector<uint32_t>> chunks = {{65}, {66}};
    auto merges = train_bpe(chunks, 256); // already at target
    EXPECT_TRUE(merges.empty());
}

TEST(TrainBPE, SingleMerge) {
    // "ab ab ab" → three chunks of [a, b], pair (a,b) most frequent.
    std::vector<std::vector<uint32_t>> chunks = {
        {97, 98}, {97, 98}, {97, 98}
    };
    auto merges = train_bpe(chunks, 257);
    ASSERT_EQ(merges.size(), 1u);
    EXPECT_EQ(merges[0].first.left, 97u);  // 'a'
    EXPECT_EQ(merges[0].first.right, 98u); // 'b'
    EXPECT_EQ(merges[0].second, 256u);     // first new ID
}

TEST(TrainBPE, MultipleMerges) {
    // "abcd abcd" — train 4 merges.
    std::vector<std::vector<uint32_t>> chunks = {
        {97, 98, 99, 100}, {97, 98, 99, 100}
    };
    auto merges = train_bpe(chunks, 260);
    EXPECT_EQ(merges.size(), 3u);
    // After all merges, each chunk should be a single token.
}

// ═══════════════════════════════════════════════════════════════════
// Full Tokenizer training
// ═══════════════════════════════════════════════════════════════════

TEST(TokenizerTrain, BasicTraining) {
    bpe::Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 280);
    EXPECT_EQ(tok.vocab_size(), 280u);
    EXPECT_EQ(tok.merges().size(), 280u - 256u);
}

TEST(TokenizerTrain, WithSpecialTokens) {
    bpe::Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 282, {"<|endoftext|>", "<|pad|>"});
    EXPECT_EQ(tok.vocab_size(), 282u);
    // Special tokens should have IDs
    auto& specials = tok.special_tokens();
    EXPECT_NE(specials.find("<|endoftext|>"), specials.end());
    EXPECT_NE(specials.find("<|pad|>"), specials.end());
}

TEST(TokenizerTrain, VocabContainsAllBytes) {
    bpe::Tokenizer tok;
    tok.train("hello", 260);
    auto& v = tok.vocab();
    // All 256 byte tokens must be present.
    for (uint32_t i = 0; i < 256; ++i) {
        EXPECT_NE(v.find(i), v.end()) << "Missing byte token " << i;
    }
}

TEST(TokenizerTrain, MergesAreOrdered) {
    bpe::Tokenizer tok;
    tok.train(test_helpers::tiny_corpus(), 290);
    auto& m = tok.merges();
    // Merged IDs should be sequential starting from 256.
    for (size_t i = 0; i < m.size(); ++i) {
        EXPECT_EQ(m[i].merged, 256u + i);
    }
}
