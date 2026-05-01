#include <gtest/gtest.h>
#include "bpe/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path unique_temp_dir(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / ("bpe_load_validation_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void write_text(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << content;
}

std::string byte_vocab_json() {
    std::string json = "{";
    for (int i = 0; i < 256; ++i) {
        if (i != 0) {
            json += ",";
        }
        json += "\"" + std::to_string(i) + "\":[" + std::to_string(i) + "]";
    }
    json += "}";
    return json;
}

std::string swapped_byte_vocab_json() {
    std::string json = "{";
    for (int i = 0; i < 256; ++i) {
        if (i != 0) {
            json += ",";
        }
        const int byte = (i == 0) ? 1 : (i == 1) ? 0 : i;
        json += "\"" + std::to_string(i) + "\":[" + std::to_string(byte) + "]";
    }
    json += "}";
    return json;
}

std::string sparse_byte_vocab_json() {
    std::string json = "{";
    for (int i = 0; i < 255; ++i) {
        if (i != 0) {
            json += ",";
        }
        json += "\"" + std::to_string(i) + "\":[" + std::to_string(i) + "]";
    }
    json += ",\"300\":[255]}";
    return json;
}

std::filesystem::path write_valid_byte_model(const std::string& name) {
    auto dir = unique_temp_dir(name);
    write_text(dir / "vocab.json", byte_vocab_json());
    write_text(dir / "merges.txt", "");
    return dir;
}

} // namespace

TEST(LoadValidation, MissingDirectoryThrows) {
    auto dir = std::filesystem::temp_directory_path() / "bpe_missing_model_dir";
    std::filesystem::remove_all(dir);
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, MissingVocabThrows) {
    auto dir = unique_temp_dir("missing_vocab");
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, MissingMergesThrows) {
    auto dir = unique_temp_dir("missing_merges");
    write_text(dir / "vocab.json", byte_vocab_json());
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, CorruptVocabThrows) {
    auto dir = unique_temp_dir("corrupt_vocab");
    write_text(dir / "vocab.json", "{not json");
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, OddLengthHexVocabKeyThrows) {
    auto dir = unique_temp_dir("odd_hex");
    write_text(dir / "vocab.json", "{\"0\":0}");
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, NonHexVocabKeyThrows) {
    auto dir = unique_temp_dir("non_hex");
    write_text(dir / "vocab.json", "{\"zz\":0}");
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, NativeByteOutOfRangeThrows) {
    auto dir = unique_temp_dir("byte_range");
    write_text(dir / "vocab.json", "{\"0\":[256]}");
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, InvalidNativeTokenIdThrows) {
    auto dir = unique_temp_dir("bad_native_id");
    write_text(dir / "vocab.json", "{\"abc\":[0]}");
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, NativeByteTokenIdsMustMatchByteValues) {
    auto dir = unique_temp_dir("swapped_byte_ids");
    write_text(dir / "vocab.json", swapped_byte_vocab_json());
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, VocabIdsMustBeContiguous) {
    auto dir = unique_temp_dir("sparse_vocab_ids");
    write_text(dir / "vocab.json", sparse_byte_vocab_json());
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, UnparseableMergeLineThrows) {
    auto dir = write_valid_byte_model("bad_merge_line");
    write_text(dir / "merges.txt", "1 2 3 4\n");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, UnknownMergeTokenThrows) {
    auto dir = write_valid_byte_model("unknown_merge");
    write_text(dir / "merges.txt", "1 2 999\n");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, UnmergedVocabTokenThrows) {
    auto dir = unique_temp_dir("unmerged_token");
    std::string vocab = byte_vocab_json();
    vocab.pop_back();
    vocab += ",\"256\":[1,2]}";
    write_text(dir / "vocab.json", vocab);
    write_text(dir / "merges.txt", "");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, MergeByteMismatchThrows) {
    auto dir = unique_temp_dir("merge_mismatch");
    std::string vocab = byte_vocab_json();
    vocab.pop_back();
    vocab += ",\"256\":[1,3]}";
    write_text(dir / "vocab.json", vocab);
    write_text(dir / "merges.txt", "1 2 256\n");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, NativeMergeIdsMustFollowPriorityOrder) {
    auto dir = unique_temp_dir("merge_order");
    std::string vocab = byte_vocab_json();
    vocab.pop_back();
    vocab += ",\"257\":[1,2]}";
    write_text(dir / "vocab.json", vocab);
    write_text(dir / "merges.txt", "1 2 257\n");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, InvalidSpecialTokenLineThrows) {
    auto dir = write_valid_byte_model("bad_special");
    write_text(dir / "special_tokens.txt", "not_an_id <|bad|>\n");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, SpecialTokenIdsMustFollowVocab) {
    auto dir = write_valid_byte_model("special_id_range");
    write_text(dir / "special_tokens.txt", "300 <|bad|>\n");
    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
}

TEST(LoadValidation, AbsentSpecialTokensFileLoads) {
    auto dir = write_valid_byte_model("no_specials");
    auto tok = bpe::Tokenizer::load(dir);
    EXPECT_EQ(tok.vocab_size(), 256u);
    EXPECT_TRUE(tok.special_tokens().empty());
}
