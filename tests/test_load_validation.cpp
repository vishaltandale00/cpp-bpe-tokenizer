#include <gtest/gtest.h>
#include "bpe/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

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

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

uint32_t byte_level_cp(uint8_t byte) {
    if ((byte >= '!' && byte <= '~') || (byte >= 0xA1 && byte <= 0xAC) || byte >= 0xAE) {
        return byte;
    }
    uint32_t extra = 0;
    for (uint32_t b = 0; b < byte; ++b) {
        if (!((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE)) {
            ++extra;
        }
    }
    return 256 + extra;
}

std::string byte_level_token(std::string_view bytes) {
    std::string token;
    for (unsigned char byte : bytes) {
        append_utf8(token, byte_level_cp(byte));
    }
    return token;
}

nlohmann::json valid_hf_tokenizer_json() {
    constexpr std::string_view qwen_regex =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    auto vocab = nlohmann::json::object();
    for (uint32_t i = 0; i < 256; ++i) {
        const std::string byte(1, static_cast<char>(i));
        vocab[byte_level_token(byte)] = i;
    }
    vocab[byte_level_token("bc")] = 256;
    vocab[byte_level_token("ab")] = 257;
    vocab[byte_level_token("abc")] = 258;

    nlohmann::json model = {
        {"type", "BPE"},
        {"byte_fallback", false},
        {"ignore_merges", false},
        {"vocab", std::move(vocab)},
        {"merges", nlohmann::json::array({
            nlohmann::json::array({byte_level_token("b"), byte_level_token("c")}),
            nlohmann::json::array({byte_level_token("a"), byte_level_token("b")}),
            nlohmann::json::array({byte_level_token("ab"), byte_level_token("c")}),
        })},
    };

    return {
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer", {
            {"type", "Sequence"},
            {"pretokenizers", nlohmann::json::array({
                {
                    {"type", "Split"},
                    {"pattern", {{"Regex", qwen_regex}}},
                    {"behavior", "Isolated"},
                    {"invert", false},
                },
                {
                    {"type", "ByteLevel"},
                    {"add_prefix_space", false},
                    {"trim_offsets", false},
                    {"use_regex", false},
                },
            })},
        }},
        {"decoder", {
            {"type", "ByteLevel"},
            {"add_prefix_space", false},
            {"trim_offsets", false},
            {"use_regex", false},
        }},
        {"post_processor", {
            {"type", "ByteLevel"},
            {"add_prefix_space", false},
            {"trim_offsets", false},
            {"use_regex", false},
        }},
        {"model", std::move(model)},
        {"added_tokens", nlohmann::json::array({
            {
                {"id", 259},
                {"content", "<|im_start|>"},
                {"single_word", false},
                {"lstrip", false},
                {"rstrip", false},
                {"normalized", false},
                {"special", true},
            },
        })},
    };
}

std::filesystem::path write_hf_tokenizer_json(const std::string& name, const nlohmann::json& j) {
    auto dir = unique_temp_dir(name);
    write_text(dir / "tokenizer.json", j.dump());
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

TEST(LoadValidation, HfTokenizerJsonLoadsFileAndPreservesPairIdentity) {
    auto dir = write_hf_tokenizer_json("hf_tokenizer_json", valid_hf_tokenizer_json());
    auto tok = bpe::Tokenizer::load(dir / "tokenizer.json");

    EXPECT_EQ(tok.vocab_size(), 260u);
    EXPECT_EQ(tok.special_tokens().at("<|im_start|>"), 259u);

    const std::vector<uint32_t> expected = {97, 256};
    EXPECT_EQ(tok.encode("abc"), expected);
    EXPECT_EQ(tok.encode("<|im_start|>abc"), (std::vector<uint32_t>{259, 97, 256}));
    EXPECT_EQ(tok.decode({259, 97, 256}), "<|im_start|>abc");
    EXPECT_EQ(tok.encode("e\xCC\x81"), tok.encode("\xC3\xA9"));
    EXPECT_EQ(tok.decode(tok.encode("e\xCC\x81")), "\xC3\xA9");

    EXPECT_THROW(tok.save(unique_temp_dir("hf_save_rejects")), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST(LoadValidation, HfTokenizerJsonRejectsUnsupportedAddedTokenFlags) {
    auto j = valid_hf_tokenizer_json();
    j["added_tokens"][0]["normalized"] = true;
    auto dir = write_hf_tokenizer_json("hf_bad_added_flags", j);

    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST(LoadValidation, HfTokenizerJsonRejectsUnsupportedPipeline) {
    auto j = valid_hf_tokenizer_json();
    j["pre_tokenizer"] = {
        {"type", "ByteLevel"},
        {"add_prefix_space", false},
        {"trim_offsets", true},
    };
    auto dir = write_hf_tokenizer_json("hf_bad_pipeline", j);

    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST(LoadValidation, HfTokenizerJsonMissingRequiredFieldThrowsRuntimeError) {
    auto j = valid_hf_tokenizer_json();
    j.erase("model");
    auto dir = write_hf_tokenizer_json("hf_missing_model", j);

    EXPECT_THROW(bpe::Tokenizer::load(dir), std::runtime_error);
    std::filesystem::remove_all(dir);
}
