#pragma once

#include "bpe/tokenizer.h"
#include <string>

namespace bench_helpers {

inline std::string tiny_corpus() {
    return "the cat sat on the mat. the cat ate the rat. "
           "the dog sat on the log. the dog ate the frog.";
}

inline std::string medium_corpus(size_t repeat = 100) {
    std::string base =
        "In a small village, there lived a young girl named Lily. She loved to explore "
        "the forest near her home. One day, she found a shiny stone by the river. "
        "The stone glowed with a warm light. Lily picked it up and felt happy. "
        "She showed it to her friend Tom. Tom said it was a magic stone. "
        "They decided to go on an adventure together. They walked through the tall trees "
        "and crossed the old bridge. On the other side, they found a beautiful garden. "
        "The garden had flowers of every color. Lily and Tom played there until sunset. "
        "When they went home, Lily put the stone on her shelf. It still glowed softly. "
        "She smiled and went to sleep, dreaming of their next adventure.\n";
    std::string result;
    result.reserve(base.size() * repeat);
    for (size_t i = 0; i < repeat; ++i) result += base;
    return result;
}

inline std::string holdout_code_corpus(size_t repeat = 32) {
    std::string base =
        "template <typename It>\n"
        "uint64_t hash_bytes(It begin, It end) {\n"
        "    uint64_t h = 1469598103934665603ULL;\n"
        "    for (It it = begin; it != end; ++it) {\n"
        "        h ^= static_cast<uint8_t>(*it);\n"
        "        h *= 1099511628211ULL;\n"
        "    }\n"
        "    return h;\n"
        "}\n"
        "\n"
        "def render_report(rows):\n"
        "    payload = {\"ok\": True, \"rows\": rows, \"count\": len(rows)}\n"
        "    return json.dumps(payload, indent=2, sort_keys=True)\n"
        "\n"
        "SELECT token, count(*) AS freq\n"
        "FROM corpus_tokens\n"
        "WHERE token LIKE 'pre%'\n"
        "GROUP BY token\n"
        "ORDER BY freq DESC;\n";
    std::string result;
    result.reserve(base.size() * repeat);
    for (size_t i = 0; i < repeat; ++i) result += base;
    return result;
}

inline std::string holdout_punctuation_corpus(size_t repeat = 80) {
    std::string base =
        "!!! ??? ... :: == != <= >= -> <- => {[]} () <> // ## @@ ~~ || && %% $$ `` '' \"\"\n"
        "*** --- ___ +++ ::: ;;; ,,, ... ??? !!! ::: === !== >>> <<< [[ ]] {{ }}\n";
    std::string result;
    result.reserve(base.size() * repeat);
    for (size_t i = 0; i < repeat; ++i) result += base;
    return result;
}

inline std::string holdout_multilingual_corpus(size_t repeat = 48) {
    std::string base =
        "Hola mundo. Bonjour le monde. Hallo Welt. Ciao mondo.\n"
        "नमस्ते दुनिया। مرحبا بالعالم. שלום עולם. Привет, мир.\n"
        "こんにちは世界。你好，世界。안녕하세요 세계. สวัสดีชาวโลก.\n"
        "Emoji mix: 🌍🚀✨🔥🎉🧠📦 and accented text: café résumé naïve coöperate.\n";
    std::string result;
    result.reserve(base.size() * repeat);
    for (size_t i = 0; i < repeat; ++i) result += base;
    return result;
}

// ~1MB of text
inline std::string large_corpus() {
    return medium_corpus(2000);
}

// Pre-trained tokenizer for encode/decode benchmarks.
inline bpe::Tokenizer& cached_tokenizer_small() {
    static bpe::Tokenizer tok = [] {
        bpe::Tokenizer t;
        t.train(medium_corpus(10), 500);
        return t;
    }();
    return tok;
}

inline bpe::Tokenizer& cached_tokenizer_large() {
    static bpe::Tokenizer tok = [] {
        bpe::Tokenizer t;
        t.train(medium_corpus(100), 2000);
        return t;
    }();
    return tok;
}

} // namespace bench_helpers
