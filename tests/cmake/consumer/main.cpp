#include <bpe/tokenizer.h>

#include <cstdint>
#include <string>
#include <vector>

int main() {
    bpe::Tokenizer tokenizer;
    tokenizer.train("the cat sat on the mat. the cat ate the rat.", 300);

    const std::string input = "the cat sat";
    const std::vector<uint32_t> ids = tokenizer.encode(input);
    return tokenizer.decode(ids) == input ? 0 : 1;
}
