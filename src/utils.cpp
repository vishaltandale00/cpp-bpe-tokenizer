#include "bpe/utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bpe {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot write file: " + path.string());
    }
    f << content;
}

} // namespace bpe
