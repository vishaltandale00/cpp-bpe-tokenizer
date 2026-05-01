#pragma once

#include <filesystem>
#include <string>

namespace bpe {

// Read an entire file into a string.
std::string read_file(const std::filesystem::path& path);

// Write a string to a file (creates/overwrites).
void write_file(const std::filesystem::path& path, const std::string& content);

} // namespace bpe
