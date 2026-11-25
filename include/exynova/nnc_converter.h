#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace exynova {
namespace compiler {

struct NncConvertReport {
    bool ok = false;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    std::string original_magic;
    std::string output_magic;
    std::string summary;
    std::string error_msg;
};

/**
 * Validates a TFLite v3 ("TFL3") or ENN ("ENNC") FlatBuffer payload in memory
 * and produces a valid "ENNC" container by rewriting the 4-byte file identifier
 * at offset 4 after checking FlatBuffer header, root table, and vtable bounds.
 */
bool convert_tflite_bytes_to_nnc(const std::vector<uint8_t>& tflite_bytes,
                                 std::vector<uint8_t>& out_nnc_bytes,
                                 NncConvertReport* report = nullptr);

/**
 * Loads a .tflite (or .nnc) file from disk, validates and converts it to an
 * ENNC FlatBuffer container, and writes the resulting bytes to out_nnc_path.
 */
bool convert_tflite_file_to_nnc(const std::string& tflite_path,
                                const std::string& out_nnc_path,
                                NncConvertReport* report = nullptr);

} // namespace compiler
} // namespace exynova
