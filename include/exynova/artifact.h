#pragma once

#include "exynova/ir.h"
#include "exynova/partitioner.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace exynova {
namespace compiler {

constexpr uint32_t EXYN_MAGIC = 0x4E595845u; // "EXYN" little-endian
constexpr uint32_t EXYN_FORMAT_VERSION = 1u;

#pragma pack(push, 1)
struct ExynHeader {
    uint32_t magic = EXYN_MAGIC;
    uint32_t format_version = EXYN_FORMAT_VERSION;
    char target_soc[16] = {};
    uint32_t flags = 0;
    uint32_t payload_size = 0;
    uint32_t crc32 = 0;
};
#pragma pack(pop)

struct EmbeddedBlob {
    std::string tag;
    std::vector<uint8_t> data;
};

struct Artifact {
    std::string target_soc = "S5E8855";
    uint32_t flags = 0;
    Graph graph;
    ExecutionPlan plan;
    std::vector<EmbeddedBlob> blobs;
};

uint32_t compute_crc32(const uint8_t* data, size_t size);

bool serialize(const Artifact& artifact,
               std::vector<uint8_t>& out_bytes,
               std::string* err = nullptr);

bool deserialize(const uint8_t* data,
                 size_t size,
                 Artifact& out_artifact,
                 std::string* err = nullptr);

bool save_file(const std::string& path,
               const Artifact& artifact,
               std::string* err = nullptr);

bool load_file(const std::string& path,
               Artifact& out_artifact,
               std::string* err = nullptr);

} // namespace compiler
} // namespace exynova
