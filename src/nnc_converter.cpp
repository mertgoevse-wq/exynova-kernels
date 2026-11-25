#include "exynova/nnc_converter.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace exynova {
namespace compiler {

namespace {

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

int32_t read_i32_le(const uint8_t* p) {
    const uint32_t u = read_u32_le(p);
    int32_t v = 0;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

bool in_bounds(size_t off, size_t len, size_t total) {
    return off <= total && len <= (total - off);
}

bool fail_report(NncConvertReport* report,
                 size_t in_size,
                 const std::string& orig_magic,
                 const std::string& msg) {
    if (report) {
        report->ok = false;
        report->input_bytes = in_size;
        report->output_bytes = 0;
        report->original_magic = orig_magic;
        report->output_magic.clear();
        report->summary.clear();
        report->error_msg = msg;
    }
    return false;
}

} // namespace

bool convert_tflite_bytes_to_nnc(const std::vector<uint8_t>& tflite_bytes,
                                 std::vector<uint8_t>& out_nnc_bytes,
                                 NncConvertReport* report) {
    const size_t n = tflite_bytes.size();
    out_nnc_bytes.clear();

    if (n < 12) {
        return fail_report(report, n, "", "Input smaller than 12-byte minimum FlatBuffer header");
    }

    const uint8_t* data = tflite_bytes.data();
    const std::string magic(reinterpret_cast<const char*>(data + 4), 4);
    if (magic != "TFL3" && magic != "ENNC") {
        return fail_report(report, n, magic,
                           "Unsupported FlatBuffer identifier '" + magic +
                               "' (expected 'TFL3' or 'ENNC')");
    }

    const uint32_t root_off = read_u32_le(data);
    if (root_off < 8 || !in_bounds(root_off, 4, n)) {
        return fail_report(report, n, magic, "FlatBuffer root offset out of bounds");
    }

    const int32_t vt_rel = read_i32_le(data + root_off);
    const int64_t vt_off_i64 = static_cast<int64_t>(root_off) - static_cast<int64_t>(vt_rel);
    if (vt_off_i64 < 8 || !in_bounds(static_cast<size_t>(vt_off_i64), 4, n)) {
        return fail_report(report, n, magic, "FlatBuffer root vtable offset out of bounds");
    }

    const size_t vt_off = static_cast<size_t>(vt_off_i64);
    const uint16_t vtable_size = read_u16_le(data + vt_off);
    const uint16_t table_size = read_u16_le(data + vt_off + 2);
    if (vtable_size < 4 || (vtable_size % 2) != 0 || !in_bounds(vt_off, vtable_size, n)) {
        return fail_report(report, n, magic, "Invalid FlatBuffer root vtable size");
    }
    if (table_size < 4 || !in_bounds(root_off, table_size, n)) {
        return fail_report(report, n, magic, "Invalid FlatBuffer root table size");
    }

    const size_t num_fields = (vtable_size - 4) / 2;
    for (size_t i = 0; i < num_fields; ++i) {
        const uint16_t f_off = read_u16_le(data + vt_off + 4 + 2 * i);
        if (f_off != 0 && f_off >= table_size) {
            return fail_report(report, n, magic, "FlatBuffer root vtable field exceeds table size");
        }
    }

    auto field_pos = [&](size_t idx) -> size_t {
        if (idx >= num_fields) return 0;
        const uint16_t rel = read_u16_le(data + vt_off + 4 + 2 * idx);
        return rel == 0 ? 0 : static_cast<size_t>(root_off) + rel;
    };

    uint32_t schema_version = 0;
    const size_t ver_pos = field_pos(0);
    if (ver_pos != 0 && in_bounds(ver_pos, 4, n)) {
        schema_version = read_u32_le(data + ver_pos);
    }

    auto read_vec_len = [&](size_t field_idx, uint32_t& out_len) -> bool {
        out_len = 0;
        const size_t pos = field_pos(field_idx);
        if (pos == 0) return true;
        if (!in_bounds(pos, 4, n)) return false;
        const uint32_t rel = read_u32_le(data + pos);
        const size_t vec_off = pos + rel;
        if (!in_bounds(vec_off, 4, n)) return false;
        out_len = read_u32_le(data + vec_off);
        return in_bounds(vec_off + 4, static_cast<size_t>(out_len) * 4, n);
    };

    uint32_t num_opcodes = 0;
    uint32_t num_subgraphs = 0;
    uint32_t num_buffers = 0;
    if (!read_vec_len(1, num_opcodes) ||
        !read_vec_len(2, num_subgraphs) ||
        !read_vec_len(4, num_buffers)) {
        return fail_report(report, n, magic, "FlatBuffer root vector bounds check failed");
    }

    out_nnc_bytes = tflite_bytes;
    out_nnc_bytes[4] = 'E';
    out_nnc_bytes[5] = 'N';
    out_nnc_bytes[6] = 'N';
    out_nnc_bytes[7] = 'C';
    uint32_t out_schema_version = schema_version;
    if (ver_pos != 0 && in_bounds(ver_pos, 4, n) && schema_version < 200u) {
        out_schema_version = 211u;
        out_nnc_bytes[ver_pos + 0] = static_cast<uint8_t>(out_schema_version & 0xFFu);
        out_nnc_bytes[ver_pos + 1] = static_cast<uint8_t>((out_schema_version >> 8) & 0xFFu);
        out_nnc_bytes[ver_pos + 2] = static_cast<uint8_t>((out_schema_version >> 16) & 0xFFu);
        out_nnc_bytes[ver_pos + 3] = static_cast<uint8_t>((out_schema_version >> 24) & 0xFFu);
    }

    if (report) {
        std::ostringstream oss;
        oss << magic << "->ENNC (bytes=" << n
            << ", schema_ver=" << schema_version << "->" << out_schema_version
            << ", opcodes=" << num_opcodes
            << ", subgraphs=" << num_subgraphs
            << ", buffers=" << num_buffers << ")";
        report->ok = true;
        report->input_bytes = n;
        report->output_bytes = out_nnc_bytes.size();
        report->original_magic = magic;
        report->output_magic = "ENNC";
        report->summary = oss.str();
        report->error_msg.clear();
    }
    return true;
}

bool convert_tflite_file_to_nnc(const std::string& tflite_path,
                                const std::string& out_nnc_path,
                                NncConvertReport* report) {
    std::ifstream ifs(tflite_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        return fail_report(report, 0, "", "Failed to open input TFLite file: " + tflite_path);
    }

    const std::streamoff end_pos = ifs.tellg();
    if (end_pos < 0) {
        return fail_report(report, 0, "", "Failed to determine input file size: " + tflite_path);
    }

    std::vector<uint8_t> in_bytes(static_cast<size_t>(end_pos));
    ifs.seekg(0, std::ios::beg);
    if (!in_bytes.empty()) {
        ifs.read(reinterpret_cast<char*>(in_bytes.data()),
                 static_cast<std::streamsize>(in_bytes.size()));
        if (!ifs.good()) {
            return fail_report(report, in_bytes.size(), "",
                               "Failed reading input TFLite file: " + tflite_path);
        }
    }

    std::vector<uint8_t> out_bytes;
    if (!convert_tflite_bytes_to_nnc(in_bytes, out_bytes, report)) {
        return false;
    }

    std::ofstream ofs(out_nnc_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        return fail_report(report, in_bytes.size(), report ? report->original_magic : "",
                           "Failed to open output NNC file for writing: " + out_nnc_path);
    }
    if (!out_bytes.empty()) {
        ofs.write(reinterpret_cast<const char*>(out_bytes.data()),
                  static_cast<std::streamsize>(out_bytes.size()));
    }
    if (!ofs.good()) {
        return fail_report(report, in_bytes.size(), report ? report->original_magic : "",
                           "Failed writing output NNC file: " + out_nnc_path);
    }
    return true;
}

} // namespace compiler
} // namespace exynova
