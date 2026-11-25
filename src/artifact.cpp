#include "exynova/artifact.h"

#include <cstring>
#include <fstream>
#include <limits>

namespace exynova {
namespace compiler {

uint32_t compute_crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = (crc & 1u) ? 0xEDB88320u : 0u;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

namespace {

class ByteWriter {
public:
    std::vector<uint8_t> buf;

    void write_u8(uint8_t v) { buf.push_back(v); }

    void write_u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
        }
    }

    void write_i32(int32_t v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        write_u32(u);
    }

    void write_u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
        }
    }

    void write_i64(int64_t v) {
        uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        write_u64(u);
    }

    void write_f32(float v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        write_u32(u);
    }

    void write_str(const std::string& s) {
        write_u32(static_cast<uint32_t>(s.size()));
        if (!s.empty()) {
            buf.insert(buf.end(), s.begin(), s.end());
        }
    }

    void write_bytes(const std::vector<uint8_t>& bytes) {
        write_u32(static_cast<uint32_t>(bytes.size()));
        if (!bytes.empty()) {
            buf.insert(buf.end(), bytes.begin(), bytes.end());
        }
    }
};

class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    bool has(size_t n) const { return n <= (size_ - pos_); }
    size_t remaining() const { return size_ - pos_; }

    bool read_u8(uint8_t& out) {
        if (!has(1)) return false;
        out = data_[pos_++];
        return true;
    }

    bool read_u32(uint32_t& out) {
        if (!has(4)) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= (static_cast<uint32_t>(data_[pos_++]) << (i * 8));
        }
        out = v;
        return true;
    }

    bool read_i32(int32_t& out) {
        uint32_t u = 0;
        if (!read_u32(u)) return false;
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }

    bool read_u64(uint64_t& out) {
        if (!has(8)) return false;
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= (static_cast<uint64_t>(data_[pos_++]) << (i * 8));
        }
        out = v;
        return true;
    }

    bool read_i64(int64_t& out) {
        uint64_t u = 0;
        if (!read_u64(u)) return false;
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }

    bool read_f32(float& out) {
        uint32_t u = 0;
        if (!read_u32(u)) return false;
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }

    bool read_str(std::string& out) {
        uint32_t len = 0;
        if (!read_u32(len) || !has(len)) return false;
        out.assign(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return true;
    }

    bool read_bytes(std::vector<uint8_t>& out) {
        uint32_t len = 0;
        if (!read_u32(len) || !has(len)) return false;
        out.assign(data_ + pos_, data_ + pos_ + len);
        pos_ += len;
        return true;
    }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
};

} // namespace

class ArtifactSerializer {
public:
    static void sync(Graph& g) { g.sync_id_counters(); }
};

bool serialize(const Artifact& artifact,
               std::vector<uint8_t>& out_bytes,
               std::string* err) {
    ByteWriter w;

    // Graph
    w.write_str(artifact.graph.name);
    w.write_u32(artifact.graph.ir_version);

    w.write_u32(static_cast<uint32_t>(artifact.graph.inputs.size()));
    for (uint32_t tid : artifact.graph.inputs) {
        w.write_u32(tid);
    }
    w.write_u32(static_cast<uint32_t>(artifact.graph.outputs.size()));
    for (uint32_t tid : artifact.graph.outputs) {
        w.write_u32(tid);
    }

    w.write_u32(static_cast<uint32_t>(artifact.graph.tensors.size()));
    for (const auto& t : artifact.graph.tensors) {
        w.write_u32(t.id);
        w.write_str(t.name);
        w.write_u32(static_cast<uint32_t>(t.shape.size()));
        for (int64_t d : t.shape) {
            w.write_i64(d);
        }
        w.write_u32(static_cast<uint32_t>(t.dtype));
        w.write_u32(static_cast<uint32_t>(t.layout));
        w.write_u8(t.quant.quantized ? 1u : 0u);
        w.write_f32(t.quant.scale);
        w.write_i32(t.quant.zero_point);
        w.write_u32(static_cast<uint32_t>(t.quant.per_channel_scales.size()));
        for (float s : t.quant.per_channel_scales) {
            w.write_f32(s);
        }
        w.write_u8(t.is_constant ? 1u : 0u);
        w.write_bytes(t.constant_data);
    }

    w.write_u32(static_cast<uint32_t>(artifact.graph.nodes.size()));
    for (const auto& n : artifact.graph.nodes) {
        w.write_u32(n.id);
        w.write_str(n.name);
        w.write_u32(static_cast<uint32_t>(n.op));
        w.write_u32(static_cast<uint32_t>(n.inputs.size()));
        for (uint32_t tid : n.inputs) {
            w.write_u32(tid);
        }
        w.write_u32(static_cast<uint32_t>(n.outputs.size()));
        for (uint32_t tid : n.outputs) {
            w.write_u32(tid);
        }
        w.write_str(n.attrs.fused_activation);
        w.write_i32(n.attrs.strides[0]);
        w.write_i32(n.attrs.strides[1]);
        for (int i = 0; i < 4; ++i) {
            w.write_i32(n.attrs.padding[static_cast<size_t>(i)]);
        }
        w.write_str(n.attrs.enn_model_path);
        w.write_u32(n.attrs.enn_preset_id);
        w.write_u32(static_cast<uint32_t>(n.preferred_backend));
        w.write_u64(n.cached_flops);
    }

    // ExecutionPlan
    w.write_u32(static_cast<uint32_t>(artifact.plan.mode));
    w.write_u8(artifact.plan.strict_mode ? 1u : 0u);
    w.write_str(artifact.plan.error_msg);
    w.write_u32(static_cast<uint32_t>(artifact.plan.segments.size()));
    for (const auto& seg : artifact.plan.segments) {
        w.write_u32(seg.segment_id);
        w.write_u32(static_cast<uint32_t>(seg.backend));
        w.write_u32(static_cast<uint32_t>(seg.node_ids.size()));
        for (uint32_t nid : seg.node_ids) {
            w.write_u32(nid);
        }
        w.write_u8(seg.is_fallback ? 1u : 0u);
        w.write_str(seg.reason);
        w.write_u64(seg.estimated_flops);
    }

    // Embedded blobs
    w.write_u32(static_cast<uint32_t>(artifact.blobs.size()));
    for (const auto& b : artifact.blobs) {
        w.write_str(b.tag);
        w.write_bytes(b.data);
    }

    if (w.buf.size() > std::numeric_limits<uint32_t>::max()) {
        if (err) {
            *err = "Artifact payload exceeds 4 GiB limit";
        }
        return false;
    }

    ExynHeader header;
    header.magic = EXYN_MAGIC;
    header.format_version = EXYN_FORMAT_VERSION;
    std::memset(header.target_soc, 0, sizeof(header.target_soc));
    const size_t soc_len = std::min(artifact.target_soc.size(), sizeof(header.target_soc) - 1);
    if (soc_len > 0) {
        std::memcpy(header.target_soc, artifact.target_soc.data(), soc_len);
    }
    header.flags = artifact.flags;
    header.payload_size = static_cast<uint32_t>(w.buf.size());
    header.crc32 = compute_crc32(w.buf.data(), w.buf.size());

    out_bytes.resize(sizeof(ExynHeader) + w.buf.size());
    std::memcpy(out_bytes.data(), &header, sizeof(ExynHeader));
    if (!w.buf.empty()) {
        std::memcpy(out_bytes.data() + sizeof(ExynHeader), w.buf.data(), w.buf.size());
    }

    if (err) {
        err->clear();
    }
    return true;
}

bool deserialize(const uint8_t* data,
                 size_t size,
                 Artifact& out_artifact,
                 std::string* err) {
    auto fail = [&](const std::string& msg) {
        if (err) {
            *err = msg;
        }
        return false;
    };

    if (!data || size < sizeof(ExynHeader)) {
        return fail("Truncated .exyn artifact: smaller than header");
    }

    ExynHeader header;
    std::memcpy(&header, data, sizeof(ExynHeader));

    if (header.magic != EXYN_MAGIC) {
        return fail("Invalid .exyn magic header");
    }
    if (header.format_version != EXYN_FORMAT_VERSION) {
        return fail("Unsupported .exyn format version: " +
                    std::to_string(header.format_version));
    }

    const size_t expected_total = sizeof(ExynHeader) + static_cast<size_t>(header.payload_size);
    if (size < expected_total) {
        return fail("Truncated .exyn payload");
    }
    if (size != expected_total) {
        return fail("Corrupt .exyn artifact: payload size mismatch");
    }

    const uint8_t* payload = data + sizeof(ExynHeader);
    const uint32_t actual_crc = compute_crc32(payload, header.payload_size);
    if (actual_crc != header.crc32) {
        return fail("CRC32 mismatch on .exyn artifact payload");
    }

    Artifact parsed;
    header.target_soc[sizeof(header.target_soc) - 1] = '\0';
    parsed.target_soc = header.target_soc;
    parsed.flags = header.flags;

    ByteReader r(payload, header.payload_size);

    if (!r.read_str(parsed.graph.name) || !r.read_u32(parsed.graph.ir_version)) {
        return fail("Malformed graph header in .exyn payload");
    }

    uint32_t count = 0;
    if (!r.read_u32(count) || count > r.remaining() / 4) {
        return fail("Malformed graph inputs in .exyn payload");
    }
    parsed.graph.inputs.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!r.read_u32(parsed.graph.inputs[i])) return fail("Truncated graph inputs");
    }

    if (!r.read_u32(count) || count > r.remaining() / 4) {
        return fail("Malformed graph outputs in .exyn payload");
    }
    parsed.graph.outputs.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!r.read_u32(parsed.graph.outputs[i])) return fail("Truncated graph outputs");
    }

    uint32_t num_tensors = 0;
    if (!r.read_u32(num_tensors) || num_tensors > r.remaining()) {
        return fail("Malformed tensor table in .exyn payload");
    }
    parsed.graph.tensors.reserve(num_tensors);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        TensorDesc t;
        uint32_t rank = 0;
        uint32_t dtype_u = 0;
        uint32_t layout_u = 0;
        uint8_t q_flag = 0;
        uint32_t scale_count = 0;
        uint8_t const_flag = 0;

        if (!r.read_u32(t.id) || !r.read_str(t.name) ||
            !r.read_u32(rank) || rank > r.remaining() / 8) {
            return fail("Malformed tensor entry in .exyn payload");
        }
        t.shape.resize(rank);
        for (uint32_t d = 0; d < rank; ++d) {
            if (!r.read_i64(t.shape[d])) return fail("Truncated tensor shape");
        }
        if (!r.read_u32(dtype_u) || dtype_u > static_cast<uint32_t>(DType::UINT8) ||
            !r.read_u32(layout_u) || layout_u > static_cast<uint32_t>(Layout::ANY) ||
            !r.read_u8(q_flag) || !r.read_f32(t.quant.scale) ||
            !r.read_i32(t.quant.zero_point) ||
            !r.read_u32(scale_count) || scale_count > r.remaining() / 4) {
            return fail("Malformed tensor metadata in .exyn payload");
        }
        t.dtype = static_cast<DType>(dtype_u);
        t.layout = static_cast<Layout>(layout_u);
        t.quant.quantized = (q_flag != 0);
        t.quant.per_channel_scales.resize(scale_count);
        for (uint32_t s = 0; s < scale_count; ++s) {
            if (!r.read_f32(t.quant.per_channel_scales[s])) {
                return fail("Truncated per-channel scales");
            }
        }
        if (!r.read_u8(const_flag) || !r.read_bytes(t.constant_data)) {
            return fail("Malformed tensor constant data");
        }
        t.is_constant = (const_flag != 0);
        parsed.graph.tensors.push_back(std::move(t));
    }

    uint32_t num_nodes = 0;
    if (!r.read_u32(num_nodes) || num_nodes > r.remaining()) {
        return fail("Malformed node table in .exyn payload");
    }
    parsed.graph.nodes.reserve(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        Node n;
        uint32_t op_u = 0;
        uint32_t in_cnt = 0;
        uint32_t out_cnt = 0;
        uint32_t backend_u = 0;

        if (!r.read_u32(n.id) || !r.read_str(n.name) ||
            !r.read_u32(op_u) || op_u > static_cast<uint32_t>(OpType::EnnPrecompiled) ||
            !r.read_u32(in_cnt) || in_cnt > r.remaining() / 4) {
            return fail("Malformed node entry in .exyn payload");
        }
        n.op = static_cast<OpType>(op_u);
        n.inputs.resize(in_cnt);
        for (uint32_t k = 0; k < in_cnt; ++k) {
            if (!r.read_u32(n.inputs[k])) return fail("Truncated node inputs");
        }
        if (!r.read_u32(out_cnt) || out_cnt > r.remaining() / 4) {
            return fail("Malformed node outputs");
        }
        n.outputs.resize(out_cnt);
        for (uint32_t k = 0; k < out_cnt; ++k) {
            if (!r.read_u32(n.outputs[k])) return fail("Truncated node outputs");
        }
        if (!r.read_str(n.attrs.fused_activation) ||
            !r.read_i32(n.attrs.strides[0]) || !r.read_i32(n.attrs.strides[1])) {
            return fail("Malformed node strides");
        }
        for (int k = 0; k < 4; ++k) {
            if (!r.read_i32(n.attrs.padding[static_cast<size_t>(k)])) {
                return fail("Malformed node padding");
            }
        }
        if (!r.read_str(n.attrs.enn_model_path) ||
            !r.read_u32(n.attrs.enn_preset_id) ||
            !r.read_u32(backend_u) ||
            backend_u > static_cast<uint32_t>(runtime::BackendKind::NPU_EXPERIMENTAL) ||
            !r.read_u64(n.cached_flops)) {
            return fail("Malformed node attributes");
        }
        n.preferred_backend = static_cast<runtime::BackendKind>(backend_u);
        parsed.graph.nodes.push_back(std::move(n));
    }

    uint32_t mode_u = 0;
    uint8_t strict_u = 0;
    uint32_t num_segs = 0;
    if (!r.read_u32(mode_u) ||
        mode_u > static_cast<uint32_t>(runtime::ExecutionMode::HYBRID) ||
        !r.read_u8(strict_u) || !r.read_str(parsed.plan.error_msg) ||
        !r.read_u32(num_segs) || num_segs > r.remaining()) {
        return fail("Malformed execution plan in .exyn payload");
    }
    parsed.plan.mode = static_cast<runtime::ExecutionMode>(mode_u);
    parsed.plan.strict_mode = (strict_u != 0);
    parsed.plan.segments.reserve(num_segs);
    for (uint32_t i = 0; i < num_segs; ++i) {
        PartitionedSegment seg;
        uint32_t backend_u = 0;
        uint32_t nid_cnt = 0;
        uint8_t fb_u = 0;
        if (!r.read_u32(seg.segment_id) ||
            !r.read_u32(backend_u) ||
            backend_u > static_cast<uint32_t>(runtime::BackendKind::NPU_EXPERIMENTAL) ||
            !r.read_u32(nid_cnt) || nid_cnt > r.remaining() / 4) {
            return fail("Malformed segment in .exyn payload");
        }
        seg.backend = static_cast<runtime::BackendKind>(backend_u);
        seg.node_ids.resize(nid_cnt);
        for (uint32_t k = 0; k < nid_cnt; ++k) {
            if (!r.read_u32(seg.node_ids[k])) return fail("Truncated segment node_ids");
        }
        if (!r.read_u8(fb_u) || !r.read_str(seg.reason) || !r.read_u64(seg.estimated_flops)) {
            return fail("Malformed segment metadata");
        }
        seg.is_fallback = (fb_u != 0);
        parsed.plan.segments.push_back(std::move(seg));
    }

    uint32_t num_blobs = 0;
    if (!r.read_u32(num_blobs) || num_blobs > r.remaining()) {
        return fail("Malformed blob table in .exyn payload");
    }
    parsed.blobs.reserve(num_blobs);
    for (uint32_t i = 0; i < num_blobs; ++i) {
        EmbeddedBlob b;
        if (!r.read_str(b.tag) || !r.read_bytes(b.data)) {
            return fail("Malformed embedded blob");
        }
        parsed.blobs.push_back(std::move(b));
    }

    if (r.remaining() != 0) {
        return fail("Trailing unread bytes in .exyn payload");
    }

    ArtifactSerializer::sync(parsed.graph);
    out_artifact = std::move(parsed);
    if (err) {
        err->clear();
    }
    return true;
}

bool save_file(const std::string& path,
               const Artifact& artifact,
               std::string* err) {
    std::vector<uint8_t> bytes;
    if (!serialize(artifact, bytes, err)) {
        return false;
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        if (err) {
            *err = "Failed to open artifact file for writing: " + path;
        }
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!ofs.good()) {
        if (err) {
            *err = "Failed writing artifact file: " + path;
        }
        return false;
    }
    return true;
}

bool load_file(const std::string& path,
               Artifact& out_artifact,
               std::string* err) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        if (err) {
            *err = "Failed to open artifact file: " + path;
        }
        return false;
    }
    const std::streamoff end_pos = ifs.tellg();
    if (end_pos < 0) {
        if (err) {
            *err = "Failed to determine artifact file size: " + path;
        }
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(end_pos));
    ifs.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        ifs.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
        if (!ifs.good()) {
            if (err) {
                *err = "Failed reading artifact file: " + path;
            }
            return false;
        }
    }
    return deserialize(bytes.data(), bytes.size(), out_artifact, err);
}

} // namespace compiler
} // namespace exynova
