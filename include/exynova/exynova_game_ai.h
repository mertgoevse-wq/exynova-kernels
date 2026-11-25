#pragma once

/**
 * exynova_game_ai.h
 *
 * Header-only C++17 Game Engine AI SDK wrapping ExyNova's stable C11 ABI
 * (`exynova_c_api.h`) and TFLite-to-ENNC converter (`nnc_converter.h`).
 *
 * Designed for drop-in integration into Android NDK game engines:
 *   - Unity Native Plugin (C++ shared library called via C# DllImport)
 *   - Unreal Engine 5 Android NDK Module
 *   - Godot 4 GDExtension
 *   - NativeActivity / GameActivity C++ loops
 */

#include "exynova/exynova_c_api.h"
#include "exynova/nnc_converter.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace exynova {
namespace game_ai {

enum class BatchOpKind : uint32_t {
    Add = 0,
    Mul = 1,
    Relu = 4
};

namespace detail {

inline uint32_t crc32_ieee(const uint8_t* data, size_t size) {
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

class ExynWriter {
public:
    std::vector<uint8_t> buf;

    void u8(uint8_t v) { buf.push_back(v); }

    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
        }
    }

    void i32(int32_t v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        u32(u);
    }

    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
        }
    }

    void i64(int64_t v) {
        uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        u64(u);
    }

    void f32(float v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        u32(u);
    }

    void str(const std::string& s) {
        u32(static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void bytes(const uint8_t* p, size_t len) {
        u32(static_cast<uint32_t>(len));
        if (len > 0 && p) {
            buf.insert(buf.end(), p, p + len);
        }
    }
};

struct TensorSpec {
    uint32_t id = 0;
    std::string name;
    std::vector<int64_t> shape;
    uint32_t dtype = 0;  // FP32 = 0
    uint32_t layout = 2; // NC = 2, ANY = 3
    bool is_constant = false;
    std::vector<uint8_t> const_bytes;
};

struct NodeSpec {
    uint32_t id = 0;
    std::string name;
    uint32_t op = 0; // Add=0, Mul=1, MatMul=2, Relu=4, EnnPrecompiled=10
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::string enn_model_path;
};

inline std::vector<uint8_t> pack_exyn_v1(const std::string& graph_name,
                                         const std::vector<uint32_t>& inputs,
                                         const std::vector<uint32_t>& outputs,
                                         const std::vector<TensorSpec>& tensors,
                                         const std::vector<NodeSpec>& nodes) {
    ExynWriter w;
    w.str(graph_name);
    w.u32(1); // ir_version

    w.u32(static_cast<uint32_t>(inputs.size()));
    for (uint32_t id : inputs) w.u32(id);

    w.u32(static_cast<uint32_t>(outputs.size()));
    for (uint32_t id : outputs) w.u32(id);

    w.u32(static_cast<uint32_t>(tensors.size()));
    for (const auto& t : tensors) {
        w.u32(t.id);
        w.str(t.name);
        w.u32(static_cast<uint32_t>(t.shape.size()));
        for (int64_t d : t.shape) w.i64(d);
        w.u32(t.dtype);
        w.u32(t.layout);
        w.u8(0); // quantized = false
        w.f32(1.0f);
        w.i32(0);
        w.u32(0); // per_channel_scales
        w.u8(t.is_constant ? 1u : 0u);
        w.bytes(t.const_bytes.data(), t.const_bytes.size());
    }

    w.u32(static_cast<uint32_t>(nodes.size()));
    for (const auto& n : nodes) {
        w.u32(n.id);
        w.str(n.name);
        w.u32(n.op);
        w.u32(static_cast<uint32_t>(n.inputs.size()));
        for (uint32_t id : n.inputs) w.u32(id);
        w.u32(static_cast<uint32_t>(n.outputs.size()));
        for (uint32_t id : n.outputs) w.u32(id);
        w.str("");           // fused_activation
        w.i32(1); w.i32(1);  // strides
        w.i32(0); w.i32(0); w.i32(0); w.i32(0); // padding
        w.str(n.enn_model_path);
        w.u32(0); // enn_preset_id
        w.u32(0); // preferred_backend
        w.u64(0); // cached_flops
    }

    // Empty ExecutionPlan (populated by exynova_compile_artifact)
    w.u32(static_cast<uint32_t>(EXYNOVA_MODE_AUTO));
    w.u8(0);
    w.str("");
    w.u32(0);

    // Embedded blobs
    w.u32(0);

    // 36-byte ExynHeader: magic("EXYN"), version(1), target_soc[16], flags(0), payload_size, crc32
    std::vector<uint8_t> out(36 + w.buf.size(), 0);
    out[0] = 'E'; out[1] = 'X'; out[2] = 'Y'; out[3] = 'N';
    const uint32_t fmt_ver = 1;
    std::memcpy(out.data() + 4, &fmt_ver, 4);
    const char soc[] = "S5E8855";
    std::memcpy(out.data() + 8, soc, sizeof(soc));
    const uint32_t payload_sz = static_cast<uint32_t>(w.buf.size());
    const uint32_t crc = crc32_ieee(w.buf.data(), w.buf.size());
    std::memcpy(out.data() + 28, &payload_sz, 4);
    std::memcpy(out.data() + 32, &crc, 4);
    if (!w.buf.empty()) {
        std::memcpy(out.data() + 36, w.buf.data(), w.buf.size());
    }
    return out;
}

inline std::vector<uint8_t> floats_to_bytes(const float* p, size_t count) {
    std::vector<uint8_t> out(count * sizeof(float));
    if (count > 0 && p) {
        std::memcpy(out.data(), p, out.size());
    }
    return out;
}

} // namespace detail

/**
 * RAII Game AI runtime wrapper for per-frame NPC MLP inference,
 * Vulkan compute batching, and TFLite->ENNC NPU model preparation.
 */
class GameAiRuntime {
public:
    GameAiRuntime() = default;
    ~GameAiRuntime() { shutdown(); }

    GameAiRuntime(const GameAiRuntime&) = delete;
    GameAiRuntime& operator=(const GameAiRuntime&) = delete;

    bool init(exynova_execution_mode_t default_mode = EXYNOVA_MODE_AUTO,
              bool strict_mode = false) {
        shutdown();
        default_mode_ = default_mode;
        strict_mode_ = strict_mode;

        const exynova_status_t st = exynova_initialize(&ctx_);
        if (st != EXYNOVA_STATUS_OK || !ctx_) {
            last_error_ = "exynova_initialize failed with status " + std::to_string(static_cast<int>(st));
            ctx_ = nullptr;
            return false;
        }
        last_error_.clear();
        return true;
    }

    void shutdown() {
        if (loaded_nnc_model_) {
            exynova_destroy_model(loaded_nnc_model_);
            loaded_nnc_model_ = nullptr;
        }
        if (ctx_) {
            exynova_shutdown(ctx_);
            ctx_ = nullptr;
        }
    }

    bool is_initialized() const { return ctx_ != nullptr; }

    std::vector<exynova_backend_info_t> backends() const {
        std::vector<exynova_backend_info_t> list;
        if (!ctx_) return list;
        uint32_t count = 0;
        if (exynova_list_backends(ctx_, nullptr, 0, &count) != EXYNOVA_STATUS_OK || count == 0) {
            return list;
        }
        list.resize(count);
        uint32_t actual = 0;
        if (exynova_list_backends(ctx_, list.data(), count, &actual) != EXYNOVA_STATUS_OK) {
            return {};
        }
        list.resize(actual);
        return list;
    }

    std::string backend_summary() const {
        if (!ctx_) {
            return "ExyNova GameAiRuntime: not initialized";
        }
        std::ostringstream oss;
        oss << "ExyNova SDK " << exynova_version_string() << " | Backends:";
        const auto list = backends();
        for (const auto& b : list) {
            oss << "\n  - " << b.name << " [status=" << static_cast<int>(b.status) << "]";
            if (b.device_name[0] != '\0') {
                oss << " device=" << b.device_name;
            }
            if (b.details[0] != '\0') {
                oss << " (" << b.details << ")";
            }
        }
        return oss.str();
    }

    /**
     * Evaluates a 2-layer NPC decision MLP:
     *   MatMul(input, w1) -> Add(b1) -> Relu -> MatMul(w2) -> out_logits
     */
    bool dispatch_npc_mlp(const float* input,
                          const float* w1,
                          const float* b1,
                          const float* w2,
                          size_t batch,
                          size_t in_dim,
                          size_t hid_dim,
                          size_t out_dim,
                          std::vector<float>& out_logits,
                          exynova_execution_mode_t mode = EXYNOVA_MODE_AUTO) {
        if (!ctx_) {
            last_error_ = "GameAiRuntime is not initialized";
            return false;
        }
        if (!input || !w1 || !b1 || !w2 || batch == 0 || in_dim == 0 || hid_dim == 0 || out_dim == 0) {
            last_error_ = "Invalid MLP dimensions or null pointers";
            return false;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const int64_t B = static_cast<int64_t>(batch);
        const int64_t I = static_cast<int64_t>(in_dim);
        const int64_t H = static_cast<int64_t>(hid_dim);
        const int64_t O = static_cast<int64_t>(out_dim);

        std::vector<detail::TensorSpec> tensors = {
            {1, "npc_in", {B, I}, 0, 2, false, {}},
            {2, "w1", {I, H}, 0, 2, true, detail::floats_to_bytes(w1, in_dim * hid_dim)},
            {3, "b1", {1, H}, 0, 2, true, detail::floats_to_bytes(b1, hid_dim)},
            {4, "w2", {H, O}, 0, 2, true, detail::floats_to_bytes(w2, hid_dim * out_dim)},
            {5, "mm1", {B, H}, 0, 2, false, {}},
            {6, "add1", {B, H}, 0, 2, false, {}},
            {7, "relu1", {B, H}, 0, 2, false, {}},
            {8, "npc_out", {B, O}, 0, 2, false, {}},
        };
        std::vector<detail::NodeSpec> nodes = {
            {1, "fc1", 2, {1, 2}, {5}, ""},
            {2, "bias1", 0, {5, 3}, {6}, ""},
            {3, "act1", 4, {6}, {7}, ""},
            {4, "fc2", 2, {7, 4}, {8}, ""},
        };

        const auto exyn = detail::pack_exyn_v1("game_npc_mlp", {1}, {8}, tensors, nodes);
        exynova_compiled_model_t model = nullptr;
        exynova_session_t session = nullptr;

        auto cleanup = [&]() {
            if (session) exynova_destroy_session(session);
            if (model) exynova_destroy_model(model);
        };

        const exynova_execution_mode_t eff_mode =
            (mode == EXYNOVA_MODE_AUTO) ? default_mode_ : mode;
        exynova_status_t st = exynova_compile_artifact(
            ctx_, exyn.data(), exyn.size(), eff_mode, strict_mode_ ? 1 : 0, &model);
        if (st != EXYNOVA_STATUS_OK || !model) {
            last_error_ = "MLP compile failed (status=" + std::to_string(static_cast<int>(st)) + ")";
            cleanup();
            return false;
        }

        if (exynova_create_session(model, &session) != EXYNOVA_STATUS_OK || !session) {
            last_error_ = "exynova_create_session failed";
            cleanup();
            return false;
        }

        const size_t in_bytes = batch * in_dim * sizeof(float);
        if (exynova_set_input(session, "npc_in", input, in_bytes) != EXYNOVA_STATUS_OK ||
            exynova_run(session) != EXYNOVA_STATUS_OK) {
            exynova_exec_report_t rep{};
            exynova_get_last_report(session, &rep);
            last_error_ = rep.error_msg[0] ? rep.error_msg : "MLP execution failed";
            cleanup();
            return false;
        }

        out_logits.resize(batch * out_dim);
        if (exynova_get_output(session, "npc_out", out_logits.data(),
                               out_logits.size() * sizeof(float)) != EXYNOVA_STATUS_OK) {
            last_error_ = "exynova_get_output failed";
            cleanup();
            return false;
        }

        exynova_exec_report_t rep{};
        if (exynova_get_last_report(session, &rep) == EXYNOVA_STATUS_OK && rep.total_us > 0.0) {
            last_frame_cost_ms_ = rep.total_us / 1000.0;
        } else {
            const auto t1 = std::chrono::steady_clock::now();
            last_frame_cost_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        cleanup();
        last_error_.clear();
        return true;
    }

    /**
     * Dispatches a vectorized batch op (Add / Mul / Relu) over float arrays,
     * targeting GPU_ONLY when requested or AUTO when GPU is unavailable (host).
     */
    bool run_vulkan_compute_batch(const float* a,
                                  const float* b,
                                  float* out,
                                  size_t count,
                                  BatchOpKind op = BatchOpKind::Add,
                                  bool require_gpu = false) {
        if (!ctx_) {
            last_error_ = "GameAiRuntime is not initialized";
            return false;
        }
        if (!a || !out || count == 0 || (op != BatchOpKind::Relu && !b)) {
            last_error_ = "Invalid batch compute buffers";
            return false;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const int64_t N = static_cast<int64_t>(count);

        std::vector<detail::TensorSpec> tensors;
        std::vector<detail::NodeSpec> nodes;
        std::vector<uint32_t> graph_inputs;

        tensors.push_back({1, "a", {1, N}, 0, 2, false, {}});
        graph_inputs.push_back(1);
        if (op != BatchOpKind::Relu) {
            tensors.push_back({2, "b", {1, N}, 0, 2, false, {}});
            tensors.push_back({3, "out", {1, N}, 0, 2, false, {}});
            nodes.push_back({1, "batch_op", static_cast<uint32_t>(op), {1, 2}, {3}, ""});
            graph_inputs.push_back(2);
        } else {
            tensors.push_back({3, "out", {1, N}, 0, 2, false, {}});
            nodes.push_back({1, "batch_op", static_cast<uint32_t>(op), {1}, {3}, ""});
        }

        const auto exyn = detail::pack_exyn_v1("vulkan_batch", graph_inputs, {3}, tensors, nodes);
        exynova_compiled_model_t model = nullptr;
        exynova_session_t session = nullptr;

        auto cleanup = [&]() {
            if (session) exynova_destroy_session(session);
            if (model) exynova_destroy_model(model);
        };

        const exynova_execution_mode_t mode = require_gpu ? EXYNOVA_MODE_GPU_ONLY : EXYNOVA_MODE_AUTO;
        const exynova_status_t st = exynova_compile_artifact(
            ctx_, exyn.data(), exyn.size(), mode, require_gpu ? 1 : 0, &model);
        if (st != EXYNOVA_STATUS_OK || !model) {
            last_error_ = "Batch compile failed (status=" + std::to_string(static_cast<int>(st)) + ")";
            cleanup();
            return false;
        }

        if (exynova_create_session(model, &session) != EXYNOVA_STATUS_OK || !session) {
            last_error_ = "exynova_create_session failed";
            cleanup();
            return false;
        }

        const size_t bytes = count * sizeof(float);
        if (exynova_set_input(session, "a", a, bytes) != EXYNOVA_STATUS_OK) {
            last_error_ = "exynova_set_input(a) failed";
            cleanup();
            return false;
        }
        if (op != BatchOpKind::Relu &&
            exynova_set_input(session, "b", b, bytes) != EXYNOVA_STATUS_OK) {
            last_error_ = "exynova_set_input(b) failed";
            cleanup();
            return false;
        }
        if (exynova_run(session) != EXYNOVA_STATUS_OK ||
            exynova_get_output(session, "out", out, bytes) != EXYNOVA_STATUS_OK) {
            last_error_ = "Batch execution/readback failed";
            cleanup();
            return false;
        }

        const auto t1 = std::chrono::steady_clock::now();
        last_frame_cost_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cleanup();
        last_error_.clear();
        return true;
    }

    /**
     * Converts a .tflite ("TFL3") model into an ENN ("ENNC") .nnc file and binds it
     * as an EnnPrecompiled node in an ExyNova runtime artifact.
     * Note: Actual NPU execution requires an Android app context with ENN NPU driver
     * access and an offline-compiled NCP v26 payload inside the .nnc container.
     */
    bool convert_and_load_nnc(const std::string& tflite_path,
                              const std::string& out_nnc_path,
                              compiler::NncConvertReport* out_convert_report = nullptr,
                              bool strict_npu = false) {
        if (!ctx_) {
            last_error_ = "GameAiRuntime is not initialized";
            return false;
        }

        compiler::NncConvertReport rep;
        if (!compiler::convert_tflite_file_to_nnc(tflite_path, out_nnc_path, &rep)) {
            last_error_ = rep.error_msg;
            if (out_convert_report) *out_convert_report = rep;
            return false;
        }
        if (out_convert_report) *out_convert_report = rep;

        std::vector<detail::TensorSpec> tensors = {
            {1, "nnc_in", {1, 4}, 0, 2, false, {}},
            {2, "nnc_out", {1, 4}, 0, 2, false, {}},
        };
        std::vector<detail::NodeSpec> nodes = {
            {1, "enn_subgraph", 10 /* OpType::EnnPrecompiled */, {1}, {2}, out_nnc_path},
        };

        const auto exyn = detail::pack_exyn_v1("enn_nnc_wrapper", {1}, {2}, tensors, nodes);
        if (loaded_nnc_model_) {
            exynova_destroy_model(loaded_nnc_model_);
            loaded_nnc_model_ = nullptr;
        }

        const exynova_execution_mode_t mode = strict_npu ? EXYNOVA_MODE_NPU_ONLY : EXYNOVA_MODE_AUTO;
        const exynova_status_t st = exynova_compile_artifact(
            ctx_, exyn.data(), exyn.size(), mode, strict_npu ? 1 : 0, &loaded_nnc_model_);
        if (st != EXYNOVA_STATUS_OK || !loaded_nnc_model_) {
            last_error_ = "NNC model converted (" + rep.summary +
                          ") but runtime NPU backend is unavailable/rejected (status=" +
                          std::to_string(static_cast<int>(st)) + ")";
            return false;
        }
        last_error_.clear();
        return true;
    }

    double last_frame_cost_ms() const { return last_frame_cost_ms_; }
    const std::string& last_error() const { return last_error_; }

private:
    exynova_context_t ctx_ = nullptr;
    exynova_compiled_model_t loaded_nnc_model_ = nullptr;
    exynova_execution_mode_t default_mode_ = EXYNOVA_MODE_AUTO;
    bool strict_mode_ = false;
    double last_frame_cost_ms_ = 0.0;
    std::string last_error_;
};

} // namespace game_ai
} // namespace exynova
