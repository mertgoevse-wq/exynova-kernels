#include "exynova/exynova.h"
#include "exynova/exynova_c_api.h"
#include "exynova/cpu_ref.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <new>

namespace exynova {
namespace sdk {

const char* version_string() {
    return "0.1.0-exynova-sdk";
}

// ============================================================================
// CPU reference kernel implementations
// ============================================================================

namespace {

std::vector<size_t> compute_strides(const std::vector<int64_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * static_cast<size_t>(shape[i - 1]);
    }
    return strides;
}

bool cpu_exec_binary(compiler::OpType op,
                     const compiler::TensorDesc& a_desc,
                     const compiler::TensorDesc& b_desc,
                     const compiler::TensorDesc& out_desc,
                     const float* a,
                     const float* b,
                     float* out) {
    const size_t total = static_cast<size_t>(out_desc.element_count());
    if (total == 0) {
        return true;
    }
    if (a_desc.shape == b_desc.shape && a_desc.shape == out_desc.shape) {
        if (op == compiler::OpType::Add) {
            runtime::cpu_ref::compute<runtime::cpu_ref::OpAdd>(a, b, out, total);
        } else {
            runtime::cpu_ref::compute<runtime::cpu_ref::OpMul>(a, b, out, total);
        }
        return true;
    }

    const size_t nd = out_desc.shape.size();
    std::vector<size_t> a_pad(nd, 1);
    std::vector<size_t> b_pad(nd, 1);
    for (size_t i = 0; i < a_desc.shape.size(); ++i) {
        a_pad[nd - a_desc.shape.size() + i] = static_cast<size_t>(a_desc.shape[i]);
    }
    for (size_t i = 0; i < b_desc.shape.size(); ++i) {
        b_pad[nd - b_desc.shape.size() + i] = static_cast<size_t>(b_desc.shape[i]);
    }

    std::vector<size_t> a_str(nd, 1);
    std::vector<size_t> b_str(nd, 1);
    for (size_t i = nd; i > 1; --i) {
        a_str[i - 2] = a_str[i - 1] * a_pad[i - 1];
        b_str[i - 2] = b_str[i - 1] * b_pad[i - 1];
    }

    std::vector<size_t> out_str = compute_strides(out_desc.shape);
    for (size_t idx = 0; idx < total; ++idx) {
        size_t rem = idx;
        size_t ia = 0;
        size_t ib = 0;
        for (size_t d = 0; d < nd; ++d) {
            const size_t coord = rem / out_str[d];
            rem %= out_str[d];
            if (a_pad[d] > 1) {
                ia += coord * a_str[d];
            }
            if (b_pad[d] > 1) {
                ib += coord * b_str[d];
            }
        }
        out[idx] = (op == compiler::OpType::Add) ? (a[ia] + b[ib]) : (a[ia] * b[ib]);
    }
    return true;
}

bool cpu_exec_matmul(const compiler::Node& node,
                     const compiler::TensorDesc& a_desc,
                     const compiler::TensorDesc& b_desc,
                     const compiler::TensorDesc& out_desc,
                     const float* a,
                     const float* b,
                     float* out) {
    const size_t ra = a_desc.shape.size();
    const size_t rb = b_desc.shape.size();
    if (ra < 2 || rb < 2) {
        return false;
    }
    const size_t m = static_cast<size_t>(a_desc.shape[ra - 2]);
    const size_t k = static_cast<size_t>(a_desc.shape[ra - 1]);
    const size_t n = static_cast<size_t>(b_desc.shape[rb - 1]);

    size_t batches = 1;
    if (out_desc.shape.size() > 2) {
        for (size_t i = 0; i + 2 < out_desc.shape.size(); ++i) {
            batches *= static_cast<size_t>(out_desc.shape[i]);
        }
    }
    const size_t a_batch_elems = static_cast<size_t>(a_desc.element_count()) / (m * k);
    const size_t b_batch_elems = static_cast<size_t>(b_desc.element_count()) / (k * n);
    const bool fuse_relu = (node.attrs.fused_activation == "Relu");

    for (size_t batch = 0; batch < batches; ++batch) {
        const size_t ba = (a_batch_elems == 1) ? 0 : (batch % a_batch_elems);
        const size_t bb = (b_batch_elems == 1) ? 0 : (batch % b_batch_elems);
        const float* a_ptr = a + ba * m * k;
        const float* b_ptr = b + bb * k * n;
        float* c_ptr = out + batch * m * n;

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    acc += static_cast<double>(a_ptr[i * k + p]) *
                           static_cast<double>(b_ptr[p * n + j]);
                }
                float val = static_cast<float>(acc);
                if (fuse_relu && val < 0.0f) {
                    val = 0.0f;
                }
                c_ptr[i * n + j] = val;
            }
        }
    }
    return true;
}

bool cpu_exec_conv2d(const compiler::Node& node,
                     const compiler::TensorDesc& x_desc,
                     const compiler::TensorDesc& w_desc,
                     const compiler::TensorDesc& out_desc,
                     const float* x,
                     const float* w,
                     const float* bias,
                     float* out) {
    if (x_desc.shape.size() != 4 || w_desc.shape.size() != 4 || out_desc.shape.size() != 4) {
        return false;
    }
    const bool is_nchw = (x_desc.layout == compiler::Layout::NCHW);
    const int64_t sh = std::max(1, node.attrs.strides[0]);
    const int64_t sw = std::max(1, node.attrs.strides[1]);
    const int64_t pt = node.attrs.padding[0];
    const int64_t pl = node.attrs.padding[2];
    const bool fuse_relu = (node.attrs.fused_activation == "Relu");

    const int64_t n_batch = x_desc.shape[0];
    const int64_t c_in = is_nchw ? x_desc.shape[1] : x_desc.shape[3];
    const int64_t h_in = is_nchw ? x_desc.shape[2] : x_desc.shape[1];
    const int64_t w_in = is_nchw ? x_desc.shape[3] : x_desc.shape[2];

    const int64_t c_out = w_desc.shape[0];
    const int64_t kh = is_nchw ? w_desc.shape[2] : w_desc.shape[1];
    const int64_t kw = is_nchw ? w_desc.shape[3] : w_desc.shape[2];

    const int64_t h_out = is_nchw ? out_desc.shape[2] : out_desc.shape[1];
    const int64_t w_out = is_nchw ? out_desc.shape[3] : out_desc.shape[2];

    for (int64_t n = 0; n < n_batch; ++n) {
        for (int64_t oc = 0; oc < c_out; ++oc) {
            for (int64_t oh = 0; oh < h_out; ++oh) {
                for (int64_t ow = 0; ow < w_out; ++ow) {
                    double acc = bias ? static_cast<double>(bias[oc]) : 0.0;
                    for (int64_t ic = 0; ic < c_in; ++ic) {
                        for (int64_t fh = 0; fh < kh; ++fh) {
                            const int64_t ih = oh * sh - pt + fh;
                            if (ih < 0 || ih >= h_in) continue;
                            for (int64_t fw = 0; fw < kw; ++fw) {
                                const int64_t iw = ow * sw - pl + fw;
                                if (iw < 0 || iw >= w_in) continue;
                                const int64_t x_idx = is_nchw
                                    ? (((n * c_in + ic) * h_in + ih) * w_in + iw)
                                    : (((n * h_in + ih) * w_in + iw) * c_in + ic);
                                const int64_t w_idx = is_nchw
                                    ? (((oc * c_in + ic) * kh + fh) * kw + fw)
                                    : (((oc * kh + fh) * kw + fw) * c_in + ic);
                                acc += static_cast<double>(x[x_idx]) *
                                       static_cast<double>(w[w_idx]);
                            }
                        }
                    }
                    float val = static_cast<float>(acc);
                    if (fuse_relu && val < 0.0f) {
                        val = 0.0f;
                    }
                    const int64_t out_idx = is_nchw
                        ? (((n * c_out + oc) * h_out + oh) * w_out + ow)
                        : (((n * h_out + oh) * w_out + ow) * c_out + oc);
                    out[out_idx] = val;
                }
            }
        }
    }
    return true;
}

bool cpu_execute_node(const compiler::Node& node,
                      const compiler::Graph& graph,
                      const std::vector<const void*>& inputs,
                      const std::vector<void*>& outputs,
                      std::string* err) {
    auto fail = [&](const std::string& msg) {
        if (err) {
            *err = msg;
        }
        return false;
    };

    switch (node.op) {
    case compiler::OpType::Add:
    case compiler::OpType::Mul: {
        if (inputs.size() < 2 || outputs.empty()) {
            return fail("Binary op missing buffers");
        }
        const auto* a_desc = graph.get_tensor(node.inputs[0]);
        const auto* b_desc = graph.get_tensor(node.inputs[1]);
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!a_desc || !b_desc || !out_desc) return fail("Missing tensor desc");
        if (a_desc->dtype != compiler::DType::FP32 ||
            b_desc->dtype != compiler::DType::FP32 ||
            out_desc->dtype != compiler::DType::FP32) {
            return fail("CPU reference Add/Mul requires FP32");
        }
        return cpu_exec_binary(node.op, *a_desc, *b_desc, *out_desc,
                               static_cast<const float*>(inputs[0]),
                               static_cast<const float*>(inputs[1]),
                               static_cast<float*>(outputs[0]));
    }
    case compiler::OpType::MatMul: {
        if (inputs.size() < 2 || outputs.empty()) return fail("MatMul missing buffers");
        const auto* a_desc = graph.get_tensor(node.inputs[0]);
        const auto* b_desc = graph.get_tensor(node.inputs[1]);
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!a_desc || !b_desc || !out_desc) return fail("Missing tensor desc");
        if (!cpu_exec_matmul(node, *a_desc, *b_desc, *out_desc,
                             static_cast<const float*>(inputs[0]),
                             static_cast<const float*>(inputs[1]),
                             static_cast<float*>(outputs[0]))) {
            return fail("MatMul shape execution error");
        }
        return true;
    }
    case compiler::OpType::Conv2D: {
        if (inputs.size() < 2 || outputs.empty()) return fail("Conv2D missing buffers");
        const auto* x_desc = graph.get_tensor(node.inputs[0]);
        const auto* w_desc = graph.get_tensor(node.inputs[1]);
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!x_desc || !w_desc || !out_desc) return fail("Missing tensor desc");
        const float* bias = (inputs.size() >= 3) ? static_cast<const float*>(inputs[2]) : nullptr;
        if (!cpu_exec_conv2d(node, *x_desc, *w_desc, *out_desc,
                             static_cast<const float*>(inputs[0]),
                             static_cast<const float*>(inputs[1]),
                             bias,
                             static_cast<float*>(outputs[0]))) {
            return fail("Conv2D execution error");
        }
        return true;
    }
    case compiler::OpType::Relu: {
        if (inputs.empty() || outputs.empty()) return fail("Relu missing buffers");
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!out_desc) return fail("Missing output desc");
        const size_t n = static_cast<size_t>(out_desc->element_count());
        const float* in = static_cast<const float*>(inputs[0]);
        float* out = static_cast<float*>(outputs[0]);
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        }
        return true;
    }
    case compiler::OpType::SiLU: {
        if (inputs.empty() || outputs.empty()) return fail("SiLU missing buffers");
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!out_desc) return fail("Missing output desc");
        const size_t n = static_cast<size_t>(out_desc->element_count());
        const float* in = static_cast<const float*>(inputs[0]);
        float* out = static_cast<float*>(outputs[0]);
        for (size_t i = 0; i < n; ++i) {
            const float x = in[i];
            out[i] = x / (1.0f + std::exp(-x));
        }
        return true;
    }
    case compiler::OpType::Softmax: {
        if (inputs.empty() || outputs.empty()) return fail("Softmax missing buffers");
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!out_desc || out_desc->shape.empty()) return fail("Invalid Softmax output shape");
        const size_t inner = static_cast<size_t>(out_desc->shape.back());
        const size_t total = static_cast<size_t>(out_desc->element_count());
        if (inner == 0) return true;
        const size_t outer = total / inner;
        const float* in = static_cast<const float*>(inputs[0]);
        float* out = static_cast<float*>(outputs[0]);
        for (size_t o = 0; o < outer; ++o) {
            const float* row_in = in + o * inner;
            float* row_out = out + o * inner;
            float max_v = row_in[0];
            for (size_t j = 1; j < inner; ++j) {
                max_v = std::max(max_v, row_in[j]);
            }
            double sum_exp = 0.0;
            for (size_t j = 0; j < inner; ++j) {
                const float e = std::exp(row_in[j] - max_v);
                row_out[j] = e;
                sum_exp += static_cast<double>(e);
            }
            const float inv = static_cast<float>(1.0 / sum_exp);
            for (size_t j = 0; j < inner; ++j) {
                row_out[j] *= inv;
            }
        }
        return true;
    }
    case compiler::OpType::RMSNorm: {
        if (inputs.empty() || outputs.empty()) return fail("RMSNorm missing buffers");
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!out_desc || out_desc->shape.empty()) return fail("Invalid RMSNorm output shape");
        const size_t inner = static_cast<size_t>(out_desc->shape.back());
        const size_t total = static_cast<size_t>(out_desc->element_count());
        if (inner == 0) return true;
        const size_t outer = total / inner;
        const float* in = static_cast<const float*>(inputs[0]);
        const float* gamma = (inputs.size() >= 2) ? static_cast<const float*>(inputs[1]) : nullptr;
        float* out = static_cast<float*>(outputs[0]);
        constexpr float eps = 1e-5f;
        for (size_t o = 0; o < outer; ++o) {
            const float* row_in = in + o * inner;
            float* row_out = out + o * inner;
            double sq_sum = 0.0;
            for (size_t j = 0; j < inner; ++j) {
                sq_sum += static_cast<double>(row_in[j]) * static_cast<double>(row_in[j]);
            }
            const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sq_sum / static_cast<double>(inner)) + eps);
            for (size_t j = 0; j < inner; ++j) {
                float val = row_in[j] * inv_rms;
                if (gamma) {
                    val *= gamma[j];
                }
                row_out[j] = val;
            }
        }
        return true;
    }
    case compiler::OpType::Quantize: {
        if (inputs.empty() || outputs.empty()) return fail("Quantize missing buffers");
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!out_desc) return fail("Missing output desc");
        const size_t n = static_cast<size_t>(out_desc->element_count());
        const float* in = static_cast<const float*>(inputs[0]);
        const float scale = (out_desc->quant.scale != 0.0f) ? out_desc->quant.scale : 1.0f;
        const int32_t zp = out_desc->quant.zero_point;
        if (out_desc->dtype == compiler::DType::INT8) {
            int8_t* out = static_cast<int8_t*>(outputs[0]);
            for (size_t i = 0; i < n; ++i) {
                const long q = std::lround(in[i] / scale) + zp;
                out[i] = static_cast<int8_t>(std::clamp<long>(q, -128, 127));
            }
            return true;
        }
        if (out_desc->dtype == compiler::DType::UINT8) {
            uint8_t* out = static_cast<uint8_t*>(outputs[0]);
            for (size_t i = 0; i < n; ++i) {
                const long q = std::lround(in[i] / scale) + zp;
                out[i] = static_cast<uint8_t>(std::clamp<long>(q, 0, 255));
            }
            return true;
        }
        if (out_desc->dtype == compiler::DType::INT16) {
            int16_t* out = static_cast<int16_t*>(outputs[0]);
            for (size_t i = 0; i < n; ++i) {
                const long q = std::lround(in[i] / scale) + zp;
                out[i] = static_cast<int16_t>(std::clamp<long>(q, -32768, 32767));
            }
            return true;
        }
        return fail("Unsupported Quantize output dtype");
    }
    case compiler::OpType::Dequantize: {
        if (inputs.empty() || outputs.empty()) return fail("Dequantize missing buffers");
        const auto* in_desc = graph.get_tensor(node.inputs[0]);
        const auto* out_desc = graph.get_tensor(node.outputs[0]);
        if (!in_desc || !out_desc) return fail("Missing tensor desc");
        const size_t n = static_cast<size_t>(out_desc->element_count());
        const float scale = (in_desc->quant.scale != 0.0f) ? in_desc->quant.scale : 1.0f;
        const int32_t zp = in_desc->quant.zero_point;
        float* out = static_cast<float*>(outputs[0]);
        if (in_desc->dtype == compiler::DType::INT8) {
            const int8_t* in = static_cast<const int8_t*>(inputs[0]);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(static_cast<int32_t>(in[i]) - zp) * scale;
            }
            return true;
        }
        if (in_desc->dtype == compiler::DType::UINT8) {
            const uint8_t* in = static_cast<const uint8_t*>(inputs[0]);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(static_cast<int32_t>(in[i]) - zp) * scale;
            }
            return true;
        }
        if (in_desc->dtype == compiler::DType::INT16) {
            const int16_t* in = static_cast<const int16_t*>(inputs[0]);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(static_cast<int32_t>(in[i]) - zp) * scale;
            }
            return true;
        }
        return fail("Unsupported Dequantize input dtype");
    }
    case compiler::OpType::EnnPrecompiled:
        return fail("CPU_REFERENCE cannot execute EnnPrecompiled model");
    }
    return fail("Unknown op");
}

} // namespace

// ============================================================================
// Context / CompiledModel / Session
// ============================================================================

Context::Context(const compiler::TargetHardwareProfile& profile) : profile_(profile) {}

std::unique_ptr<Context> Context::create(const compiler::TargetHardwareProfile& profile) {
    return std::unique_ptr<Context>(new Context(profile));
}

void Context::register_executor(std::shared_ptr<BackendExecutor> executor) {
    if (!executor) {
        return;
    }
    const runtime::BackendKind k = executor->kind();
    profile_.upsert_backend(executor->info());
    executors_[k] = std::move(executor);
}

std::shared_ptr<BackendExecutor> Context::get_executor(runtime::BackendKind kind) const {
    auto it = executors_.find(kind);
    return (it != executors_.end()) ? it->second : nullptr;
}

std::vector<runtime::BackendInfo> Context::list_backends() const {
    return profile_.backends;
}

bool Context::compile_artifact(const compiler::Artifact& artifact,
                               runtime::ExecutionMode mode,
                               bool strict_mode,
                               std::unique_ptr<CompiledModel>& out_model,
                               std::string* err) const {
    compiler::Artifact compiled = artifact;
    if (!compiled.graph.infer_shapes(err)) {
        return false;
    }
    compiled.plan = compiler::Partitioner::partition(compiled.graph, profile_, mode, strict_mode);
    if (!compiled.plan.ok()) {
        if (err) {
            *err = compiled.plan.error_msg;
        }
        return false;
    }
    out_model.reset(new CompiledModel(std::move(compiled), profile_, executors_));
    if (err) {
        err->clear();
    }
    return true;
}

bool Context::compile_artifact_bytes(const uint8_t* data,
                                     size_t size,
                                     runtime::ExecutionMode mode,
                                     bool strict_mode,
                                     std::unique_ptr<CompiledModel>& out_model,
                                     std::string* err) const {
    compiler::Artifact art;
    if (!compiler::deserialize(data, size, art, err)) {
        return false;
    }
    return compile_artifact(art, mode, strict_mode, out_model, err);
}

bool Context::load_artifact_file(const std::string& path,
                                 std::unique_ptr<CompiledModel>& out_model,
                                 std::string* err) const {
    compiler::Artifact art;
    if (!compiler::load_file(path, art, err)) {
        return false;
    }
    if (!art.graph.validate(err)) {
        return false;
    }
    if (!art.plan.ok()) {
        if (err) {
            *err = art.plan.error_msg;
        }
        return false;
    }
    if (art.plan.segments.empty() && !art.graph.nodes.empty()) {
        return compile_artifact(art, art.plan.mode, art.plan.strict_mode, out_model, err);
    }
    out_model.reset(new CompiledModel(std::move(art), profile_, executors_));
    return true;
}

bool Context::load_artifact_bytes(const uint8_t* data,
                                  size_t size,
                                  std::unique_ptr<CompiledModel>& out_model,
                                  std::string* err) const {
    compiler::Artifact art;
    if (!compiler::deserialize(data, size, art, err)) {
        return false;
    }
    if (!art.graph.validate(err)) {
        return false;
    }
    if (!art.plan.ok()) {
        if (err) {
            *err = art.plan.error_msg;
        }
        return false;
    }
    out_model.reset(new CompiledModel(std::move(art), profile_, executors_));
    return true;
}

CompiledModel::CompiledModel(
    compiler::Artifact artifact,
    const compiler::TargetHardwareProfile& profile,
    std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors)
    : artifact_(std::move(artifact)),
      profile_(profile),
      executors_(std::move(executors)) {}

std::unique_ptr<Session> CompiledModel::create_session() const {
    return std::unique_ptr<Session>(new Session(artifact_, profile_, executors_));
}

Session::Session(const compiler::Artifact& artifact,
                 const compiler::TargetHardwareProfile& profile,
                 std::map<runtime::BackendKind, std::shared_ptr<BackendExecutor>> executors)
    : artifact_(artifact),
      profile_(profile),
      executors_(std::move(executors)) {
    init_buffers();
}

void Session::init_buffers() {
    tensor_buffers_.clear();
    name_to_id_.clear();
    for (const auto& t : artifact_.graph.tensors) {
        name_to_id_[t.name] = t.id;
        const size_t bytes = t.byte_size();
        std::vector<uint8_t> buf(bytes, 0);
        if (t.is_constant && !t.constant_data.empty()) {
            const size_t copy_n = std::min(bytes, t.constant_data.size());
            if (copy_n > 0) {
                std::memcpy(buf.data(), t.constant_data.data(), copy_n);
            }
        }
        tensor_buffers_[t.id] = std::move(buf);
    }
}

bool Session::set_input(const std::string& tensor_name,
                        const void* data,
                        size_t byte_len,
                        std::string* err) {
    auto it = name_to_id_.find(tensor_name);
    if (it == name_to_id_.end()) {
        if (err) *err = "Unknown input tensor: " + tensor_name;
        return false;
    }
    auto bit = tensor_buffers_.find(it->second);
    if (bit == tensor_buffers_.end() || !data) {
        if (err) *err = "Invalid input buffer for tensor: " + tensor_name;
        return false;
    }
    if (bit->second.size() != byte_len) {
        if (err) {
            *err = "Input size mismatch for '" + tensor_name + "': expected " +
                   std::to_string(bit->second.size()) + " bytes, got " +
                   std::to_string(byte_len);
        }
        return false;
    }
    if (byte_len > 0) {
        std::memcpy(bit->second.data(), data, byte_len);
    }
    return true;
}

bool Session::set_input(const std::string& tensor_name,
                        const std::vector<float>& data,
                        std::string* err) {
    return set_input(tensor_name, data.data(), data.size() * sizeof(float), err);
}

bool Session::run(std::string* err) {
    last_report_ = runtime::ExecutionReport{};
    last_report_.mode = artifact_.plan.mode;
    last_report_.strict_mode = artifact_.plan.strict_mode;

    if (!artifact_.plan.ok()) {
        last_report_.error_msg = artifact_.plan.error_msg;
        if (err) *err = last_report_.error_msg;
        return false;
    }

    const auto total_t0 = std::chrono::steady_clock::now();

    for (const auto& seg : artifact_.plan.segments) {
        if (seg.is_fallback) {
            last_report_.any_fallback = true;
        }
        for (uint32_t nid : seg.node_ids) {
            const compiler::Node* node = artifact_.graph.get_node(nid);
            if (!node) {
                last_report_.error_msg = "Missing node in execution plan: " + std::to_string(nid);
                if (err) *err = last_report_.error_msg;
                return false;
            }

            std::vector<const void*> in_ptrs;
            in_ptrs.reserve(node->inputs.size());
            for (uint32_t tid : node->inputs) {
                auto bit = tensor_buffers_.find(tid);
                if (bit == tensor_buffers_.end()) {
                    last_report_.error_msg = "Missing input tensor buffer";
                    if (err) *err = last_report_.error_msg;
                    return false;
                }
                in_ptrs.push_back(bit->second.data());
            }

            std::vector<void*> out_ptrs;
            out_ptrs.reserve(node->outputs.size());
            for (uint32_t tid : node->outputs) {
                auto bit = tensor_buffers_.find(tid);
                if (bit == tensor_buffers_.end()) {
                    last_report_.error_msg = "Missing output tensor buffer";
                    if (err) *err = last_report_.error_msg;
                    return false;
                }
                out_ptrs.push_back(bit->second.data());
            }

            const auto node_t0 = std::chrono::steady_clock::now();
            runtime::BackendKind actual_backend = seg.backend;
            bool fallback_used = seg.is_fallback;

            auto ex_it = executors_.find(seg.backend);
            bool ok = false;
            std::string node_err;
            if (ex_it != executors_.end() && ex_it->second &&
                ex_it->second->supports_op(*node, artifact_.graph)) {
                ok = ex_it->second->execute_node(*node, artifact_.graph, in_ptrs, out_ptrs, &node_err);
            } else if (seg.backend == runtime::BackendKind::CPU_REFERENCE ||
                       seg.backend == runtime::BackendKind::CPU_NEON) {
                actual_backend = runtime::BackendKind::CPU_REFERENCE;
                ok = cpu_execute_node(*node, artifact_.graph, in_ptrs, out_ptrs, &node_err);
            } else {
                if (artifact_.plan.strict_mode) {
                    last_report_.error_msg =
                        "Strict execution failed: no registered executor for " +
                        std::string(runtime::to_string(seg.backend));
                    if (err) *err = last_report_.error_msg;
                    return false;
                }
                actual_backend = runtime::BackendKind::CPU_REFERENCE;
                fallback_used = true;
                last_report_.any_fallback = true;
                ok = cpu_execute_node(*node, artifact_.graph, in_ptrs, out_ptrs, &node_err);
            }

            const auto node_t1 = std::chrono::steady_clock::now();
            const double dur_us =
                std::chrono::duration<double, std::micro>(node_t1 - node_t0).count();

            runtime::NodeExecutionTrace trace;
            trace.node_name = node->name;
            trace.requested = seg.backend;
            trace.actual = actual_backend;
            trace.fallback_used = fallback_used;
            trace.duration_us = dur_us;
            last_report_.traces.push_back(std::move(trace));

            if (!ok) {
                last_report_.error_msg = node_err;
                if (err) *err = node_err;
                return false;
            }
        }
    }

    const auto total_t1 = std::chrono::steady_clock::now();
    last_report_.total_us =
        std::chrono::duration<double, std::micro>(total_t1 - total_t0).count();
    return true;
}

bool Session::get_output(const std::string& tensor_name,
                         void* out_data,
                         size_t byte_len,
                         std::string* err) const {
    auto it = name_to_id_.find(tensor_name);
    if (it == name_to_id_.end()) {
        if (err) *err = "Unknown output tensor: " + tensor_name;
        return false;
    }
    auto bit = tensor_buffers_.find(it->second);
    if (bit == tensor_buffers_.end() || !out_data) {
        if (err) *err = "Invalid output buffer for tensor: " + tensor_name;
        return false;
    }
    if (bit->second.size() != byte_len) {
        if (err) {
            *err = "Output size mismatch for '" + tensor_name + "': expected " +
                   std::to_string(bit->second.size()) + " bytes, got " +
                   std::to_string(byte_len);
        }
        return false;
    }
    if (byte_len > 0) {
        std::memcpy(out_data, bit->second.data(), byte_len);
    }
    return true;
}

bool Session::get_output(const std::string& tensor_name,
                         std::vector<float>& out_data,
                         std::string* err) const {
    auto it = name_to_id_.find(tensor_name);
    if (it == name_to_id_.end()) {
        if (err) *err = "Unknown output tensor: " + tensor_name;
        return false;
    }
    auto bit = tensor_buffers_.find(it->second);
    if (bit == tensor_buffers_.end()) {
        if (err) *err = "Missing output tensor buffer: " + tensor_name;
        return false;
    }
    out_data.resize(bit->second.size() / sizeof(float));
    if (!bit->second.empty()) {
        std::memcpy(out_data.data(), bit->second.data(), out_data.size() * sizeof(float));
    }
    return true;
}

} // namespace sdk
} // namespace exynova

// ============================================================================
// C11 ABI Implementation
// ============================================================================

struct exynova_context {
    std::unique_ptr<exynova::sdk::Context> impl;
};

struct exynova_compiled_model {
    std::unique_ptr<exynova::sdk::CompiledModel> impl;
};

struct exynova_session {
    std::unique_ptr<exynova::sdk::Session> impl;
};

static void copy_c_str(char* dst, size_t cap, const std::string& src) {
    if (cap == 0) return;
    const size_t n = std::min(cap - 1, src.size());
    if (n > 0) {
        std::memcpy(dst, src.data(), n);
    }
    dst[n] = '\0';
}

extern "C" {

const char* exynova_version_string(void) {
    return exynova::sdk::version_string();
}

exynova_status_t exynova_initialize(exynova_context_t* out_ctx) {
    if (!out_ctx) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        auto* ctx = new exynova_context();
        ctx->impl = exynova::sdk::Context::create();
        *out_ctx = ctx;
        return EXYNOVA_STATUS_OK;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

void exynova_shutdown(exynova_context_t ctx) {
    delete ctx;
}

exynova_status_t exynova_list_backends(exynova_context_t ctx,
                                       exynova_backend_info_t* out_infos,
                                       uint32_t max_count,
                                       uint32_t* out_count) {
    if (!ctx || !ctx->impl || !out_count) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        const auto list = ctx->impl->list_backends();
        *out_count = static_cast<uint32_t>(list.size());
        if (out_infos && max_count > 0) {
            const uint32_t n = std::min<uint32_t>(max_count, static_cast<uint32_t>(list.size()));
            for (uint32_t i = 0; i < n; ++i) {
                std::memset(&out_infos[i], 0, sizeof(exynova_backend_info_t));
                out_infos[i].kind = static_cast<exynova_backend_kind_t>(list[i].kind);
                out_infos[i].status = static_cast<exynova_capability_status_t>(list[i].status);
                copy_c_str(out_infos[i].name, sizeof(out_infos[i].name), list[i].name);
                copy_c_str(out_infos[i].device_name, sizeof(out_infos[i].device_name), list[i].device_name);
                copy_c_str(out_infos[i].details, sizeof(out_infos[i].details), list[i].details);
            }
        }
        return EXYNOVA_STATUS_OK;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_compile_artifact(exynova_context_t ctx,
                                          const uint8_t* artifact_bytes,
                                          size_t byte_len,
                                          exynova_execution_mode_t mode,
                                          int strict_mode,
                                          exynova_compiled_model_t* out_model) {
    if (!ctx || !ctx->impl || !artifact_bytes || byte_len == 0 || !out_model) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        std::unique_ptr<exynova::sdk::CompiledModel> model;
        std::string err;
        exynova::compiler::Artifact art;
        if (!exynova::compiler::deserialize(artifact_bytes, byte_len, art, &err)) {
            return EXYNOVA_STATUS_CORRUPT_ARTIFACT;
        }
        if (!ctx->impl->compile_artifact(
                art,
                static_cast<exynova::runtime::ExecutionMode>(mode),
                strict_mode != 0,
                model,
                &err)) {
            return strict_mode ? EXYNOVA_STATUS_STRICT_REJECTED : EXYNOVA_STATUS_UNSUPPORTED;
        }
        auto* handle = new exynova_compiled_model();
        handle->impl = std::move(model);
        *out_model = handle;
        return EXYNOVA_STATUS_OK;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_load_artifact(exynova_context_t ctx,
                                       const char* file_path,
                                       exynova_compiled_model_t* out_model) {
    if (!ctx || !ctx->impl || !file_path || !out_model) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        std::unique_ptr<exynova::sdk::CompiledModel> model;
        std::string err;
        if (!ctx->impl->load_artifact_file(file_path, model, &err)) {
            if (err.find("open") != std::string::npos || err.find("reading") != std::string::npos) {
                return EXYNOVA_STATUS_IO_ERROR;
            }
            return EXYNOVA_STATUS_CORRUPT_ARTIFACT;
        }
        auto* handle = new exynova_compiled_model();
        handle->impl = std::move(model);
        *out_model = handle;
        return EXYNOVA_STATUS_OK;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_create_session(exynova_compiled_model_t model,
                                        exynova_session_t* out_session) {
    if (!model || !model->impl || !out_session) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        auto* sess = new exynova_session();
        sess->impl = model->impl->create_session();
        *out_session = sess;
        return EXYNOVA_STATUS_OK;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_set_input(exynova_session_t session,
                                   const char* tensor_name,
                                   const void* data,
                                   size_t byte_len) {
    if (!session || !session->impl || !tensor_name || !data) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        std::string err;
        return session->impl->set_input(tensor_name, data, byte_len, &err)
                   ? EXYNOVA_STATUS_OK
                   : EXYNOVA_STATUS_INVALID_ARG;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_run(exynova_session_t session) {
    if (!session || !session->impl) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        std::string err;
        return session->impl->run(&err) ? EXYNOVA_STATUS_OK : EXYNOVA_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_get_output(exynova_session_t session,
                                    const char* tensor_name,
                                    void* out_data,
                                    size_t byte_len) {
    if (!session || !session->impl || !tensor_name || !out_data) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        std::string err;
        return session->impl->get_output(tensor_name, out_data, byte_len, &err)
                   ? EXYNOVA_STATUS_OK
                   : EXYNOVA_STATUS_INVALID_ARG;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

exynova_status_t exynova_get_last_report(exynova_session_t session,
                                         exynova_exec_report_t* out_report) {
    if (!session || !session->impl || !out_report) {
        return EXYNOVA_STATUS_INVALID_ARG;
    }
    try {
        const auto& r = session->impl->last_report();
        std::memset(out_report, 0, sizeof(exynova_exec_report_t));
        out_report->mode = static_cast<exynova_execution_mode_t>(r.mode);
        out_report->strict_mode = r.strict_mode ? 1 : 0;
        out_report->any_fallback = r.any_fallback ? 1 : 0;
        out_report->total_us = r.total_us;
        out_report->trace_count = static_cast<uint32_t>(r.traces.size());
        copy_c_str(out_report->error_msg, sizeof(out_report->error_msg), r.error_msg);
        return EXYNOVA_STATUS_OK;
    } catch (...) {
        return EXYNOVA_STATUS_RUNTIME_ERROR;
    }
}

void exynova_destroy_session(exynova_session_t session) {
    delete session;
}

void exynova_destroy_model(exynova_compiled_model_t model) {
    delete model;
}

} // extern "C"
