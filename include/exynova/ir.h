#pragma once

#include "exynova/backend_registry.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace exynova {
namespace compiler {

enum class DType {
    FP32 = 0,
    FP16 = 1,
    INT32 = 2,
    INT16 = 3,
    INT8 = 4,
    UINT8 = 5
};

enum class Layout {
    NHWC = 0,
    NCHW = 1,
    NC = 2,
    ANY = 3
};

enum class OpType {
    Add = 0,
    Mul = 1,
    MatMul = 2,
    Conv2D = 3,
    Relu = 4,
    SiLU = 5,
    Softmax = 6,
    RMSNorm = 7,
    Quantize = 8,
    Dequantize = 9,
    EnnPrecompiled = 10
};

inline size_t dtype_size_bytes(DType dtype) {
    switch (dtype) {
    case DType::FP32:  return 4;
    case DType::FP16:  return 2;
    case DType::INT32: return 4;
    case DType::INT16: return 2;
    case DType::INT8:  return 1;
    case DType::UINT8: return 1;
    }
    return 0;
}

inline const char* to_string(DType dtype) {
    switch (dtype) {
    case DType::FP32:  return "FP32";
    case DType::FP16:  return "FP16";
    case DType::INT32: return "INT32";
    case DType::INT16: return "INT16";
    case DType::INT8:  return "INT8";
    case DType::UINT8: return "UINT8";
    }
    return "UNKNOWN";
}

inline const char* to_string(Layout layout) {
    switch (layout) {
    case Layout::NHWC: return "NHWC";
    case Layout::NCHW: return "NCHW";
    case Layout::NC:   return "NC";
    case Layout::ANY:  return "ANY";
    }
    return "UNKNOWN";
}

inline const char* to_string(OpType op) {
    switch (op) {
    case OpType::Add:            return "Add";
    case OpType::Mul:            return "Mul";
    case OpType::MatMul:         return "MatMul";
    case OpType::Conv2D:         return "Conv2D";
    case OpType::Relu:           return "Relu";
    case OpType::SiLU:           return "SiLU";
    case OpType::Softmax:        return "Softmax";
    case OpType::RMSNorm:        return "RMSNorm";
    case OpType::Quantize:       return "Quantize";
    case OpType::Dequantize:     return "Dequantize";
    case OpType::EnnPrecompiled: return "EnnPrecompiled";
    }
    return "UNKNOWN";
}

struct QuantParams {
    bool quantized = false;
    float scale = 1.0f;
    int32_t zero_point = 0;
    std::vector<float> per_channel_scales;
};

struct TensorDesc {
    uint32_t id = 0;
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::FP32;
    Layout layout = Layout::ANY;
    QuantParams quant;
    bool is_constant = false;
    std::vector<uint8_t> constant_data;

    int64_t element_count() const;
    size_t byte_size() const;
};

struct NodeAttrs {
    std::string fused_activation;
    std::array<int32_t, 2> strides{{1, 1}};
    std::array<int32_t, 4> padding{{0, 0, 0, 0}};
    std::string enn_model_path;
    uint32_t enn_preset_id = 0;
};

struct Node {
    uint32_t id = 0;
    std::string name;
    OpType op = OpType::Add;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    NodeAttrs attrs;
    runtime::BackendKind preferred_backend = runtime::BackendKind::CPU_REFERENCE;
    uint64_t cached_flops = 0;

    uint64_t estimated_flops() const;
    uint64_t estimated_flops(const std::vector<TensorDesc>& tensors) const;
};

struct MemoryPlan {
    size_t peak_bytes = 0;
    size_t total_unshared_bytes = 0;
    size_t buffer_reuse_count = 0;
    size_t allocated_buffers = 0;
    std::vector<std::pair<uint32_t, size_t>> tensor_to_buffer;
};

class Graph {
public:
    std::string name;
    uint32_t ir_version = 1;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::vector<TensorDesc> tensors;
    std::vector<Node> nodes;

    uint32_t add_tensor(const std::string& tensor_name,
                        const std::vector<int64_t>& shape,
                        DType dtype = DType::FP32,
                        Layout layout = Layout::ANY,
                        bool is_constant = false,
                        std::vector<uint8_t> constant_data = {});

    uint32_t add_constant_f32(const std::string& tensor_name,
                              const std::vector<int64_t>& shape,
                              const std::vector<float>& values,
                              Layout layout = Layout::ANY);

    uint32_t add_node(const std::string& node_name,
                      OpType op,
                      std::vector<uint32_t> node_inputs,
                      std::vector<uint32_t> node_outputs,
                      NodeAttrs attrs = {},
                      runtime::BackendKind preferred_backend =
                          runtime::BackendKind::CPU_REFERENCE);

    const TensorDesc* get_tensor(uint32_t id) const;
    TensorDesc* get_tensor_mut(uint32_t id);
    const TensorDesc* find_tensor_by_name(const std::string& tensor_name) const;
    const Node* get_node(uint32_t id) const;
    Node* get_node_mut(uint32_t id);

    bool validate(std::string* err = nullptr) const;
    bool infer_shapes(std::string* err = nullptr);
    std::vector<uint32_t> topological_order(bool* has_cycle = nullptr) const;
    size_t run_dead_code_elimination();
    size_t run_operator_fusion();
    MemoryPlan compute_memory_plan() const;
    std::string dump_summary() const;

private:
    uint32_t next_tensor_id_ = 1;
    uint32_t next_node_id_ = 1;
    void sync_id_counters();
    friend class ArtifactSerializer;
};

} // namespace compiler
} // namespace exynova
