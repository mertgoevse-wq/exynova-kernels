#include "exynova/partitioner.h"

#include <sstream>

namespace exynova {
namespace compiler {

runtime::CapabilityStatus TargetHardwareProfile::status_of(runtime::BackendKind kind) const {
    for (const auto& b : backends) {
        if (b.kind == kind) {
            return b.status;
        }
    }
    return runtime::CapabilityStatus::UNKNOWN;
}

bool TargetHardwareProfile::is_execution_ready(runtime::BackendKind kind) const {
    return runtime::is_capability_ready(status_of(kind));
}

void TargetHardwareProfile::upsert_backend(const runtime::BackendInfo& info) {
    for (auto& b : backends) {
        if (b.kind == info.kind) {
            b = info;
            return;
        }
    }
    backends.push_back(info);
}

TargetHardwareProfile TargetHardwareProfile::exynos_1580(bool npu_exec_ready,
                                                         bool gpu_exec_ready) {
    TargetHardwareProfile profile;
    profile.soc_name = "S5E8855";
    profile.device_model = "Samsung Galaxy A56 (Exynos 1580)";
    profile.gpu_name = "Samsung Xclipse 540 (Vulkan 1.3)";
    profile.npu_name = "Samsung Pablo/Rose NPU (ENN)";

    profile.backends = {
        {runtime::BackendKind::CPU_REFERENCE,
         "cpu_reference",
         runtime::CapabilityStatus::EXECUTION_VERIFIED,
         "ARMv9-A / Host CPU",
         "FP32 reference kernels"},
        {runtime::BackendKind::CPU_NEON,
         "cpu_neon",
         runtime::CapabilityStatus::DETECTED,
         "Cortex-A720 / A520",
         "NEON SIMD backend"},
        {runtime::BackendKind::GPU_VULKAN,
         "gpu_vulkan",
         gpu_exec_ready ? runtime::CapabilityStatus::EXECUTION_VERIFIED
                        : runtime::CapabilityStatus::DETECTED,
         "Samsung Xclipse 540",
         gpu_exec_ready ? "Vulkan 1.3 compute pipeline verified"
                        : "Vulkan 1.3 device detected; compute executor not verified"},
        {runtime::BackendKind::GPU_LITERT,
         "gpu_litert",
         runtime::CapabilityStatus::DETECTED,
         "Samsung Xclipse 540",
         "LiteRT GPU delegate"},
        {runtime::BackendKind::NPU_NNAPI,
         "npu_nnapi",
         runtime::CapabilityStatus::UNSUPPORTED,
         "Android NNAPI",
         "Deprecated on Android 15 (API 35)"},
        {runtime::BackendKind::NPU_ENN,
         "npu_enn",
         npu_exec_ready ? runtime::CapabilityStatus::EXECUTION_VERIFIED
                        : runtime::CapabilityStatus::DETECTED,
         "Samsung Pablo/Rose NPU",
         npu_exec_ready ? "ENN C-API NPU execution verified"
                        : "ENN libraries detected; NPU execution not verified"},
        {runtime::BackendKind::NPU_LITECORE,
         "npu_litecore",
         runtime::CapabilityStatus::EXPERIMENTAL,
         "LiteCore",
         "Vendor private runtime"},
        {runtime::BackendKind::NPU_EXPERIMENTAL,
         "npu_experimental",
         runtime::CapabilityStatus::EXPERIMENTAL,
         "Exynos NPU",
         "Experimental probe target"},
    };
    return profile;
}

std::string ExecutionPlan::dump_ascii() const {
    std::ostringstream oss;
    oss << "ExecutionPlan(mode=" << runtime::to_string(mode)
        << ", strict=" << (strict_mode ? "true" : "false") << ")\n";
    if (!error_msg.empty()) {
        oss << "  ERROR: " << error_msg << "\n";
        return oss.str();
    }
    for (const auto& seg : segments) {
        oss << "  Segment #" << seg.segment_id
            << " [" << runtime::to_string(seg.backend) << "]"
            << (seg.is_fallback ? " (FALLBACK)" : "")
            << " nodes=" << seg.node_ids.size()
            << " flops=" << seg.estimated_flops
            << " | " << seg.reason << "\n";
    }
    return oss.str();
}

static bool has_quantized_tensor(const Node& node, const Graph& graph) {
    auto is_q = [](DType dt) {
        return dt == DType::INT8 || dt == DType::INT16;
    };
    for (uint32_t tid : node.inputs) {
        const TensorDesc* t = graph.get_tensor(tid);
        if (t && (is_q(t->dtype) || t->quant.quantized)) {
            return true;
        }
    }
    for (uint32_t tid : node.outputs) {
        const TensorDesc* t = graph.get_tensor(tid);
        if (t && (is_q(t->dtype) || t->quant.quantized)) {
            return true;
        }
    }
    return false;
}

static int64_t primary_output_elements(const Node& node, const Graph& graph) {
    if (node.outputs.empty()) {
        return 0;
    }
    const TensorDesc* out = graph.get_tensor(node.outputs[0]);
    return out ? out->element_count() : 0;
}

bool Partitioner::backend_supports_op(runtime::BackendKind backend,
                                      const Node& node,
                                      const Graph& /*graph*/) {
    switch (backend) {
    case runtime::BackendKind::CPU_REFERENCE:
    case runtime::BackendKind::CPU_NEON:
        return node.op != OpType::EnnPrecompiled;
    case runtime::BackendKind::GPU_VULKAN:
    case runtime::BackendKind::GPU_LITERT:
        return node.op == OpType::Add || node.op == OpType::Mul ||
               node.op == OpType::MatMul || node.op == OpType::Conv2D ||
               node.op == OpType::Relu || node.op == OpType::SiLU ||
               node.op == OpType::Softmax || node.op == OpType::RMSNorm;
    case runtime::BackendKind::NPU_ENN:
        return node.op == OpType::EnnPrecompiled || node.op == OpType::Conv2D ||
               node.op == OpType::MatMul || node.op == OpType::Add ||
               node.op == OpType::Mul || node.op == OpType::Relu ||
               node.op == OpType::Quantize || node.op == OpType::Dequantize;
    case runtime::BackendKind::NPU_NNAPI:
    case runtime::BackendKind::NPU_LITECORE:
    case runtime::BackendKind::NPU_EXPERIMENTAL:
        return false;
    }
    return false;
}

ExecutionPlan Partitioner::partition(const Graph& graph,
                                     const TargetHardwareProfile& profile,
                                     runtime::ExecutionMode mode,
                                     bool strict_mode) {
    ExecutionPlan plan;
    plan.mode = mode;
    plan.strict_mode = strict_mode;

    std::string val_err;
    if (!graph.validate(&val_err)) {
        plan.error_msg = "Invalid graph: " + val_err;
        return plan;
    }

    const std::vector<uint32_t> order = graph.topological_order();

    struct NodeAssignment {
        uint32_t node_id = 0;
        runtime::BackendKind backend = runtime::BackendKind::CPU_REFERENCE;
        bool is_fallback = false;
        std::string reason;
        uint64_t flops = 0;
    };

    auto fail_plan = [&](const std::string& msg) {
        plan.segments.clear();
        plan.error_msg = msg;
        return plan;
    };

    const bool cpu_ready = profile.is_execution_ready(runtime::BackendKind::CPU_REFERENCE);
    const bool gpu_ready = profile.is_execution_ready(runtime::BackendKind::GPU_VULKAN);
    const bool npu_ready = profile.is_execution_ready(runtime::BackendKind::NPU_ENN);

    std::vector<NodeAssignment> assignments;
    assignments.reserve(order.size());

    for (uint32_t nid : order) {
        const Node* node = graph.get_node(nid);
        if (!node) {
            continue;
        }
        const uint64_t flops = node->estimated_flops(graph.tensors);
        NodeAssignment a;
        a.node_id = nid;
        a.flops = flops;

        if (mode == runtime::ExecutionMode::CPU_ONLY) {
            if (!cpu_ready) {
                return fail_plan("Strict/CPU_ONLY rejected: CPU_REFERENCE status is " +
                                 std::string(runtime::to_string(
                                     profile.status_of(runtime::BackendKind::CPU_REFERENCE))));
            }
            if (!backend_supports_op(runtime::BackendKind::CPU_REFERENCE, *node, graph)) {
                return fail_plan("CPU_ONLY cannot execute op '" +
                                 std::string(to_string(node->op)) + "' on node '" +
                                 node->name + "'");
            }
            a.backend = runtime::BackendKind::CPU_REFERENCE;
            a.is_fallback = false;
            a.reason = "CPU_ONLY mode";
        } else if (mode == runtime::ExecutionMode::GPU_ONLY) {
            const bool supported = backend_supports_op(runtime::BackendKind::GPU_VULKAN, *node, graph);
            if (gpu_ready && supported) {
                a.backend = runtime::BackendKind::GPU_VULKAN;
                a.is_fallback = false;
                a.reason = "GPU_ONLY -> GPU_VULKAN";
            } else {
                if (strict_mode) {
                    if (!gpu_ready) {
                        return fail_plan("Strict GPU_ONLY rejected: GPU_VULKAN status is " +
                                         std::string(runtime::to_string(
                                             profile.status_of(runtime::BackendKind::GPU_VULKAN))) +
                                         " (no silent CPU fallback permitted)");
                    }
                    return fail_plan("Strict GPU_ONLY rejected: op '" +
                                     std::string(to_string(node->op)) + "' on node '" +
                                     node->name + "' unsupported on GPU_VULKAN");
                }
                if (!cpu_ready || !backend_supports_op(runtime::BackendKind::CPU_REFERENCE, *node, graph)) {
                    return fail_plan("GPU_ONLY fallback failed: CPU_REFERENCE unavailable for '" +
                                     node->name + "'");
                }
                a.backend = runtime::BackendKind::CPU_REFERENCE;
                a.is_fallback = true;
                a.reason = !gpu_ready ? "GPU_VULKAN not verified; explicit fallback to CPU_REFERENCE"
                                      : "Op unsupported on GPU_VULKAN; explicit fallback to CPU_REFERENCE";
            }
        } else if (mode == runtime::ExecutionMode::NPU_ONLY) {
            const bool supported = backend_supports_op(runtime::BackendKind::NPU_ENN, *node, graph);
            if (npu_ready && supported) {
                a.backend = runtime::BackendKind::NPU_ENN;
                a.is_fallback = false;
                a.reason = "NPU_ONLY -> NPU_ENN";
            } else {
                if (strict_mode) {
                    if (!npu_ready) {
                        return fail_plan("Strict NPU_ONLY rejected: NPU_ENN status is " +
                                         std::string(runtime::to_string(
                                             profile.status_of(runtime::BackendKind::NPU_ENN))) +
                                         " (no silent CPU fallback permitted)");
                    }
                    return fail_plan("Strict NPU_ONLY rejected: op '" +
                                     std::string(to_string(node->op)) + "' on node '" +
                                     node->name + "' unsupported on NPU_ENN");
                }
                if (!cpu_ready || !backend_supports_op(runtime::BackendKind::CPU_REFERENCE, *node, graph)) {
                    return fail_plan("NPU_ONLY fallback failed: CPU_REFERENCE unavailable for '" +
                                     node->name + "'");
                }
                a.backend = runtime::BackendKind::CPU_REFERENCE;
                a.is_fallback = true;
                a.reason = !npu_ready ? "NPU_ENN not verified; explicit fallback to CPU_REFERENCE"
                                      : "Op unsupported on NPU_ENN; explicit fallback to CPU_REFERENCE";
            }
        } else {
            // AUTO / HYBRID heterogeneous routing
            const bool is_quant_heavy =
                has_quantized_tensor(*node, graph) &&
                (node->op == OpType::Conv2D || node->op == OpType::MatMul ||
                 node->op == OpType::Add || node->op == OpType::Mul ||
                 node->op == OpType::Relu || node->op == OpType::Quantize ||
                 node->op == OpType::Dequantize);

            const int64_t out_elems = primary_output_elements(*node, graph);
            const bool is_gpu_heavy =
                (node->op == OpType::MatMul || node->op == OpType::Conv2D ||
                 node->op == OpType::SiLU || node->op == OpType::Softmax ||
                 ((node->op == OpType::Add || node->op == OpType::Mul ||
                   node->op == OpType::Relu) &&
                  out_elems >= 4096));

            if (node->op == OpType::EnnPrecompiled) {
                if (npu_ready) {
                    a.backend = runtime::BackendKind::NPU_ENN;
                    a.is_fallback = false;
                    a.reason = "EnnPrecompiled -> NPU_ENN";
                } else {
                    return fail_plan("EnnPrecompiled node '" + node->name +
                                     "' requires verified NPU_ENN backend");
                }
            } else if (is_quant_heavy) {
                if (npu_ready) {
                    a.backend = runtime::BackendKind::NPU_ENN;
                    a.is_fallback = false;
                    a.reason = "Quantized heavy op -> NPU_ENN";
                } else if (gpu_ready &&
                           backend_supports_op(runtime::BackendKind::GPU_VULKAN, *node, graph)) {
                    a.backend = runtime::BackendKind::GPU_VULKAN;
                    a.is_fallback = true;
                    a.reason = "NPU_ENN unavailable for quantized op; routed to GPU_VULKAN";
                } else if (cpu_ready) {
                    a.backend = runtime::BackendKind::CPU_REFERENCE;
                    a.is_fallback = true;
                    a.reason = "NPU_ENN unavailable for quantized op; fallback to CPU_REFERENCE";
                } else {
                    return fail_plan("No backend available for quantized node '" + node->name + "'");
                }
            } else if (is_gpu_heavy) {
                if (gpu_ready) {
                    a.backend = runtime::BackendKind::GPU_VULKAN;
                    a.is_fallback = false;
                    a.reason = "Compute-heavy FP op -> GPU_VULKAN";
                } else if (cpu_ready) {
                    a.backend = runtime::BackendKind::CPU_REFERENCE;
                    a.is_fallback = true;
                    a.reason = "GPU_VULKAN not execution-ready; fallback to CPU_REFERENCE";
                } else {
                    return fail_plan("No backend available for node '" + node->name + "'");
                }
            } else {
                if (!cpu_ready) {
                    return fail_plan("CPU_REFERENCE unavailable for node '" + node->name + "'");
                }
                a.backend = runtime::BackendKind::CPU_REFERENCE;
                a.is_fallback = false;
                a.reason = "Small/scalar-friendly op -> CPU_REFERENCE";
            }
        }

        assignments.push_back(std::move(a));
    }

    for (const auto& a : assignments) {
        if (!plan.segments.empty() &&
            plan.segments.back().backend == a.backend &&
            plan.segments.back().is_fallback == a.is_fallback) {
            plan.segments.back().node_ids.push_back(a.node_id);
            plan.segments.back().estimated_flops += a.flops;
        } else {
            PartitionedSegment seg;
            seg.segment_id = static_cast<uint32_t>(plan.segments.size());
            seg.backend = a.backend;
            seg.node_ids.push_back(a.node_id);
            seg.is_fallback = a.is_fallback;
            seg.reason = a.reason;
            seg.estimated_flops = a.flops;
            plan.segments.push_back(std::move(seg));
        }
    }

    return plan;
}

} // namespace compiler
} // namespace exynova
