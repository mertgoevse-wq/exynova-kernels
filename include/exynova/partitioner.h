#pragma once

#include "exynova/backend_registry.h"
#include "exynova/ir.h"

#include <cstdint>
#include <string>
#include <vector>

namespace exynova {
namespace compiler {

struct TargetHardwareProfile {
    std::string soc_name;
    std::string device_model;
    std::string gpu_name;
    std::string npu_name;
    std::vector<runtime::BackendInfo> backends;

    runtime::CapabilityStatus status_of(runtime::BackendKind kind) const;
    bool is_execution_ready(runtime::BackendKind kind) const;
    void upsert_backend(const runtime::BackendInfo& info);

    static TargetHardwareProfile exynos_1580(bool npu_exec_ready = false,
                                             bool gpu_exec_ready = false);
};

struct PartitionedSegment {
    uint32_t segment_id = 0;
    runtime::BackendKind backend = runtime::BackendKind::CPU_REFERENCE;
    std::vector<uint32_t> node_ids;
    bool is_fallback = false;
    std::string reason;
    uint64_t estimated_flops = 0;
};

struct ExecutionPlan {
    runtime::ExecutionMode mode = runtime::ExecutionMode::AUTO;
    bool strict_mode = false;
    std::vector<PartitionedSegment> segments;
    std::string error_msg;

    bool ok() const { return error_msg.empty(); }
    std::string dump_ascii() const;
};

class Partitioner {
public:
    static bool backend_supports_op(runtime::BackendKind backend,
                                    const Node& node,
                                    const Graph& graph);

    static ExecutionPlan partition(const Graph& graph,
                                   const TargetHardwareProfile& profile,
                                   runtime::ExecutionMode mode,
                                   bool strict_mode);
};

} // namespace compiler
} // namespace exynova
