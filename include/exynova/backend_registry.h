#pragma once

#include <string>
#include <vector>

namespace exynova {
namespace runtime {

enum class BackendKind {
    CPU_REFERENCE = 0,
    CPU_NEON = 1,
    GPU_VULKAN = 2,
    GPU_LITERT = 3,
    NPU_NNAPI = 4,
    NPU_ENN = 5,
    NPU_LITECORE = 6,
    NPU_EXPERIMENTAL = 7
};

enum class CapabilityStatus {
    UNKNOWN = 0,
    DETECTED = 1,
    ACCESSIBLE = 2,
    INITIALIZED = 3,
    COMPILE_VERIFIED = 4,
    EXECUTION_VERIFIED = 5,
    UNSUPPORTED = 6,
    ACCESS_DENIED = 7,
    EXPERIMENTAL = 8
};

enum class ExecutionMode {
    CPU_ONLY = 0,
    GPU_ONLY = 1,
    NPU_ONLY = 2,
    AUTO = 3,
    HYBRID = 4
};

struct BackendInfo {
    BackendKind kind = BackendKind::CPU_REFERENCE;
    std::string name;
    CapabilityStatus status = CapabilityStatus::UNKNOWN;
    std::string device_name;
    std::string details;
};

struct NodeExecutionTrace {
    std::string node_name;
    BackendKind requested = BackendKind::CPU_REFERENCE;
    BackendKind actual = BackendKind::CPU_REFERENCE;
    bool fallback_used = false;
    double duration_us = 0.0;
};

struct ExecutionReport {
    ExecutionMode mode = ExecutionMode::AUTO;
    bool strict_mode = false;
    bool any_fallback = false;
    double total_us = 0.0;
    std::vector<NodeExecutionTrace> traces;
    std::string error_msg;
};

inline const char* to_string(BackendKind kind) {
    switch (kind) {
    case BackendKind::CPU_REFERENCE:    return "CPU_REFERENCE";
    case BackendKind::CPU_NEON:         return "CPU_NEON";
    case BackendKind::GPU_VULKAN:       return "GPU_VULKAN";
    case BackendKind::GPU_LITERT:       return "GPU_LITERT";
    case BackendKind::NPU_NNAPI:        return "NPU_NNAPI";
    case BackendKind::NPU_ENN:          return "NPU_ENN";
    case BackendKind::NPU_LITECORE:     return "NPU_LITECORE";
    case BackendKind::NPU_EXPERIMENTAL: return "NPU_EXPERIMENTAL";
    }
    return "UNKNOWN";
}

inline const char* to_string(CapabilityStatus status) {
    switch (status) {
    case CapabilityStatus::UNKNOWN:            return "UNKNOWN";
    case CapabilityStatus::DETECTED:           return "DETECTED";
    case CapabilityStatus::ACCESSIBLE:         return "ACCESSIBLE";
    case CapabilityStatus::INITIALIZED:        return "INITIALIZED";
    case CapabilityStatus::COMPILE_VERIFIED:   return "COMPILE_VERIFIED";
    case CapabilityStatus::EXECUTION_VERIFIED: return "EXECUTION_VERIFIED";
    case CapabilityStatus::UNSUPPORTED:        return "UNSUPPORTED";
    case CapabilityStatus::ACCESS_DENIED:      return "ACCESS_DENIED";
    case CapabilityStatus::EXPERIMENTAL:       return "EXPERIMENTAL";
    }
    return "UNKNOWN";
}

inline const char* to_string(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::CPU_ONLY: return "CPU_ONLY";
    case ExecutionMode::GPU_ONLY: return "GPU_ONLY";
    case ExecutionMode::NPU_ONLY: return "NPU_ONLY";
    case ExecutionMode::AUTO:     return "AUTO";
    case ExecutionMode::HYBRID:   return "HYBRID";
    }
    return "UNKNOWN";
}

inline bool is_capability_ready(CapabilityStatus status) {
    return status == CapabilityStatus::COMPILE_VERIFIED ||
           status == CapabilityStatus::EXECUTION_VERIFIED;
}

} // namespace runtime

using runtime::BackendInfo;
using runtime::BackendKind;
using runtime::CapabilityStatus;
using runtime::ExecutionMode;
using runtime::ExecutionReport;
using runtime::NodeExecutionTrace;
using runtime::to_string;

} // namespace exynova
