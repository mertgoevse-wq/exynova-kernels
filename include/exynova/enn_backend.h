#pragma once

#include "exynova/backend_registry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace exynova {
namespace npu {

// Verified Samsung ENN ABI types (reverse-engineered from A56 libenn_public_api_cpp.so
// ENN_VERSION_2.3.30 and cross-checked with libenn_user.samsung_slsi.so).
enum class EnnStatus : int32_t {
    SUCCESS = 0,
    FAILED = 1,
    IO = 2,
    INVAL = 3,
    FILTERED = 4,
    CANCELED = 5,
    MEM_ERR = 6,
    SIZE = 7,
    TIMEOUT_ENN = 10,
    TIMEOUT_DD = 11,
    TIMEOUT_FW = 12,
    TIMEOUT_HW_NOTRECOVERED = 13,
    TIMEOUT_HW_RECOVERED = 14,
    SERVICE_NULL = 15,
    RESOURCE_BUSY = 16,
    NOT_SUPPORTED = 0xFF
};

inline const char* enn_status_to_string(int32_t code) {
    switch (static_cast<EnnStatus>(code)) {
    case EnnStatus::SUCCESS:                return "ENN_RET_SUCCESS";
    case EnnStatus::FAILED:                 return "ENN_RET_FAILED";
    case EnnStatus::IO:                     return "ENN_RET_IO";
    case EnnStatus::INVAL:                  return "ENN_RET_INVAL";
    case EnnStatus::FILTERED:               return "ENN_RET_FILTERED";
    case EnnStatus::CANCELED:               return "ENN_RET_CANCELED";
    case EnnStatus::MEM_ERR:                return "ENN_RET_MEM_ERR";
    case EnnStatus::SIZE:                   return "ENN_RET_SIZE";
    case EnnStatus::TIMEOUT_ENN:            return "ENN_RET_FAILED_TIMEOUT_ENN";
    case EnnStatus::TIMEOUT_DD:             return "ENN_RET_FAILED_TIMEOUT_DD";
    case EnnStatus::TIMEOUT_FW:             return "ENN_RET_FAILED_TIMEOUT_FW";
    case EnnStatus::TIMEOUT_HW_NOTRECOVERED:return "ENN_RET_FAILED_TIMEOUT_HW_NOTRECOVERED";
    case EnnStatus::TIMEOUT_HW_RECOVERED:   return "ENN_RET_FAILED_TIMEOUT_HW_RECOVERED";
    case EnnStatus::SERVICE_NULL:           return "ENN_RET_FAILED_SERVICE_NULL";
    case EnnStatus::RESOURCE_BUSY:          return "ENN_RET_FAILED_RESOURCE_BUSY";
    case EnnStatus::NOT_SUPPORTED:          return "ENN_RET_NOT_SUPPORTED";
    }
    return "ENN_RET_UNKNOWN";
}

struct NnapiDeviceDesc {
    std::string name;
    int32_t type = 0; // ANeuralNetworksDeviceType
    std::string type_name;
    std::string version;
};

struct NnapiProbeReport {
    bool library_loaded = false;
    std::string library_path;
    std::string error_msg;
    std::vector<NnapiDeviceDesc> devices;
};

struct EnnTensorDesc {
    uint32_t index = 0;
    bool is_output = false;
    std::string label;
    uint32_t n = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channel = 0;
    uint32_t size_bytes = 0;
    uint32_t buffer_type = 0;
};

struct EnnProbeReport {
    runtime::CapabilityStatus status = runtime::CapabilityStatus::UNKNOWN;
    bool vertex10_exists = false;
    bool vertex10_rw = false;
    bool library_loaded = false;
    bool symbols_resolved = false;
    bool initialized = false;
    std::string library_path;
    std::string enn_version;
    std::string meta_version_302;
    int32_t init_ret_code = -1;
    std::string error_msg;
};

struct EnnModelRunReport {
    bool opened = false;
    bool buffers_allocated = false;
    bool executed = false;
    uint64_t model_id = 0;
    int32_t last_ret_code = -1;
    std::string last_ret_name;
    double open_ms = 0.0;
    double exec_ms = 0.0;
    std::vector<EnnTensorDesc> inputs;
    std::vector<EnnTensorDesc> outputs;
    uint64_t first_output_fnv1a = 0;
    bool first_output_nonzero = false;
    std::string error_msg;
};

NnapiProbeReport probe_nnapi();

class EnnNpuBackend {
public:
    EnnNpuBackend();
    ~EnnNpuBackend();

    EnnNpuBackend(const EnnNpuBackend&) = delete;
    EnnNpuBackend& operator=(const EnnNpuBackend&) = delete;

    EnnProbeReport probe_and_initialize(bool set_perf_mode = false);
    void shutdown();

    bool is_initialized() const { return initialized_; }
    const EnnProbeReport& last_probe() const { return probe_report_; }

    EnnModelRunReport inspect_and_run_model(const std::string& nnc_path,
                                            bool execute_inference = true,
                                            uint32_t perf_preset_id = 0);

    bool open_session(const std::string& nnc_path, std::string* err = nullptr,
                      uint32_t perf_preset_id = 0);
    bool execute_session(const void* in_bytes, size_t in_len,
                         std::vector<uint8_t>* out_bytes,
                         double* exec_ms = nullptr,
                         std::string* err = nullptr);
    void close_session();
    bool has_open_session() const;
    size_t session_input_size() const;
    size_t session_output_size() const;

private:
    struct EnnSymbols;
    struct OpenSessionState;
    EnnSymbols* syms_ = nullptr;
    OpenSessionState* session_ = nullptr;
    void* lib_handle_ = nullptr;
    bool initialized_ = false;
    EnnProbeReport probe_report_;
};

} // namespace npu
} // namespace exynova
