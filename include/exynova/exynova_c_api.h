#ifndef EXYNOVA_C_API_H
#define EXYNOVA_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(EXYNOVA_SHARED_EXPORTS)
#define EXYNOVA_C_API __declspec(dllexport)
#elif defined(_WIN32) && defined(EXYNOVA_SHARED_IMPORTS)
#define EXYNOVA_C_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define EXYNOVA_C_API __attribute__((visibility("default")))
#else
#define EXYNOVA_C_API
#endif

typedef struct exynova_context* exynova_context_t;
typedef struct exynova_compiled_model* exynova_compiled_model_t;
typedef struct exynova_session* exynova_session_t;

typedef enum exynova_status_t {
    EXYNOVA_STATUS_OK = 0,
    EXYNOVA_STATUS_INVALID_ARG = -1,
    EXYNOVA_STATUS_NOT_INITIALIZED = -2,
    EXYNOVA_STATUS_UNSUPPORTED = -3,
    EXYNOVA_STATUS_STRICT_REJECTED = -4,
    EXYNOVA_STATUS_IO_ERROR = -5,
    EXYNOVA_STATUS_CORRUPT_ARTIFACT = -6,
    EXYNOVA_STATUS_RUNTIME_ERROR = -7
} exynova_status_t;

typedef enum exynova_backend_kind_t {
    EXYNOVA_BACKEND_CPU_REFERENCE = 0,
    EXYNOVA_BACKEND_CPU_NEON = 1,
    EXYNOVA_BACKEND_GPU_VULKAN = 2,
    EXYNOVA_BACKEND_GPU_LITERT = 3,
    EXYNOVA_BACKEND_NPU_NNAPI = 4,
    EXYNOVA_BACKEND_NPU_ENN = 5,
    EXYNOVA_BACKEND_NPU_LITECORE = 6,
    EXYNOVA_BACKEND_NPU_EXPERIMENTAL = 7
} exynova_backend_kind_t;

typedef enum exynova_capability_status_t {
    EXYNOVA_CAP_UNKNOWN = 0,
    EXYNOVA_CAP_DETECTED = 1,
    EXYNOVA_CAP_ACCESSIBLE = 2,
    EXYNOVA_CAP_INITIALIZED = 3,
    EXYNOVA_CAP_COMPILE_VERIFIED = 4,
    EXYNOVA_CAP_EXECUTION_VERIFIED = 5,
    EXYNOVA_CAP_UNSUPPORTED = 6,
    EXYNOVA_CAP_ACCESS_DENIED = 7,
    EXYNOVA_CAP_EXPERIMENTAL = 8
} exynova_capability_status_t;

typedef enum exynova_execution_mode_t {
    EXYNOVA_MODE_CPU_ONLY = 0,
    EXYNOVA_MODE_GPU_ONLY = 1,
    EXYNOVA_MODE_NPU_ONLY = 2,
    EXYNOVA_MODE_AUTO = 3,
    EXYNOVA_MODE_HYBRID = 4
} exynova_execution_mode_t;

typedef struct exynova_backend_info_t {
    exynova_backend_kind_t kind;
    exynova_capability_status_t status;
    char name[64];
    char device_name[128];
    char details[256];
} exynova_backend_info_t;

typedef struct exynova_exec_report_t {
    exynova_execution_mode_t mode;
    int strict_mode;
    int any_fallback;
    double total_us;
    uint32_t trace_count;
    char error_msg[256];
} exynova_exec_report_t;

EXYNOVA_C_API const char* exynova_version_string(void);

EXYNOVA_C_API exynova_status_t exynova_initialize(exynova_context_t* out_ctx);

EXYNOVA_C_API void exynova_shutdown(exynova_context_t ctx);

EXYNOVA_C_API exynova_status_t exynova_list_backends(
    exynova_context_t ctx,
    exynova_backend_info_t* out_infos,
    uint32_t max_count,
    uint32_t* out_count);

EXYNOVA_C_API exynova_status_t exynova_compile_artifact(
    exynova_context_t ctx,
    const uint8_t* artifact_bytes,
    size_t byte_len,
    exynova_execution_mode_t mode,
    int strict_mode,
    exynova_compiled_model_t* out_model);

EXYNOVA_C_API exynova_status_t exynova_load_artifact(
    exynova_context_t ctx,
    const char* file_path,
    exynova_compiled_model_t* out_model);

EXYNOVA_C_API exynova_status_t exynova_create_session(
    exynova_compiled_model_t model,
    exynova_session_t* out_session);

EXYNOVA_C_API exynova_status_t exynova_set_input(
    exynova_session_t session,
    const char* tensor_name,
    const void* data,
    size_t byte_len);

EXYNOVA_C_API exynova_status_t exynova_run(exynova_session_t session);

EXYNOVA_C_API exynova_status_t exynova_get_output(
    exynova_session_t session,
    const char* tensor_name,
    void* out_data,
    size_t byte_len);

EXYNOVA_C_API exynova_status_t exynova_get_last_report(
    exynova_session_t session,
    exynova_exec_report_t* out_report);

EXYNOVA_C_API void exynova_destroy_session(exynova_session_t session);

EXYNOVA_C_API void exynova_destroy_model(exynova_compiled_model_t model);

#ifdef __cplusplus
}
#endif

#endif /* EXYNOVA_C_API_H */
