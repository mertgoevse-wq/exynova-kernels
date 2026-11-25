#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal Vulkan compatibility layer for host builds (tests only).
// On Android the real vulkan.h is linked via the NDK.
// ---------------------------------------------------------------------------
#ifndef VK_MAKE_VERSION
#define VK_MAKE_VERSION(major, minor, patch) (((major) << 22) | ((minor) << 12) | (patch))
#endif
#ifndef VK_QUEUE_GRAPHICS_BIT
#define VK_QUEUE_GRAPHICS_BIT 0x00000001u
#endif
#ifndef VK_QUEUE_COMPUTE_BIT
#define VK_QUEUE_COMPUTE_BIT 0x00000002u
#endif
#ifndef VK_QUEUE_TRANSFER_BIT
#define VK_QUEUE_TRANSFER_BIT 0x00000004u
#endif
#ifndef VK_QUEUE_SPARSE_BINDING_BIT
#define VK_QUEUE_SPARSE_BINDING_BIT 0x00000008u
#endif
#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif

namespace exynova {
namespace gpu {

// =============================================================================
// Data types — device characteristics derived from Vulkan probing
// =============================================================================

struct QueueFamilyInfo {
    uint32_t    index          = 0;
    uint32_t    queue_count    = 0;
    float       queue_time     = 0.0f;   // timestamp granularity (seconds)
    bool        graphics       = false;
    bool        compute        = false;
    bool        transfer       = false;
    bool        sparse_binding = false;
};

struct ExtensionDesc {
    std::string name;
    uint32_t    spec_version = 0;
};

struct PhysicalDeviceCaps {
    std::string                        device_name;
    std::string                        driver_version;
    std::string                        driver_info;
    uint32_t                           api_version      = 0;
    uint32_t                           hardware_version  = 0;
    uint32_t                           vendor_id         = 0;
    uint32_t                           device_id         = 0;
    uint32_t                           pipeline_cache_uuid[4] = {};
    uint32_t                           max_workgroup_count[3] = {};
    uint32_t                           max_workgroup_size[3]  = {};
    uint32_t                           max_workgroup_invocations = 0;
    uint32_t                           max_shared_memory_size    = 0;
    uint32_t                           max_storage_buffer_range  = 0;
    uint32_t                           subgroup_size             = 0;
    float                              timestamp_period_ns       = 0.0f;
    uint64_t                           device_local_memory_bytes = 0;
    uint64_t                           host_visible_memory_bytes = 0;
    std::vector<QueueFamilyInfo>       queue_families;
    std::vector<ExtensionDesc>         extensions;
    bool                               supports_float16_int8 = false;
    bool                               supports_shader_core_props = false;
    bool                               has_compute_queue   = false;
};

// =============================================================================
// Probe result — aggregates all findings
// =============================================================================

struct ProbeResult {
    enum class Status : int {
        SUCCESS          =  0,
        NO_DEVICE        = -1,
        VERSION_TOO_LOW  = -2,
        NO_COMPUTE_QUEUE = -3,
        MISSING_FLOAT16  = -4,
        GENERIC_ERROR    = -5,
    };

    Status        status     = Status::GENERIC_ERROR;
    PhysicalDeviceCaps caps;
    std::string   error_msg;
};

// =============================================================================
// Public probe API — all functions take platform handles
// =============================================================================

// Enumerates physical devices and returns the first compute-capable one.
ProbeResult probe_first_compute_device(void* instance, void* physical_device);

// Queries detailed capabilities from a physical device handle.
ProbeResult probe_device_capabilities(void* physical_device);

struct OpenClProbeReport {
    bool library_loaded = false;
    std::string library_path;
    uint32_t platform_count = 0;
    uint32_t gpu_device_count = 0;
    std::string platform_name;
    std::string platform_version;
    std::string device_name;
    std::string driver_version;
    std::string error_msg;
};

// Probes OpenCL runtime availability (e.g. libOpenCL.so / libSGPUOpenCL.so on Xclipse).
OpenClProbeReport probe_opencl();

}  // namespace gpu
}  // namespace exynova
