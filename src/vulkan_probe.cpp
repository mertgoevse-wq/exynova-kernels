#include "exynova/gpu_probe.h"

#ifdef EXYNOVA_ANDROID
// Stub-Makros aus gpu_probe.h dürfen die echten Enum-Konstanten in vulkan_core.h
// nicht beschatten — vor dem Include entfernen.
#  undef VK_QUEUE_GRAPHICS_BIT
#  undef VK_QUEUE_COMPUTE_BIT
#  undef VK_QUEUE_TRANSFER_BIT
#  undef VK_QUEUE_SPARSE_BINDING_BIT
#  include <vulkan/vulkan.h>
#endif

#include <cstring>

namespace exynova {
namespace gpu {
namespace {

constexpr uint32_t kMinApiVersion = VK_MAKE_VERSION(1, 2, 0);

}  // anonymous namespace

// ----------------------------------------------------------------------------
// probe_first_compute_device
// ----------------------------------------------------------------------------
ProbeResult probe_first_compute_device(void* instance, void* physical_device_handle) {
    ProbeResult result;
#ifndef EXYNOVA_ANDROID
    // Host mode: no GPU available — return error
    (void)instance;
    (void)physical_device_handle;
    result.status = ProbeResult::Status::NO_DEVICE;
    result.error_msg = "probe_first_compute_device not available on host (EXYNOVA_ANDROID not defined)";
    return result;
#else
    if (!physical_device_handle) {
        result.status = ProbeResult::Status::NO_DEVICE;
        result.error_msg = "Null physical device handle";
        return result;
    }

    VkPhysicalDevice phys_dev = reinterpret_cast<VkPhysicalDevice>(physical_device_handle);

    // Query basic properties
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_dev, &props);
    result.caps.device_name      = props.deviceName;
    result.caps.api_version      = props.apiVersion;
    result.caps.vendor_id        = props.vendorID;
    result.caps.device_id        = props.deviceID;
    result.caps.hardware_version = props.driverVersion;
    std::memcpy(result.caps.pipeline_cache_uuid, props.pipelineCacheUUID,
                sizeof(props.pipelineCacheUUID));

    for (uint32_t i = 0; i < 3; ++i) {
        result.caps.max_workgroup_count[i] = props.limits.maxComputeWorkGroupCount[i];
        result.caps.max_workgroup_size[i]  = props.limits.maxComputeWorkGroupSize[i];
    }
    result.caps.max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    result.caps.max_shared_memory_size    = props.limits.maxComputeSharedMemorySize;
    result.caps.max_storage_buffer_range  = props.limits.maxStorageBufferRange;
    result.caps.timestamp_period_ns       = props.limits.timestampPeriod;

    // Query extended properties (Vulkan 1.1 subgroup + Vulkan 1.2 driver info)
    PFN_vkGetPhysicalDeviceProperties2 pfn_props2 = nullptr;
    if (instance) {
        pfn_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            vkGetInstanceProcAddr(reinterpret_cast<VkInstance>(instance),
                                  "vkGetPhysicalDeviceProperties2"));
    }
#if defined(__ANDROID_API__) && __ANDROID_API__ >= 29
    if (!pfn_props2) {
        pfn_props2 = &vkGetPhysicalDeviceProperties2;
    }
#endif
    if (pfn_props2 && result.caps.api_version >= VK_MAKE_VERSION(1, 1, 0)) {
        VkPhysicalDeviceSubgroupProperties subgroup_props{};
        subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceDriverProperties driver_props{};
        driver_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

        if (result.caps.api_version >= VK_MAKE_VERSION(1, 2, 0)) {
            subgroup_props.pNext = &driver_props;
        }

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup_props;

        pfn_props2(phys_dev, &props2);

        result.caps.subgroup_size = subgroup_props.subgroupSize;
        if (result.caps.api_version >= VK_MAKE_VERSION(1, 2, 0)) {
            result.caps.driver_info = driver_props.driverInfo;
            if (result.caps.driver_version.empty()) {
                result.caps.driver_version =
                    std::to_string(driver_props.conformanceVersion.major) + "." +
                    std::to_string(driver_props.conformanceVersion.minor) + "." +
                    std::to_string(driver_props.conformanceVersion.subminor) + "." +
                    std::to_string(driver_props.conformanceVersion.patch);
            }
        }
    }

    // Query memory heap capacities
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

    bool host_visible_heaps[VK_MAX_MEMORY_HEAPS] = {};
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            uint32_t heap_idx = mem_props.memoryTypes[i].heapIndex;
            if (heap_idx < VK_MAX_MEMORY_HEAPS) {
                host_visible_heaps[heap_idx] = true;
            }
        }
    }
    for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
        if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            result.caps.device_local_memory_bytes += mem_props.memoryHeaps[h].size;
        }
        if (host_visible_heaps[h]) {
            result.caps.host_visible_memory_bytes += mem_props.memoryHeaps[h].size;
        }
    }

    // Validate API version
    if (result.caps.api_version < kMinApiVersion) {
        result.status = ProbeResult::Status::VERSION_TOO_LOW;
        result.error_msg = "Vulkan API version " + std::to_string(VK_VERSION_MAJOR(result.caps.api_version))
                         + "." + std::to_string(VK_VERSION_MINOR(result.caps.api_version))
                         + " below required minimum 1.2";
        return result;
    }

    // Collect queue families
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, qf_props.data());

    for (uint32_t i = 0; i < qf_count; ++i) {
        QueueFamilyInfo qfi;
        qfi.index          = i;
        qfi.queue_count    = qf_props[i].queueCount;
        qfi.queue_time     = static_cast<float>(qf_props[i].timestampValidBits) * 1e-9f;
        qfi.graphics       = !!(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
        qfi.compute        = !!(qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT);
        qfi.transfer       = !!(qf_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT);
        qfi.sparse_binding = !!(qf_props[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
        result.caps.queue_families.push_back(qfi);

        if (qfi.compute) result.caps.has_compute_queue = true;
    }

    if (!result.caps.has_compute_queue) {
        result.status = ProbeResult::Status::NO_COMPUTE_QUEUE;
        result.error_msg = "No queue family supports VK_QUEUE_COMPUTE_BIT";
        return result;
    }

    // Check extensions
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(phys_dev, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(phys_dev, nullptr, &ext_count, exts.data());

    for (const auto& ext : exts) {
        std::string name(ext.extensionName);
        result.caps.extensions.push_back({name, ext.specVersion});
        if (name == "VK_KHR_shader_float16_int8")
            result.caps.supports_float16_int8 = true;
        if (name == "VK_EXT_shader_core_properties" || name == "VK_AMD_shader_core_properties")
            result.caps.supports_shader_core_props = true;
    }

    // Validate required extensions
    if (!result.caps.supports_float16_int8) {
        result.status = ProbeResult::Status::MISSING_FLOAT16;
        result.error_msg = "VK_KHR_shader_float16_int8 not available";
        return result;
    }

    result.status = ProbeResult::Status::SUCCESS;
    return result;
#endif
}

#ifdef EXYNOVA_ANDROID
#  include <dlfcn.h>
#endif

// ----------------------------------------------------------------------------
// probe_device_capabilities
// ----------------------------------------------------------------------------
ProbeResult probe_device_capabilities(void* physical_device_handle) {
    return probe_first_compute_device(nullptr, physical_device_handle);
}

OpenClProbeReport probe_opencl() {
    OpenClProbeReport rep;
#ifndef EXYNOVA_ANDROID
    rep.error_msg = "OpenCL device probe runs on Android target";
    return rep;
#else
    const char* candidates[] = {
        "/vendor/lib64/libOpenCL.so",
        "libOpenCL.so"
    };
    void* handle = nullptr;
    std::string errs;
    for (const char* p : candidates) {
        dlerror();
        handle = dlopen(p, RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            rep.library_loaded = true;
            rep.library_path = p;
            break;
        }
        const char* e = dlerror();
        errs += std::string(p) + ": " + (e ? e : "unknown") + "; ";
    }
    if (!handle) {
        rep.error_msg = errs;
        return rep;
    }

    using cl_int = int32_t;
    using cl_uint = uint32_t;
    using cl_ulong = uint64_t;
    using cl_platform_id = void*;
    using cl_device_id = void*;

    using FnGetPlatforms = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
    using FnGetPlatInfo = cl_int (*)(cl_platform_id, cl_uint, size_t, void*, size_t*);
    using FnGetDevices = cl_int (*)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
    using FnGetDevInfo = cl_int (*)(cl_device_id, cl_uint, size_t, void*, size_t*);

    auto get_plats = reinterpret_cast<FnGetPlatforms>(dlsym(handle, "clGetPlatformIDs"));
    auto get_pinfo = reinterpret_cast<FnGetPlatInfo>(dlsym(handle, "clGetPlatformInfo"));
    auto get_devs = reinterpret_cast<FnGetDevices>(dlsym(handle, "clGetDeviceIDs"));
    auto get_dinfo = reinterpret_cast<FnGetDevInfo>(dlsym(handle, "clGetDeviceInfo"));

    if (!get_plats || !get_devs) {
        rep.error_msg = "OpenCL entry points missing";
        dlclose(handle);
        return rep;
    }

    cl_uint n_plat = 0;
    cl_int rc = get_plats(0, nullptr, &n_plat);
    if (rc != 0 || n_plat == 0) {
        rep.error_msg = "clGetPlatformIDs returned rc=" + std::to_string(rc) +
                        " count=" + std::to_string(n_plat);
        dlclose(handle);
        return rep;
    }
    rep.platform_count = n_plat;
    std::vector<cl_platform_id> plats(n_plat);
    get_plats(n_plat, plats.data(), nullptr);

    char buf[256] = {};
    if (get_pinfo) {
        if (get_pinfo(plats[0], 0x0902 /*CL_PLATFORM_NAME*/, sizeof(buf), buf, nullptr) == 0)
            rep.platform_name = buf;
        std::memset(buf, 0, sizeof(buf));
        if (get_pinfo(plats[0], 0x0901 /*CL_PLATFORM_VERSION*/, sizeof(buf), buf, nullptr) == 0)
            rep.platform_version = buf;
    }

    cl_uint n_dev = 0;
    if (get_devs(plats[0], 4 /*CL_DEVICE_TYPE_GPU*/, 0, nullptr, &n_dev) == 0 && n_dev > 0) {
        rep.gpu_device_count = n_dev;
        std::vector<cl_device_id> devs(n_dev);
        get_devs(plats[0], 4, n_dev, devs.data(), nullptr);
        if (get_dinfo) {
            std::memset(buf, 0, sizeof(buf));
            if (get_dinfo(devs[0], 0x102B /*CL_DEVICE_NAME*/, sizeof(buf), buf, nullptr) == 0)
                rep.device_name = buf;
            std::memset(buf, 0, sizeof(buf));
            if (get_dinfo(devs[0], 0x102D /*CL_DRIVER_VERSION*/, sizeof(buf), buf, nullptr) == 0)
                rep.driver_version = buf;
        }
    }

    dlclose(handle);
    return rep;
#endif
}

}  // namespace gpu
}  // namespace exynova
