#include "exynova/enn_backend.h"
#include "exynova/gpu_probe.h"

#include <cstdio>

#ifdef EXYNOVA_ANDROID
#  undef VK_QUEUE_GRAPHICS_BIT
#  undef VK_QUEUE_COMPUTE_BIT
#  undef VK_QUEUE_TRANSFER_BIT
#  undef VK_QUEUE_SPARSE_BINDING_BIT
#  include <vulkan/vulkan.h>
#endif

int main() {
    std::printf("=== ExyNova Hardware Probe ===\n");

    auto ocl = exynova::gpu::probe_opencl();
    std::printf("[OpenCL] loaded=%s path=%s platforms=%u gpus=%u\n",
                ocl.library_loaded ? "true" : "false",
                ocl.library_path.c_str(),
                ocl.platform_count,
                ocl.gpu_device_count);
    if (!ocl.device_name.empty()) {
        std::printf("  device=%s (%s)\n", ocl.device_name.c_str(), ocl.driver_version.c_str());
    } else if (!ocl.error_msg.empty()) {
        std::printf("  note=%s\n", ocl.error_msg.c_str());
    }

    auto nnapi = exynova::npu::probe_nnapi();
    std::printf("[NNAPI] loaded=%s devices=%zu\n",
                nnapi.library_loaded ? "true" : "false",
                nnapi.devices.size());
    for (const auto& d : nnapi.devices) {
        std::printf("  device=%s type=%s ver=%s\n",
                    d.name.c_str(), d.type_name.c_str(), d.version.c_str());
    }

    exynova::npu::EnnNpuBackend enn;
    auto enn_rep = enn.probe_and_initialize(false);
    std::printf("[ENN NPU] lib_loaded=%s vertex10=%s initialized=%s version=%s\n",
                enn_rep.library_loaded ? "true" : "false",
                enn_rep.vertex10_exists ? "true" : "false",
                enn_rep.initialized ? "true" : "false",
                enn_rep.enn_version.c_str());
    if (!enn_rep.error_msg.empty()) {
        std::printf("  note=%s\n", enn_rep.error_msg.c_str());
    }

#ifdef EXYNOVA_ANDROID
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "exynova_vulkan_probe";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) == VK_SUCCESS && instance) {
        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        std::printf("[Vulkan] physical_devices=%u\n", dev_count);
        if (dev_count > 0) {
            std::vector<VkPhysicalDevice> devs(dev_count);
            vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());
            auto res = exynova::gpu::probe_device_capabilities(devs[0]);
            if (res.status == exynova::gpu::ProbeResult::Status::SUCCESS) {
                const auto& c = res.caps;
                std::printf("  device=%s driver=%s subgroup=%u shared_mem=%u B fp16_int8=%s\n",
                            c.device_name.c_str(),
                            c.driver_info.c_str(),
                            c.subgroup_size,
                            c.max_shared_memory_size,
                            c.supports_float16_int8 ? "true" : "false");
            }
        }
        vkDestroyInstance(instance, nullptr);
    }
#else
    auto res = exynova::gpu::probe_first_compute_device(nullptr, nullptr);
    std::printf("[Vulkan] host mode status=%d (%s)\n",
                static_cast<int>(res.status), res.error_msg.c_str());
#endif
    return 0;
}
