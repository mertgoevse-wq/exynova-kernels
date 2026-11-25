#ifdef EXYNOVA_ANDROID
#  undef VK_QUEUE_GRAPHICS_BIT
#  undef VK_QUEUE_COMPUTE_BIT
#  undef VK_QUEUE_TRANSFER_BIT
#  undef VK_QUEUE_SPARSE_BINDING_BIT
#  include <vulkan/vulkan.h>
#endif

#include "exynova/vulkan_backend.h"

#ifdef EXYNOVA_ANDROID
#  include "../generated/shaders_spv.h"
#endif

#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace exynova {
namespace gpu {

namespace {

inline void set_error(std::string* err, const std::string& msg) {
    if (err) {
        *err = msg;
    }
}

}  // namespace

#ifndef EXYNOVA_ANDROID

struct VulkanComputeBackend::Impl {
    bool initialized = false;
};

VulkanComputeBackend::VulkanComputeBackend() : impl_(std::make_unique<Impl>()) {}
VulkanComputeBackend::~VulkanComputeBackend() = default;

bool VulkanComputeBackend::initialize(std::string* err) {
    set_error(err, "VulkanComputeBackend is only supported on Android (EXYNOVA_ANDROID not defined)");
    return false;
}

void VulkanComputeBackend::shutdown() {
    if (impl_) {
        impl_->initialized = false;
    }
    caps_ = PhysicalDeviceCaps{};
}

void VulkanComputeBackend::clear_weight_cache() {
}

bool VulkanComputeBackend::is_initialized() const {
    return impl_ && impl_->initialized;
}

const PhysicalDeviceCaps& VulkanComputeBackend::device_caps() const {
    return caps_;
}

bool VulkanComputeBackend::add(const float*, const float*, float*, size_t, double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::mul(const float*, const float*, float*, size_t, double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::relu(const float*, float*, size_t, double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::silu(const float*, float*, size_t, double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::softmax(const float*, float*, uint32_t, uint32_t, double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::matmul(const float*, const float*, float*, uint32_t, uint32_t, uint32_t, double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::rmsnorm(const float*, const float*, float*, uint32_t, uint32_t, float, double*, std::string* err, bool) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::rope(const float*, float*, uint32_t, uint32_t, uint32_t, float, double*, std::string* err, uint32_t) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::matvec_q8_0(const int32_t*, const float*, const float*, float*, uint32_t, uint32_t, double*, std::string* err, bool) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::matvec_q4_0(const uint32_t*, const float*, const float*, float*, uint32_t, uint32_t, double*, std::string* err, bool) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::matvec_f16(const uint32_t*, const float*, float*, uint32_t, uint32_t, double*, std::string* err, bool) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

bool VulkanComputeBackend::conv2d_nchw(const float*, const float*, const float*, float*,
                                       uint32_t, uint32_t, uint32_t, uint32_t,
                                       uint32_t, uint32_t, uint32_t,
                                       uint32_t, uint32_t, uint32_t, uint32_t,
                                       double*, std::string* err) {
    set_error(err, "VulkanComputeBackend is not available on host");
    return false;
}

#else  // EXYNOVA_ANDROID

namespace {

enum class KernelId : uint32_t {
    Add = 0,
    Mul,
    Relu,
    Silu,
    Softmax,
    MatMul,
    RMSNorm,
    RoPE,
    MatVecQ8_0,
    MatVecQ4_0,
    MatVecF16,
    Conv2D,
};

struct KernelSpec {
    const uint32_t* code;
    size_t          code_bytes;
    uint32_t        num_buffers;
    uint32_t        push_const_bytes;
};

KernelSpec get_kernel_spec(KernelId id) {
    switch (id) {
        case KernelId::Add:
            return {spv::kShader_vec_add, sizeof(spv::kShader_vec_add), 3, sizeof(uint32_t)};
        case KernelId::Mul:
            return {spv::kShader_vec_mul, sizeof(spv::kShader_vec_mul), 3, sizeof(uint32_t)};
        case KernelId::Relu:
            return {spv::kShader_relu, sizeof(spv::kShader_relu), 2, sizeof(uint32_t)};
        case KernelId::Silu:
            return {spv::kShader_silu, sizeof(spv::kShader_silu), 2, sizeof(uint32_t)};
        case KernelId::Softmax:
            return {spv::kShader_softmax, sizeof(spv::kShader_softmax), 2, 2 * sizeof(uint32_t)};
        case KernelId::MatMul:
            return {spv::kShader_matmul_f32, sizeof(spv::kShader_matmul_f32), 3, 3 * sizeof(uint32_t)};
        case KernelId::RMSNorm:
            return {spv::kShader_rmsnorm, sizeof(spv::kShader_rmsnorm), 3, 2 * sizeof(uint32_t)};
        case KernelId::RoPE:
            return {spv::kShader_rope_f32, sizeof(spv::kShader_rope_f32), 2, 5 * sizeof(uint32_t)};
        case KernelId::MatVecQ8_0:
            return {spv::kShader_matvec_q8_0, sizeof(spv::kShader_matvec_q8_0), 4, 2 * sizeof(uint32_t)};
        case KernelId::MatVecQ4_0:
            return {spv::kShader_matvec_q4_0, sizeof(spv::kShader_matvec_q4_0), 4, 2 * sizeof(uint32_t)};
        case KernelId::MatVecF16:
            return {spv::kShader_matvec_f16, sizeof(spv::kShader_matvec_f16), 3, 2 * sizeof(uint32_t)};
        case KernelId::Conv2D:
            return {spv::kShader_conv2d_f32, sizeof(spv::kShader_conv2d_f32), 4, 13 * sizeof(uint32_t)};
    }
    return {nullptr, 0, 0, 0};
}

struct KernelPipeline {
    VkShaderModule        shader_module   = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            pipeline        = VK_NULL_HANDLE;
    uint32_t              num_buffers     = 0;
    uint32_t              push_const_bytes = 0;
};

struct GpuBuffer {
    VkBuffer       buffer   = VK_NULL_HANDLE;
    VkDeviceMemory memory   = VK_NULL_HANDLE;
    VkDeviceSize   size     = 0;
    bool           coherent = true;
};

struct BufferBinding {
    const void*  upload_data   = nullptr;
    void*        download_data = nullptr;
    VkDeviceSize size_bytes    = 0;
    bool         cache_upload  = false;
};

}  // namespace

struct VulkanComputeBackend::Impl {
    bool                             initialized          = false;
    VkInstance                       instance             = VK_NULL_HANDLE;
    VkPhysicalDevice                 phys_dev             = VK_NULL_HANDLE;
    VkDevice                         device               = VK_NULL_HANDLE;
    VkQueue                          queue                = VK_NULL_HANDLE;
    uint32_t                         compute_queue_family = UINT32_MAX;
    uint32_t                         timestamp_valid_bits = 0;
    VkPhysicalDeviceMemoryProperties mem_props{};
    VkCommandPool                    cmd_pool             = VK_NULL_HANDLE;
    VkCommandBuffer                  cmd_buf              = VK_NULL_HANDLE;
    VkDescriptorPool                 desc_pool            = VK_NULL_HANDLE;
    VkFence                          fence                = VK_NULL_HANDLE;
    VkQueryPool                      query_pool           = VK_NULL_HANDLE;
    float                            timestamp_period_ns  = 0.0f;
    std::unordered_map<uint32_t, KernelPipeline> pipelines;
    std::unordered_map<const void*, GpuBuffer>   weight_cache;
    VkDeviceSize                                 cached_bytes = 0;

    void clear_weight_cache() {
        if (device != VK_NULL_HANDLE) {
            for (auto& kv : weight_cache) {
                destroy_buffer(kv.second);
            }
        }
        weight_cache.clear();
        cached_bytes = 0;
    }

    void destroy_all() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            clear_weight_cache();
            for (auto& kv : pipelines) {
                KernelPipeline& kp = kv.second;
                if (kp.pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, kp.pipeline, nullptr);
                if (kp.pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, kp.pipeline_layout, nullptr);
                if (kp.desc_set_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, kp.desc_set_layout, nullptr);
                if (kp.shader_module != VK_NULL_HANDLE)
                    vkDestroyShaderModule(device, kp.shader_module, nullptr);
            }
            pipelines.clear();

            if (query_pool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device, query_pool, nullptr);
                query_pool = VK_NULL_HANDLE;
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
                fence = VK_NULL_HANDLE;
            }
            if (desc_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, desc_pool, nullptr);
                desc_pool = VK_NULL_HANDLE;
            }
            if (cmd_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, cmd_pool, nullptr);
                cmd_pool = VK_NULL_HANDLE;
                cmd_buf = VK_NULL_HANDLE;
            }
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        phys_dev = VK_NULL_HANDLE;
        queue = VK_NULL_HANDLE;
        compute_queue_family = UINT32_MAX;
        timestamp_valid_bits = 0;
        timestamp_period_ns = 0.0f;
        initialized = false;
    }

    uint32_t find_memory_type(uint32_t type_bits, bool* out_coherent) const {
        const VkMemoryPropertyFlags ideal =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & ideal) == ideal) {
                *out_coherent = true;
                return i;
            }
        }
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                *out_coherent = false;
                return i;
            }
        }
        return UINT32_MAX;
    }

    bool create_buffer(VkDeviceSize size_bytes, GpuBuffer& out_buf, std::string* err) {
        out_buf = GpuBuffer{};
        out_buf.size = size_bytes;

        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = size_bytes;
        buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &out_buf.buffer);
        if (res != VK_SUCCESS) {
            set_error(err, "vkCreateBuffer failed: " + std::to_string(res));
            return false;
        }

        VkMemoryRequirements mem_req{};
        vkGetBufferMemoryRequirements(device, out_buf.buffer, &mem_req);

        uint32_t mem_type = find_memory_type(mem_req.memoryTypeBits, &out_buf.coherent);
        if (mem_type == UINT32_MAX) {
            vkDestroyBuffer(device, out_buf.buffer, nullptr);
            out_buf.buffer = VK_NULL_HANDLE;
            set_error(err, "No HOST_VISIBLE memory type found for storage buffer");
            return false;
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = mem_type;

        res = vkAllocateMemory(device, &alloc_info, nullptr, &out_buf.memory);
        if (res != VK_SUCCESS) {
            vkDestroyBuffer(device, out_buf.buffer, nullptr);
            out_buf.buffer = VK_NULL_HANDLE;
            set_error(err, "vkAllocateMemory failed: " + std::to_string(res));
            return false;
        }

        res = vkBindBufferMemory(device, out_buf.buffer, out_buf.memory, 0);
        if (res != VK_SUCCESS) {
            vkFreeMemory(device, out_buf.memory, nullptr);
            vkDestroyBuffer(device, out_buf.buffer, nullptr);
            out_buf.memory = VK_NULL_HANDLE;
            out_buf.buffer = VK_NULL_HANDLE;
            set_error(err, "vkBindBufferMemory failed: " + std::to_string(res));
            return false;
        }

        return true;
    }

    void destroy_buffer(GpuBuffer& buf) {
        if (buf.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buf.buffer, nullptr);
            buf.buffer = VK_NULL_HANDLE;
        }
        if (buf.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, buf.memory, nullptr);
            buf.memory = VK_NULL_HANDLE;
        }
    }

    const KernelPipeline* get_or_create_pipeline(KernelId id, std::string* err) {
        const uint32_t key = static_cast<uint32_t>(id);
        auto it = pipelines.find(key);
        if (it != pipelines.end()) {
            return &it->second;
        }

        KernelSpec spec = get_kernel_spec(id);
        if (!spec.code || spec.code_bytes == 0) {
            set_error(err, "Invalid kernel specification");
            return nullptr;
        }

        KernelPipeline kp;
        kp.num_buffers = spec.num_buffers;
        kp.push_const_bytes = spec.push_const_bytes;

        auto cleanup_partial = [&]() {
            if (kp.pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device, kp.pipeline, nullptr);
            if (kp.pipeline_layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device, kp.pipeline_layout, nullptr);
            if (kp.desc_set_layout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(device, kp.desc_set_layout, nullptr);
            if (kp.shader_module != VK_NULL_HANDLE)
                vkDestroyShaderModule(device, kp.shader_module, nullptr);
        };

        VkShaderModuleCreateInfo sm_info{};
        sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_info.codeSize = spec.code_bytes;
        sm_info.pCode = spec.code;

        VkResult res = vkCreateShaderModule(device, &sm_info, nullptr, &kp.shader_module);
        if (res != VK_SUCCESS) {
            set_error(err, "vkCreateShaderModule failed: " + std::to_string(res));
            return nullptr;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings(spec.num_buffers);
        for (uint32_t i = 0; i < spec.num_buffers; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dsl_info{};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = spec.num_buffers;
        dsl_info.pBindings = bindings.data();

        res = vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &kp.desc_set_layout);
        if (res != VK_SUCCESS) {
            cleanup_partial();
            set_error(err, "vkCreateDescriptorSetLayout failed: " + std::to_string(res));
            return nullptr;
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = spec.push_const_bytes;

        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount = 1;
        pl_info.pSetLayouts = &kp.desc_set_layout;
        if (spec.push_const_bytes > 0) {
            pl_info.pushConstantRangeCount = 1;
            pl_info.pPushConstantRanges = &push_range;
        }

        res = vkCreatePipelineLayout(device, &pl_info, nullptr, &kp.pipeline_layout);
        if (res != VK_SUCCESS) {
            cleanup_partial();
            set_error(err, "vkCreatePipelineLayout failed: " + std::to_string(res));
            return nullptr;
        }

        VkComputePipelineCreateInfo cp_info{};
        cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_info.stage.module = kp.shader_module;
        cp_info.stage.pName = "main";
        cp_info.layout = kp.pipeline_layout;

        res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &kp.pipeline);
        if (res != VK_SUCCESS) {
            cleanup_partial();
            set_error(err, "vkCreateComputePipelines failed: " + std::to_string(res));
            return nullptr;
        }

        auto inserted = pipelines.emplace(key, kp);
        return &inserted.first->second;
    }

    bool dispatch_kernel(KernelId id,
                         const BufferBinding* bindings,
                         uint32_t binding_count,
                         const void* push_constants,
                         uint32_t push_size,
                         uint32_t gx,
                         uint32_t gy,
                         uint32_t gz,
                         double* gpu_us,
                         std::string* err) {
        if (!initialized) {
            set_error(err, "VulkanComputeBackend is not initialized");
            return false;
        }

        const KernelPipeline* kp = get_or_create_pipeline(id, err);
        if (!kp) {
            return false;
        }
        if (binding_count != kp->num_buffers) {
            set_error(err, "Buffer binding count mismatch");
            return false;
        }

        std::vector<GpuBuffer> gpu_bufs(binding_count);
        std::vector<bool> is_cached(binding_count, false);
        auto cleanup_bufs = [&]() {
            for (uint32_t i = 0; i < binding_count; ++i) {
                if (!is_cached[i]) {
                    destroy_buffer(gpu_bufs[i]);
                }
            }
        };

        for (uint32_t i = 0; i < binding_count; ++i) {
            if (bindings[i].cache_upload && bindings[i].upload_data && !bindings[i].download_data) {
                auto it = weight_cache.find(bindings[i].upload_data);
                if (it != weight_cache.end() && it->second.size == bindings[i].size_bytes) {
                    gpu_bufs[i] = it->second;
                    is_cached[i] = true;
                    continue;
                }
                if (it != weight_cache.end()) {
                    cached_bytes -= it->second.size;
                    destroy_buffer(it->second);
                    weight_cache.erase(it);
                }
                constexpr VkDeviceSize kMaxCacheBytes = 256ULL * 1024ULL * 1024ULL;
                if (cached_bytes + bindings[i].size_bytes > kMaxCacheBytes) {
                    clear_weight_cache();
                }
            }

            if (!create_buffer(bindings[i].size_bytes, gpu_bufs[i], err)) {
                cleanup_bufs();
                return false;
            }
            if (bindings[i].upload_data) {
                void* mapped = nullptr;
                VkResult res = vkMapMemory(device, gpu_bufs[i].memory, 0, bindings[i].size_bytes, 0, &mapped);
                if (res != VK_SUCCESS) {
                    cleanup_bufs();
                    set_error(err, "vkMapMemory (upload) failed: " + std::to_string(res));
                    return false;
                }
                std::memcpy(mapped, bindings[i].upload_data, static_cast<size_t>(bindings[i].size_bytes));
                if (!gpu_bufs[i].coherent) {
                    VkMappedMemoryRange range{};
                    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                    range.memory = gpu_bufs[i].memory;
                    range.offset = 0;
                    range.size = VK_WHOLE_SIZE;
                    vkFlushMappedMemoryRanges(device, 1, &range);
                }
                vkUnmapMemory(device, gpu_bufs[i].memory);

                if (bindings[i].cache_upload && !bindings[i].download_data) {
                    weight_cache.emplace(bindings[i].upload_data, gpu_bufs[i]);
                    cached_bytes += bindings[i].size_bytes;
                    is_cached[i] = true;
                }
            }
        }

        VkResult res = vkResetDescriptorPool(device, desc_pool, 0);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkResetDescriptorPool failed: " + std::to_string(res));
            return false;
        }

        VkDescriptorSetAllocateInfo ds_alloc{};
        ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_alloc.descriptorPool = desc_pool;
        ds_alloc.descriptorSetCount = 1;
        ds_alloc.pSetLayouts = &kp->desc_set_layout;

        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        res = vkAllocateDescriptorSets(device, &ds_alloc, &desc_set);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkAllocateDescriptorSets failed: " + std::to_string(res));
            return false;
        }

        std::vector<VkDescriptorBufferInfo> buf_infos(binding_count);
        std::vector<VkWriteDescriptorSet> writes(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            buf_infos[i].buffer = gpu_bufs[i].buffer;
            buf_infos[i].offset = 0;
            buf_infos[i].range = bindings[i].size_bytes;

            writes[i] = VkWriteDescriptorSet{};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device, binding_count, writes.data(), 0, nullptr);

        res = vkResetCommandBuffer(cmd_buf, 0);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkResetCommandBuffer failed: " + std::to_string(res));
            return false;
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        res = vkBeginCommandBuffer(cmd_buf, &begin_info);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkBeginCommandBuffer failed: " + std::to_string(res));
            return false;
        }

        if (query_pool != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(cmd_buf, query_pool, 0, 2);
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        }

        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, kp->pipeline);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                kp->pipeline_layout, 0, 1, &desc_set, 0, nullptr);
        if (push_constants && push_size > 0) {
            vkCmdPushConstants(cmd_buf, kp->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, push_size, push_constants);
        }
        vkCmdDispatch(cmd_buf, gx, gy, gz);

        if (query_pool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        }

        res = vkEndCommandBuffer(cmd_buf);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkEndCommandBuffer failed: " + std::to_string(res));
            return false;
        }

        res = vkResetFences(device, 1, &fence);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkResetFences failed: " + std::to_string(res));
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buf;

        const auto t_start = std::chrono::steady_clock::now();
        res = vkQueueSubmit(queue, 1, &submit_info, fence);
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkQueueSubmit failed: " + std::to_string(res));
            return false;
        }

        res = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        const auto t_end = std::chrono::steady_clock::now();
        if (res != VK_SUCCESS) {
            cleanup_bufs();
            set_error(err, "vkWaitForFences failed: " + std::to_string(res));
            return false;
        }

        if (gpu_us) {
            double measured_us =
                std::chrono::duration<double, std::micro>(t_end - t_start).count();
            if (query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f) {
                uint64_t timestamps[2] = {};
                VkResult qres = vkGetQueryPoolResults(
                    device, query_pool, 0, 2, sizeof(timestamps), timestamps,
                    sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                if (qres == VK_SUCCESS) {
                    const uint64_t mask = (timestamp_valid_bits >= 64)
                        ? ~0ULL
                        : ((1ULL << timestamp_valid_bits) - 1ULL);
                    const uint64_t delta = (timestamps[1] - timestamps[0]) & mask;
                    measured_us = static_cast<double>(delta) *
                                  static_cast<double>(timestamp_period_ns) / 1000.0;
                }
            }
            *gpu_us = measured_us;
        }

        for (uint32_t i = 0; i < binding_count; ++i) {
            if (bindings[i].download_data) {
                void* mapped = nullptr;
                res = vkMapMemory(device, gpu_bufs[i].memory, 0, bindings[i].size_bytes, 0, &mapped);
                if (res != VK_SUCCESS) {
                    cleanup_bufs();
                    set_error(err, "vkMapMemory (download) failed: " + std::to_string(res));
                    return false;
                }
                if (!gpu_bufs[i].coherent) {
                    VkMappedMemoryRange range{};
                    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                    range.memory = gpu_bufs[i].memory;
                    range.offset = 0;
                    range.size = VK_WHOLE_SIZE;
                    vkInvalidateMappedMemoryRanges(device, 1, &range);
                }
                std::memcpy(bindings[i].download_data, mapped, static_cast<size_t>(bindings[i].size_bytes));
                vkUnmapMemory(device, gpu_bufs[i].memory);
            }
        }

        cleanup_bufs();
        return true;
    }
};

VulkanComputeBackend::VulkanComputeBackend() : impl_(std::make_unique<Impl>()) {}

VulkanComputeBackend::~VulkanComputeBackend() {
    shutdown();
}

bool VulkanComputeBackend::initialize(std::string* err) {
    if (impl_->initialized) {
        return true;
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "ExyNova";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "ExyNova";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo inst_info{};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;

    VkResult res = vkCreateInstance(&inst_info, nullptr, &impl_->instance);
    if (res != VK_SUCCESS) {
        set_error(err, "vkCreateInstance failed: " + std::to_string(res));
        return false;
    }

    uint32_t dev_count = 0;
    res = vkEnumeratePhysicalDevices(impl_->instance, &dev_count, nullptr);
    if (res != VK_SUCCESS || dev_count == 0) {
        impl_->destroy_all();
        set_error(err, "No Vulkan physical devices found");
        return false;
    }

    std::vector<VkPhysicalDevice> phys_devs(dev_count);
    res = vkEnumeratePhysicalDevices(impl_->instance, &dev_count, phys_devs.data());
    if (res != VK_SUCCESS) {
        impl_->destroy_all();
        set_error(err, "vkEnumeratePhysicalDevices failed: " + std::to_string(res));
        return false;
    }

    ProbeResult best_probe;
    VkPhysicalDevice chosen_dev = VK_NULL_HANDLE;
    for (VkPhysicalDevice dev : phys_devs) {
        ProbeResult pr = probe_first_compute_device(impl_->instance, dev);
        if (pr.status == ProbeResult::Status::SUCCESS) {
            chosen_dev = dev;
            best_probe = std::move(pr);
            break;
        }
        if (pr.caps.has_compute_queue && chosen_dev == VK_NULL_HANDLE &&
            pr.status == ProbeResult::Status::MISSING_FLOAT16) {
            chosen_dev = dev;
            best_probe = std::move(pr);
        }
    }

    if (chosen_dev == VK_NULL_HANDLE) {
        impl_->destroy_all();
        set_error(err, "No suitable Vulkan compute device found");
        return false;
    }

    impl_->phys_dev = chosen_dev;
    caps_ = std::move(best_probe.caps);
    impl_->timestamp_period_ns = caps_.timestamp_period_ns;

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(impl_->phys_dev, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(impl_->phys_dev, &qf_count, qf_props.data());

    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            impl_->compute_queue_family = i;
            impl_->timestamp_valid_bits = qf_props[i].timestampValidBits;
            break;
        }
    }
    if (impl_->compute_queue_family == UINT32_MAX) {
        impl_->destroy_all();
        set_error(err, "No compute queue family available");
        return false;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = impl_->compute_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dev_info{};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &queue_info;

    res = vkCreateDevice(impl_->phys_dev, &dev_info, nullptr, &impl_->device);
    if (res != VK_SUCCESS) {
        impl_->destroy_all();
        set_error(err, "vkCreateDevice failed: " + std::to_string(res));
        return false;
    }

    vkGetDeviceQueue(impl_->device, impl_->compute_queue_family, 0, &impl_->queue);
    vkGetPhysicalDeviceMemoryProperties(impl_->phys_dev, &impl_->mem_props);

    VkCommandPoolCreateInfo cp_info{};
    cp_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_info.queueFamilyIndex = impl_->compute_queue_family;
    cp_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    res = vkCreateCommandPool(impl_->device, &cp_info, nullptr, &impl_->cmd_pool);
    if (res != VK_SUCCESS) {
        impl_->destroy_all();
        set_error(err, "vkCreateCommandPool failed: " + std::to_string(res));
        return false;
    }

    VkCommandBufferAllocateInfo cb_info{};
    cb_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_info.commandPool = impl_->cmd_pool;
    cb_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_info.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(impl_->device, &cb_info, &impl_->cmd_buf);
    if (res != VK_SUCCESS) {
        impl_->destroy_all();
        set_error(err, "vkAllocateCommandBuffers failed: " + std::to_string(res));
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 16;

    VkDescriptorPoolCreateInfo dp_info{};
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.maxSets = 4;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = &pool_size;

    res = vkCreateDescriptorPool(impl_->device, &dp_info, nullptr, &impl_->desc_pool);
    if (res != VK_SUCCESS) {
        impl_->destroy_all();
        set_error(err, "vkCreateDescriptorPool failed: " + std::to_string(res));
        return false;
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = vkCreateFence(impl_->device, &fence_info, nullptr, &impl_->fence);
    if (res != VK_SUCCESS) {
        impl_->destroy_all();
        set_error(err, "vkCreateFence failed: " + std::to_string(res));
        return false;
    }

    if (impl_->timestamp_period_ns > 0.0f && impl_->timestamp_valid_bits > 0) {
        VkQueryPoolCreateInfo qp_info{};
        qp_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp_info.queryCount = 2;
        if (vkCreateQueryPool(impl_->device, &qp_info, nullptr, &impl_->query_pool) != VK_SUCCESS) {
            impl_->query_pool = VK_NULL_HANDLE;
        }
    }

    impl_->initialized = true;
    return true;
}

void VulkanComputeBackend::shutdown() {
    if (impl_) {
        impl_->destroy_all();
    }
    caps_ = PhysicalDeviceCaps{};
}

void VulkanComputeBackend::clear_weight_cache() {
    if (impl_ && impl_->initialized) {
        if (impl_->device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(impl_->device);
        }
        impl_->clear_weight_cache();
    }
}

bool VulkanComputeBackend::is_initialized() const {
    return impl_ && impl_->initialized;
}

const PhysicalDeviceCaps& VulkanComputeBackend::device_caps() const {
    return caps_;
}

bool VulkanComputeBackend::add(const float* a, const float* b, float* out, size_t count,
                               double* gpu_us, std::string* err) {
    if (count == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!a || !b || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::add");
        return false;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
    const uint32_t pc_count = static_cast<uint32_t>(count);
    const BufferBinding bindings[3] = {
        {a, nullptr, bytes},
        {b, nullptr, bytes},
        {nullptr, out, bytes},
    };
    const uint32_t gx = (pc_count + 255u) / 256u;
    return impl_->dispatch_kernel(KernelId::Add, bindings, 3,
                                  &pc_count, sizeof(pc_count), gx, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::mul(const float* a, const float* b, float* out, size_t count,
                               double* gpu_us, std::string* err) {
    if (count == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!a || !b || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::mul");
        return false;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
    const uint32_t pc_count = static_cast<uint32_t>(count);
    const BufferBinding bindings[3] = {
        {a, nullptr, bytes},
        {b, nullptr, bytes},
        {nullptr, out, bytes},
    };
    const uint32_t gx = (pc_count + 255u) / 256u;
    return impl_->dispatch_kernel(KernelId::Mul, bindings, 3,
                                  &pc_count, sizeof(pc_count), gx, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::relu(const float* in, float* out, size_t count,
                                double* gpu_us, std::string* err) {
    if (count == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!in || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::relu");
        return false;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
    const uint32_t pc_count = static_cast<uint32_t>(count);
    const BufferBinding bindings[2] = {
        {in, nullptr, bytes},
        {nullptr, out, bytes},
    };
    const uint32_t gx = (pc_count + 255u) / 256u;
    return impl_->dispatch_kernel(KernelId::Relu, bindings, 2,
                                  &pc_count, sizeof(pc_count), gx, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::silu(const float* in, float* out, size_t count,
                                double* gpu_us, std::string* err) {
    if (count == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!in || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::silu");
        return false;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
    const uint32_t pc_count = static_cast<uint32_t>(count);
    const BufferBinding bindings[2] = {
        {in, nullptr, bytes},
        {nullptr, out, bytes},
    };
    const uint32_t gx = (pc_count + 255u) / 256u;
    return impl_->dispatch_kernel(KernelId::Silu, bindings, 2,
                                  &pc_count, sizeof(pc_count), gx, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::softmax(const float* in, float* out, uint32_t rows, uint32_t cols,
                                   double* gpu_us, std::string* err) {
    if (rows == 0 || cols == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!in || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::softmax");
        return false;
    }
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(rows) * static_cast<VkDeviceSize>(cols) * sizeof(float);
    const uint32_t pc[2] = {rows, cols};
    const BufferBinding bindings[2] = {
        {in, nullptr, bytes},
        {nullptr, out, bytes},
    };
    return impl_->dispatch_kernel(KernelId::Softmax, bindings, 2,
                                  pc, sizeof(pc), rows, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::matmul(const float* a, const float* b, float* c,
                                  uint32_t M, uint32_t N, uint32_t K,
                                  double* gpu_us, std::string* err) {
    if (M == 0 || N == 0 || K == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!a || !b || !c) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::matmul");
        return false;
    }
    const VkDeviceSize bytes_a = static_cast<VkDeviceSize>(M) * K * sizeof(float);
    const VkDeviceSize bytes_b = static_cast<VkDeviceSize>(K) * N * sizeof(float);
    const VkDeviceSize bytes_c = static_cast<VkDeviceSize>(M) * N * sizeof(float);
    const uint32_t pc[3] = {M, N, K};
    const BufferBinding bindings[3] = {
        {a, nullptr, bytes_a},
        {b, nullptr, bytes_b},
        {nullptr, c, bytes_c},
    };
    const uint32_t gx = (N + 15u) / 16u;
    const uint32_t gy = (M + 15u) / 16u;
    return impl_->dispatch_kernel(KernelId::MatMul, bindings, 3,
                                  pc, sizeof(pc), gx, gy, 1, gpu_us, err);
}

bool VulkanComputeBackend::rmsnorm(const float* x, const float* weight, float* out,
                                   uint32_t rows, uint32_t cols, float eps,
                                   double* gpu_us, std::string* err, bool cache_weights) {
    if (rows == 0 || cols == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!x || !weight || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::rmsnorm");
        return false;
    }
    const VkDeviceSize bytes_x = static_cast<VkDeviceSize>(rows) * cols * sizeof(float);
    const VkDeviceSize bytes_w = static_cast<VkDeviceSize>(cols) * sizeof(float);
    struct PushConst {
        uint32_t cols;
        float eps;
    } pc{cols, eps};
    const BufferBinding bindings[3] = {
        {x, nullptr, bytes_x, false},
        {weight, nullptr, bytes_w, cache_weights},
        {nullptr, out, bytes_x, false},
    };
    return impl_->dispatch_kernel(KernelId::RMSNorm, bindings, 3,
                                  &pc, sizeof(pc), rows, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::rope(const float* in, float* out,
                                uint32_t num_heads, uint32_t head_dim, uint32_t pos,
                                float theta_base, double* gpu_us, std::string* err,
                                uint32_t rope_mode) {
    if (num_heads == 0 || head_dim == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!in || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::rope");
        return false;
    }
    const uint32_t half_dim = head_dim / 2u;
    if (half_dim == 0) {
        std::memcpy(out, in, static_cast<size_t>(num_heads) * sizeof(float));
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(num_heads) * head_dim * sizeof(float);
    struct PushConst {
        uint32_t num_heads;
        uint32_t head_dim;
        uint32_t pos;
        float theta_base;
        uint32_t rope_mode;
    } pc{num_heads, head_dim, pos, theta_base, rope_mode};
    const BufferBinding bindings[2] = {
        {in, nullptr, bytes, false},
        {nullptr, out, bytes, false},
    };
    const uint32_t gx = (num_heads * half_dim + 255u) / 256u;
    return impl_->dispatch_kernel(KernelId::RoPE, bindings, 2,
                                  &pc, sizeof(pc), gx, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::matvec_q8_0(const int32_t* packed_i8, const float* scales,
                                       const float* x, float* y,
                                       uint32_t M, uint32_t K,
                                       double* gpu_us, std::string* err,
                                       bool cache_weights) {
    if (M == 0 || K == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!packed_i8 || !scales || !x || !y) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::matvec_q8_0");
        return false;
    }
    if (K % 32u != 0u) {
        set_error(err, "VulkanComputeBackend::matvec_q8_0 requires K to be a multiple of 32");
        return false;
    }
    const VkDeviceSize bytes_w = static_cast<VkDeviceSize>(M) * (K / 4u) * sizeof(int32_t);
    const VkDeviceSize bytes_s = static_cast<VkDeviceSize>(M) * (K / 32u) * sizeof(float);
    const VkDeviceSize bytes_x = static_cast<VkDeviceSize>(K) * sizeof(float);
    const VkDeviceSize bytes_y = static_cast<VkDeviceSize>(M) * sizeof(float);
    const uint32_t pc[2] = {M, K};
    const BufferBinding bindings[4] = {
        {packed_i8, nullptr, bytes_w, cache_weights},
        {scales, nullptr, bytes_s, cache_weights},
        {x, nullptr, bytes_x, false},
        {nullptr, y, bytes_y, false},
    };
    return impl_->dispatch_kernel(KernelId::MatVecQ8_0, bindings, 4,
                                  pc, sizeof(pc), M, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::matvec_q4_0(const uint32_t* packed_q4, const float* scales,
                                       const float* x, float* y,
                                       uint32_t M, uint32_t K,
                                       double* gpu_us, std::string* err,
                                       bool cache_weights) {
    if (M == 0 || K == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!packed_q4 || !scales || !x || !y) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::matvec_q4_0");
        return false;
    }
    if (K % 32u != 0u) {
        set_error(err, "VulkanComputeBackend::matvec_q4_0 requires K to be a multiple of 32");
        return false;
    }
    const VkDeviceSize bytes_w = static_cast<VkDeviceSize>(M) * (K / 8u) * sizeof(uint32_t);
    const VkDeviceSize bytes_s = static_cast<VkDeviceSize>(M) * (K / 32u) * sizeof(float);
    const VkDeviceSize bytes_x = static_cast<VkDeviceSize>(K) * sizeof(float);
    const VkDeviceSize bytes_y = static_cast<VkDeviceSize>(M) * sizeof(float);
    const uint32_t pc[2] = {M, K};
    const BufferBinding bindings[4] = {
        {packed_q4, nullptr, bytes_w, cache_weights},
        {scales, nullptr, bytes_s, cache_weights},
        {x, nullptr, bytes_x, false},
        {nullptr, y, bytes_y, false},
    };
    return impl_->dispatch_kernel(KernelId::MatVecQ4_0, bindings, 4,
                                  pc, sizeof(pc), M, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::matvec_f16(const uint32_t* packed_f16x2,
                                      const float* x, float* y,
                                      uint32_t M, uint32_t K,
                                      double* gpu_us, std::string* err,
                                      bool cache_weights) {
    if (M == 0 || K == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!packed_f16x2 || !x || !y) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::matvec_f16");
        return false;
    }
    if (K % 2u != 0u) {
        set_error(err, "VulkanComputeBackend::matvec_f16 requires K to be even");
        return false;
    }
    const VkDeviceSize bytes_w = static_cast<VkDeviceSize>(M) * (K / 2u) * sizeof(uint32_t);
    const VkDeviceSize bytes_x = static_cast<VkDeviceSize>(K) * sizeof(float);
    const VkDeviceSize bytes_y = static_cast<VkDeviceSize>(M) * sizeof(float);
    const uint32_t pc[2] = {M, K};
    const BufferBinding bindings[3] = {
        {packed_f16x2, nullptr, bytes_w, cache_weights},
        {x, nullptr, bytes_x, false},
        {nullptr, y, bytes_y, false},
    };
    return impl_->dispatch_kernel(KernelId::MatVecF16, bindings, 3,
                                  pc, sizeof(pc), M, 1, 1, gpu_us, err);
}

bool VulkanComputeBackend::conv2d_nchw(const float* input, const float* weight, const float* bias, float* out,
                                       uint32_t N, uint32_t C_in, uint32_t H_in, uint32_t W_in,
                                       uint32_t C_out, uint32_t KH, uint32_t KW,
                                       uint32_t stride_h, uint32_t stride_w,
                                       uint32_t pad_h, uint32_t pad_w,
                                       double* gpu_us, std::string* err) {
    if (stride_h == 0 || stride_w == 0 || KH == 0 || KW == 0) {
        set_error(err, "Invalid stride or kernel size in VulkanComputeBackend::conv2d_nchw");
        return false;
    }
    if (H_in + 2u * pad_h < KH || W_in + 2u * pad_w < KW) {
        set_error(err, "Kernel size exceeds padded input dimensions in conv2d_nchw");
        return false;
    }
    const uint32_t H_out = (H_in + 2u * pad_h - KH) / stride_h + 1u;
    const uint32_t W_out = (W_in + 2u * pad_w - KW) / stride_w + 1u;
    if (N == 0 || C_in == 0 || C_out == 0 || H_out == 0 || W_out == 0) {
        if (gpu_us) *gpu_us = 0.0;
        return true;
    }
    if (!input || !weight || !out) {
        set_error(err, "Null pointer passed to VulkanComputeBackend::conv2d_nchw");
        return false;
    }

    std::vector<float> zero_bias;
    const float* bias_ptr = bias;
    if (!bias_ptr) {
        zero_bias.assign(C_out, 0.0f);
        bias_ptr = zero_bias.data();
    }

    const VkDeviceSize bytes_in = static_cast<VkDeviceSize>(N) * C_in * H_in * W_in * sizeof(float);
    const VkDeviceSize bytes_w = static_cast<VkDeviceSize>(C_out) * C_in * KH * KW * sizeof(float);
    const VkDeviceSize bytes_b = static_cast<VkDeviceSize>(C_out) * sizeof(float);
    const VkDeviceSize bytes_out = static_cast<VkDeviceSize>(N) * C_out * H_out * W_out * sizeof(float);

    struct PushConst {
        uint32_t N;
        uint32_t C_in;
        uint32_t H_in;
        uint32_t W_in;
        uint32_t C_out;
        uint32_t H_out;
        uint32_t W_out;
        uint32_t KH;
        uint32_t KW;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t pad_h;
        uint32_t pad_w;
    } pc{N, C_in, H_in, W_in, C_out, H_out, W_out, KH, KW, stride_h, stride_w, pad_h, pad_w};

    const BufferBinding bindings[4] = {
        {input, nullptr, bytes_in},
        {weight, nullptr, bytes_w},
        {bias_ptr, nullptr, bytes_b},
        {nullptr, out, bytes_out},
    };
    const uint32_t total_out = N * C_out * H_out * W_out;
    const uint32_t gx = (total_out + 255u) / 256u;
    return impl_->dispatch_kernel(KernelId::Conv2D, bindings, 4,
                                  &pc, sizeof(pc), gx, 1, 1, gpu_us, err);
}

#endif  // EXYNOVA_ANDROID

}  // namespace gpu
}  // namespace exynova
