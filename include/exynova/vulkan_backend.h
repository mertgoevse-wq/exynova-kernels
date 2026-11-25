#pragma once

#include "exynova/gpu_probe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace exynova {
namespace gpu {

class VulkanComputeBackend {
public:
    VulkanComputeBackend();
    ~VulkanComputeBackend();

    VulkanComputeBackend(const VulkanComputeBackend&) = delete;
    VulkanComputeBackend& operator=(const VulkanComputeBackend&) = delete;

    bool initialize(std::string* err = nullptr);
    void shutdown();
    void clear_weight_cache();

    bool is_initialized() const;
    const PhysicalDeviceCaps& device_caps() const;

    bool add(const float* a, const float* b, float* out, size_t count,
             double* gpu_us = nullptr, std::string* err = nullptr);
    bool mul(const float* a, const float* b, float* out, size_t count,
             double* gpu_us = nullptr, std::string* err = nullptr);
    bool relu(const float* in, float* out, size_t count,
              double* gpu_us = nullptr, std::string* err = nullptr);
    bool silu(const float* in, float* out, size_t count,
              double* gpu_us = nullptr, std::string* err = nullptr);
    bool softmax(const float* in, float* out, uint32_t rows, uint32_t cols,
                 double* gpu_us = nullptr, std::string* err = nullptr);
    bool matmul(const float* a, const float* b, float* c,
                uint32_t M, uint32_t N, uint32_t K,
                double* gpu_us = nullptr, std::string* err = nullptr);

    bool rmsnorm(const float* x, const float* weight, float* out,
                 uint32_t rows, uint32_t cols, float eps = 1e-5f,
                 double* gpu_us = nullptr, std::string* err = nullptr,
                 bool cache_weights = false);
    bool rope(const float* in, float* out,
              uint32_t num_heads, uint32_t head_dim, uint32_t pos,
              float theta_base = 10000.0f,
              double* gpu_us = nullptr, std::string* err = nullptr,
              uint32_t rope_mode = 0);
    bool matvec_q8_0(const int32_t* packed_i8, const float* scales,
                     const float* x, float* y,
                     uint32_t M, uint32_t K,
                     double* gpu_us = nullptr, std::string* err = nullptr,
                     bool cache_weights = false);
    bool matvec_q4_0(const uint32_t* packed_q4, const float* scales,
                     const float* x, float* y,
                     uint32_t M, uint32_t K,
                     double* gpu_us = nullptr, std::string* err = nullptr,
                     bool cache_weights = false);
    bool matvec_f16(const uint32_t* packed_f16x2,
                    const float* x, float* y,
                    uint32_t M, uint32_t K,
                    double* gpu_us = nullptr, std::string* err = nullptr,
                    bool cache_weights = false);
    bool conv2d_nchw(const float* input, const float* weight, const float* bias, float* out,
                     uint32_t N, uint32_t C_in, uint32_t H_in, uint32_t W_in,
                     uint32_t C_out, uint32_t KH, uint32_t KW,
                     uint32_t stride_h, uint32_t stride_w,
                     uint32_t pad_h, uint32_t pad_w,
                     double* gpu_us = nullptr, std::string* err = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PhysicalDeviceCaps caps_;
};

}  // namespace gpu
}  // namespace exynova
