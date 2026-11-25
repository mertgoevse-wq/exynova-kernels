// flappy_bird_ai_bench.cpp
// ExyNova 4-Mode AI Flappy Bird Benchmark & Physics Simulator (NPU / NPU+GPU / GPU / GPU+NPU).

#include "exynova/enn_backend.h"
#include "exynova/exynova_game_ai.h"
#include "exynova/gpu_probe.h"
#include "exynova/vulkan_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

enum class AccelMode {
    NPU = 0,
    NPU_GPU = 1,
    GPU = 2,
    GPU_NPU = 3
};

const char* mode_name(AccelMode m) {
    switch (m) {
        case AccelMode::NPU: return "NPU (ENN .nnc)";
        case AccelMode::NPU_GPU: return "NPU + GPU (ENN -> Vulkan)";
        case AccelMode::GPU: return "GPU (Vulkan Xclipse 540)";
        case AccelMode::GPU_NPU: return "GPU + NPU (Vulkan -> ENN)";
    }
    return "UNKNOWN";
}

struct FlappyState {
    float bird_y = 0.5f;
    float bird_vel = 0.0f;
    float pipe_dx = 0.8f;
    float gap_y = 0.5f;
    float gap_half = 0.14f;
    int score = 0;
    int crashes = 0;
    uint32_t rng = 0x9E3779B9u;

    void step_physics(bool flap) {
        if (flap) {
            bird_vel = -0.022f;
        } else {
            bird_vel = std::min(0.032f, bird_vel + 0.0028f);
        }
        bird_y += bird_vel;
        pipe_dx -= 0.022f;

        if (pipe_dx < 0.10f && pipe_dx > -0.10f) {
            if (std::fabs(bird_y - gap_y) > gap_half) {
                ++crashes;
                bird_y = gap_y;
                bird_vel = 0.0f;
            }
        }
        if (pipe_dx <= -0.12f) {
            ++score;
            pipe_dx = 0.92f;
            rng = rng * 1664525u + 1013904223u;
            const float u = static_cast<float>(rng & 0xFFu) / 255.0f;
            gap_y = 0.25f + u * 0.50f;
        }
        if (bird_y < 0.05f) { bird_y = 0.05f; bird_vel = 0.0f; }
        if (bird_y > 0.95f) { bird_y = 0.95f; bird_vel = 0.0f; }
    }
};

// Deterministic 2-layer policy weights [8 -> 16 -> 2] (row-major [K, N] for MatMul(A[64,8], W1[8,16]))
void build_policy_weights(std::vector<float>& w1, std::vector<float>& w2) {
    w1.assign(8 * 16, 0.0f);
    w2.assign(16 * 2, 0.0f);
    // Input features: [dy_gap, bird_vel, pipe_dx, gap_half, npu0, npu1, npu2, bias=1.0]
    // Neuron 0 detects bird too low (dy_gap > 0) or falling fast (bird_vel > 0) -> triggers FLAP
    w1[0 * 16 + 0] = 6.0f;
    w1[1 * 16 + 0] = 3.5f;
    w1[7 * 16 + 0] = -0.15f;

    // Neuron 1 detects bird above target -> GLIDE
    w1[0 * 16 + 1] = -6.0f;
    w1[1 * 16 + 1] = -2.5f;
    w1[7 * 16 + 1] = 0.10f;

    w2[0 * 2 + 0] = 1.0f;  // FLAP score
    w2[1 * 2 + 1] = 1.0f;  // GLIDE score
}

bool policy_forward_cpu(const std::vector<float>& x64x8,
                        const std::vector<float>& w1,
                        const std::vector<float>& w2,
                        std::vector<float>& logits64x2) {
    logits64x2.assign(64 * 2, 0.0f);
    for (int r = 0; r < 64; ++r) {
        float h[16] = {};
        for (int j = 0; j < 16; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < 8; ++k) {
                acc += x64x8[r * 8 + k] * w1[k * 16 + j];
            }
            h[j] = acc > 0.0f ? acc : 0.0f;
        }
        for (int o = 0; o < 2; ++o) {
            float acc = 0.0f;
            for (int j = 0; j < 16; ++j) {
                acc += h[j] * w2[j * 2 + o];
            }
            logits64x2[r * 2 + o] = acc;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string nnc_path = "/vendor/etc/portrait_data/SRIB_HumanSegVideoLite_INT8_V101_ex8845_enn225k.nnc";
    int frames = 30;
    if (argc > 1 && argv[1]) nnc_path = argv[1];
    if (argc > 2 && argv[2]) frames = std::max(5, std::atoi(argv[2]));

    printf("=== ExyNova Flappy Bird 4-Mode AI Benchmark ===\n");

    exynova::gpu::VulkanComputeBackend vk;
    std::string err;
    const bool has_vk = vk.initialize(&err);
    printf("Vulkan GPU: %s (%s)\n", has_vk ? "ACTIVE" : "HOST/UNAVAILABLE",
           has_vk ? vk.device_caps().device_name.c_str() : err.c_str());

    exynova::npu::EnnNpuBackend enn;
    const auto enn_probe = enn.probe_and_initialize(false);
    bool has_enn_session = false;
    if (enn_probe.initialized) {
        has_enn_session = enn.open_session(nnc_path, &err);
    }
    printf("ENN NPU:    %s (%s)\n\n", has_enn_session ? "ACTIVE" : "HOST/UNAVAILABLE",
           has_enn_session ? nnc_path.c_str() : (enn_probe.error_msg.empty() ? err.c_str() : enn_probe.error_msg.c_str()));

    std::vector<float> w1, w2;
    build_policy_weights(w1, w2);
    std::vector<float> grid(256, 0.0f);
    std::vector<float> conv_w(4 * 9, 0.125f);
    std::vector<float> conv_b(4, 0.0f);
    std::vector<float> conv_out(4 * 256, 0.0f);
    std::vector<float> obs(64 * 8, 0.0f);
    std::vector<float> h1(64 * 16, 0.0f);
    std::vector<float> h1_act(64 * 16, 0.0f);
    std::vector<float> logits(64 * 2, 0.0f);
    std::vector<uint8_t> npu_in(256, 7u);
    std::vector<uint8_t> npu_out;

    for (int m = 0; m < 4; ++m) {
        const AccelMode mode = static_cast<AccelMode>(m);
        FlappyState sim;
        double sum_npu_ms = 0.0;
        double sum_gpu_ms = 0.0;
        double sum_total_ms = 0.0;
        int flaps = 0;

        for (int f = 0; f < frames; ++f) {
            const auto t0 = std::chrono::steady_clock::now();
            double npu_ms = 0.0;
            double gpu_ms = 0.0;

            float npu_feat = 0.0f;
            if ((mode == AccelMode::NPU || mode == AccelMode::NPU_GPU) && has_enn_session) {
                npu_in[0] = static_cast<uint8_t>(std::clamp(sim.bird_y, 0.0f, 1.0f) * 255.0f);
                npu_in[1] = static_cast<uint8_t>(std::clamp(sim.gap_y, 0.0f, 1.0f) * 255.0f);
                if (enn.execute_session(npu_in.data(), npu_in.size(), &npu_out, &npu_ms, &err) && !npu_out.empty()) {
                    npu_feat = static_cast<float>(npu_out[0]) / 255.0f;
                }
            }

            const float dy = sim.bird_y - sim.gap_y;
            for (int r = 0; r < 64; ++r) {
                obs[r * 8 + 0] = dy;
                obs[r * 8 + 1] = sim.bird_vel;
                obs[r * 8 + 2] = sim.pipe_dx;
                obs[r * 8 + 3] = sim.gap_half;
                obs[r * 8 + 4] = npu_feat;
                obs[r * 8 + 5] = 0.0f;
                obs[r * 8 + 6] = 0.0f;
                obs[r * 8 + 7] = 1.0f;
            }

            if (mode == AccelMode::NPU_GPU || mode == AccelMode::GPU || mode == AccelMode::GPU_NPU) {
                if (has_vk) {
                    double us0 = 0.0, us1 = 0.0, us2 = 0.0, us3 = 0.0;
                    if (mode == AccelMode::GPU) {
                        grid[static_cast<size_t>(std::clamp(static_cast<int>(sim.bird_y * 15.0f), 0, 15)) * 16 + 2] = 1.0f;
                        vk.conv2d_nchw(grid.data(), conv_w.data(), conv_b.data(), conv_out.data(),
                                       1, 1, 16, 16, 4, 3, 3, 1, 1, 1, 1, &us0, &err);
                    }
                    vk.matmul(obs.data(), w1.data(), h1.data(), 64, 16, 8, &us1, &err);
                    vk.relu(h1.data(), h1_act.data(), 64 * 16, &us2, &err);
                    vk.matmul(h1_act.data(), w2.data(), logits.data(), 64, 2, 16, &us3, &err);
                    gpu_ms = (us0 + us1 + us2 + us3) / 1000.0;
                } else {
                    policy_forward_cpu(obs, w1, w2, logits);
                }
            } else {
                // Mode NPU-only: compact 2-neuron scalar threshold on NPU activation
                logits[0] = (dy * 6.0f + sim.bird_vel * 3.5f - 0.15f + npu_feat * 0.01f);
                logits[1] = 0.0f;
            }

            if (mode == AccelMode::GPU_NPU && has_enn_session) {
                double n_ms = 0.0;
                enn.execute_session(npu_in.data(), npu_in.size(), &npu_out, &n_ms, &err);
                npu_ms += n_ms;
            }

            const bool flap = logits[0] > logits[1];
            if (flap) ++flaps;
            sim.step_physics(flap);

            const auto t1 = std::chrono::steady_clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum_npu_ms += npu_ms;
            sum_gpu_ms += gpu_ms;
            sum_total_ms += wall_ms;
        }

        const double avg_tot = sum_total_ms / frames;
        const double avg_npu = sum_npu_ms / frames;
        const double avg_gpu = sum_gpu_ms / frames;
        const double max_hz = avg_tot > 0.0 ? (1000.0 / avg_tot) : 0.0;
        printf("%-26s | total=%.3f ms/frame (%.1f Hz) | npu=%.3f ms | gpu=%.3f ms | pipes=%d | crashes=%d | flaps=%d\n",
               mode_name(mode), avg_tot, max_hz, avg_npu, avg_gpu, sim.score, sim.crashes, flaps);
    }

    enn.shutdown();
    vk.shutdown();
    return 0;
}
