#include "exynova/exynova_game_ai.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    exynova::game_ai::GameAiRuntime ai;
    if (!ai.init(EXYNOVA_MODE_AUTO, false)) {
        std::fprintf(stderr, "Init failed: %s\n", ai.last_error().c_str());
        return 1;
    }

    std::printf("%s\n\n", ai.backend_summary().c_str());

    // 1. Dispatch a 2-layer NPC decision MLP (batch=2, in=4, hidden=8, out=3)
    const size_t batch = 2, in_dim = 4, hid_dim = 8, out_dim = 3;
    std::vector<float> npc_state = {
        0.8f, -0.3f, 0.5f, 1.0f,
        -0.4f, 0.9f, 0.1f, -0.2f
    };
    std::vector<float> w1(in_dim * hid_dim, 0.25f);
    std::vector<float> b1(hid_dim, 0.05f);
    std::vector<float> w2(hid_dim * out_dim, 0.15f);
    std::vector<float> logits;

    if (!ai.dispatch_npc_mlp(npc_state.data(), w1.data(), b1.data(), w2.data(),
                             batch, in_dim, hid_dim, out_dim, logits)) {
        std::fprintf(stderr, "dispatch_npc_mlp failed: %s\n", ai.last_error().c_str());
        return 1;
    }

    std::printf("[NPC MLP] batch=%zu out_dim=%zu cost=%.4f ms logits[0]=(%.4f, %.4f, %.4f)\n",
                batch, out_dim, ai.last_frame_cost_ms(),
                logits[0], logits[1], logits[2]);

    // 2. Run vectorized compute batch (Add)
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> out(4, 0.0f);
    if (!ai.run_vulkan_compute_batch(a.data(), b.data(), out.data(), a.size(),
                                     exynova::game_ai::BatchOpKind::Add, false)) {
        std::fprintf(stderr, "run_vulkan_compute_batch failed: %s\n", ai.last_error().c_str());
        return 1;
    }
    std::printf("[Compute Batch ADD] result=(%.1f, %.1f, %.1f, %.1f) cost=%.4f ms\n",
                out[0], out[1], out[2], out[3], ai.last_frame_cost_ms());

    for (size_t i = 0; i < 4; ++i) {
        if (std::fabs(out[i] - (a[i] + b[i])) > 1e-5f) {
            std::fprintf(stderr, "Accuracy mismatch at %zu\n", i);
            return 1;
        }
    }

    // 3. Demonstrate TFL3 -> ENNC conversion error handling & report struct
    exynova::compiler::NncConvertReport conv_rep;
    std::vector<uint8_t> bad_payload = {0, 1, 2, 3};
    std::vector<uint8_t> nnc_out;
    const bool bad_ok = exynova::compiler::convert_tflite_bytes_to_nnc(bad_payload, nnc_out, &conv_rep);
    std::printf("[NNC Converter Check] rejected_invalid=%s (%s)\n",
                (!bad_ok ? "yes" : "no"), conv_rep.error_msg.c_str());

    return 0;
}
