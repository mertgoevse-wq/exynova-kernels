#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace exynova {

namespace gpu {
class VulkanComputeBackend;
}  // namespace gpu

namespace llm {

enum class QuantType : uint32_t {
    F32  = 0,
    Q8_0 = 1,
    Q4_0 = 2,
    FP16 = 3,
};

inline const char* quant_type_name(QuantType qt) {
    switch (qt) {
        case QuantType::F32:  return "FP32";
        case QuantType::Q8_0: return "Q8_0";
        case QuantType::Q4_0: return "Q4_0";
        case QuantType::FP16: return "FP16";
    }
    return "UNKNOWN";
}

struct LlmConfig {
    uint32_t vocab_size = 256;
    uint32_t dim = 128;
    uint32_t hidden_dim = 256;
    uint32_t n_layers = 2;
    uint32_t n_heads = 4;
    uint32_t n_kv_heads = 4;
    uint32_t max_seq_len = 128;
    float rms_eps = 1e-5f;
    float rope_theta = 10000.0f;
    uint32_t rope_mode = 0;  // 0 = Llama interleaved, 1 = NeoX half-split
    uint32_t bos_token_id = 1;
    uint32_t eos_token_id = 2;
    bool has_qkv_bias = false;
    uint32_t alt_eos_token_id = 0;
};

struct Q8_0Matrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<int32_t> packed_i8;
    std::vector<float> scales;

    uint32_t padded_cols() const {
        return ((cols + 31u) / 32u) * 32u;
    }
};

struct Q4_0Matrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint32_t> packed_q4;  // 8 nibbles per uint32, 4 words per 32-block
    std::vector<float> scales;

    uint32_t padded_cols() const {
        return ((cols + 31u) / 32u) * 32u;
    }
};

struct FP16Matrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint32_t> packed_f16;  // 2 IEEE-754 half floats per uint32

    uint32_t padded_cols() const {
        return ((cols + 1u) / 2u) * 2u;
    }
};

struct WeightMatrix {
    QuantType quant = QuantType::Q8_0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    Q8_0Matrix q8;
    Q4_0Matrix q4;
    FP16Matrix f16;
    std::vector<float> f32;

    size_t storage_bytes() const {
        switch (quant) {
            case QuantType::Q8_0:
                return q8.packed_i8.size() * sizeof(int32_t) + q8.scales.size() * sizeof(float);
            case QuantType::Q4_0:
                return q4.packed_q4.size() * sizeof(uint32_t) + q4.scales.size() * sizeof(float);
            case QuantType::FP16:
                return f16.packed_f16.size() * sizeof(uint32_t);
            case QuantType::F32:
                return f32.size() * sizeof(float);
        }
        return 0;
    }
};

uint16_t float32_to_float16(float value);
float float16_to_float32(uint16_t h);

Q8_0Matrix quantize_q8_0(const float* src, uint32_t rows, uint32_t cols);
std::vector<float> dequantize_q8_0(const Q8_0Matrix& mat);

Q4_0Matrix quantize_q4_0(const float* src, uint32_t rows, uint32_t cols);
std::vector<float> dequantize_q4_0(const Q4_0Matrix& mat);

FP16Matrix quantize_f16(const float* src, uint32_t rows, uint32_t cols);
std::vector<float> dequantize_f16(const FP16Matrix& mat);

struct TransformerLayerWeights {
    std::vector<float> attn_norm;
    WeightMatrix wq;
    WeightMatrix wk;
    WeightMatrix wv;
    std::vector<float> bq;
    std::vector<float> bk;
    std::vector<float> bv;
    WeightMatrix wo;

    std::vector<float> ffn_norm;
    WeightMatrix w1;
    WeightMatrix w2;
    WeightMatrix w3;
};

struct GenerationResult {
    std::string text;
    uint32_t tokens_generated = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double tokens_per_sec = 0.0;
    bool used_vulkan_gpu = false;
    bool is_demo_weights = false;
    bool used_bpe_vocab = false;
    std::string quant_name;
    size_t model_bytes = 0;
};

class LlmEngine {
public:
    LlmEngine() = default;
    ~LlmEngine();

    LlmEngine(const LlmEngine&) = delete;
    LlmEngine& operator=(const LlmEngine&) = delete;

    bool init_demo_tiny_lm(const LlmConfig& cfg,
                           gpu::VulkanComputeBackend* vk = nullptr,
                           std::string* err = nullptr,
                           QuantType quant = QuantType::Q8_0);

    bool load_exyllm_file(const std::string& path,
                          gpu::VulkanComputeBackend* vk = nullptr,
                          std::string* err = nullptr);

    bool load_vocab_file(const std::string& tok_path, std::string* err = nullptr);

    void reset_kv_cache();

    bool forward_token(uint32_t token_id,
                       uint32_t pos,
                       std::vector<float>& logits,
                       std::string* err = nullptr);

    GenerationResult generate(const std::string& prompt,
                              uint32_t max_new_tokens,
                              float temperature = 0.0f);

    bool is_initialized() const { return initialized_; }
    bool is_demo_weights() const { return is_demo_; }
    bool has_vocab() const { return has_vocab_; }
    QuantType quant_type() const { return quant_type_; }
    size_t model_bytes() const;
    const LlmConfig& config() const { return cfg_; }
    std::vector<uint32_t> encode_prompt(const std::string& prompt) const;

private:
    struct VocabEntry {
        std::string piece;
        float score = 0.0f;
    };

    bool validate_config(const LlmConfig& cfg, std::string* err) const;
    void allocate_state();
    std::string decode_token(uint32_t token_id) const;

    bool run_rmsnorm(const float* x, const float* w, float* out,
                     uint32_t cols, std::string* err);
    bool run_matvec(const WeightMatrix& w, const float* x, float* y,
                    std::string* err);
    bool run_rope(const float* in, float* out, uint32_t num_heads,
                  uint32_t head_dim, uint32_t pos, std::string* err);
    bool run_silu(const float* in, float* out, uint32_t count, std::string* err);
    bool run_softmax(const float* in, float* out, uint32_t rows, uint32_t cols,
                     std::string* err);
    uint32_t sample_token(const std::vector<float>& logits, float temperature,
                          uint64_t& rng_state) const;

    LlmConfig cfg_;
    gpu::VulkanComputeBackend* vk_ = nullptr;
    bool initialized_ = false;
    bool is_demo_ = false;
    bool has_vocab_ = false;
    uint32_t tok_style_ = 0;  // 0 = SentencePiece '▁', 1 = Byte-level BPE 'Ġ'
    QuantType quant_type_ = QuantType::Q8_0;

    std::vector<VocabEntry> vocab_;
    std::unordered_map<std::string, uint32_t> piece_to_id_;

    std::vector<float> token_embedding_;
    std::vector<uint16_t> token_embedding_f16_;
    std::vector<TransformerLayerWeights> layers_;
    std::vector<float> final_norm_;
    WeightMatrix lm_head_;

    // KV cache: [n_layers][max_seq_len * kv_dim]
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;

    // Scratch buffers
    std::vector<float> x_;
    std::vector<float> xb_;
    std::vector<float> xb2_;
    std::vector<float> q_;
    std::vector<float> q_rot_;
    std::vector<float> k_;
    std::vector<float> k_rot_;
    std::vector<float> v_;
    std::vector<float> hb_;
    std::vector<float> hb_act_;
    std::vector<float> hb2_;
    std::vector<float> att_scores_;
    std::vector<float> att_probs_;
    std::vector<float> x_pad_;
};

}  // namespace llm
}  // namespace exynova
