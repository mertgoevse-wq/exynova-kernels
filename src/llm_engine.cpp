#include "exynova/llm_engine.h"
#include "exynova/vulkan_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace exynova {
namespace llm {

namespace {

inline void set_err(std::string* err, const std::string& msg) {
    if (err) {
        *err = msg;
    }
}

uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

std::vector<float> make_seeded_weights(size_t count, uint32_t& rng, float scale) {
    std::vector<float> w(count);
    for (size_t i = 0; i < count; ++i) {
        uint32_t r = xorshift32(rng);
        float u = (static_cast<float>(r & 0xFFFFu) / 32767.5f) - 1.0f;
        w[i] = u * scale;
    }
    return w;
}

WeightMatrix make_weight_matrix(const float* src, uint32_t rows, uint32_t cols, QuantType qt) {
    WeightMatrix wm;
    wm.quant = qt;
    wm.rows = rows;
    wm.cols = cols;
    switch (qt) {
        case QuantType::Q8_0:
            wm.q8 = quantize_q8_0(src, rows, cols);
            break;
        case QuantType::Q4_0:
            wm.q4 = quantize_q4_0(src, rows, cols);
            break;
        case QuantType::FP16:
            wm.f16 = quantize_f16(src, rows, cols);
            break;
        case QuantType::F32:
            if (src && rows > 0 && cols > 0) {
                wm.f32.assign(src, src + static_cast<size_t>(rows) * cols);
            }
            break;
    }
    return wm;
}

template <typename T>
bool read_pod(std::ifstream& ifs, T& val) {
    ifs.read(reinterpret_cast<char*>(&val), sizeof(T));
    return static_cast<bool>(ifs);
}

bool read_f32_vec(std::ifstream& ifs, std::vector<float>& dst, size_t count) {
    dst.resize(count);
    if (count == 0) {
        return true;
    }
    ifs.read(reinterpret_cast<char*>(dst.data()),
             static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(ifs);
}

bool read_matrix(std::ifstream& ifs, uint32_t quant_type,
                 uint32_t rows, uint32_t cols, WeightMatrix& out,
                 std::string* err) {
    out = WeightMatrix{};
    out.rows = rows;
    out.cols = cols;

    if (quant_type == 0) {
        out.quant = QuantType::F32;
        if (!read_f32_vec(ifs, out.f32, static_cast<size_t>(rows) * cols)) {
            set_err(err, "Unexpected EOF reading F32 matrix from .exyllm file");
            return false;
        }
        return true;
    }
    if (quant_type == 1) {
        out.quant = QuantType::Q8_0;
        out.q8.rows = rows;
        out.q8.cols = cols;
        const uint32_t pad_cols = out.q8.padded_cols();
        const size_t words = static_cast<size_t>(rows) * (pad_cols / 4u);
        const size_t blocks = static_cast<size_t>(rows) * (pad_cols / 32u);
        out.q8.packed_i8.resize(words);
        out.q8.scales.resize(blocks);
        if (words > 0) {
            ifs.read(reinterpret_cast<char*>(out.q8.packed_i8.data()),
                     static_cast<std::streamsize>(words * sizeof(int32_t)));
        }
        if (blocks > 0) {
            ifs.read(reinterpret_cast<char*>(out.q8.scales.data()),
                     static_cast<std::streamsize>(blocks * sizeof(float)));
        }
        if (!ifs) {
            set_err(err, "Unexpected EOF reading Q8_0 matrix from .exyllm file");
            return false;
        }
        return true;
    }
    if (quant_type == 2) {
        out.quant = QuantType::Q4_0;
        out.q4.rows = rows;
        out.q4.cols = cols;
        const uint32_t pad_cols = out.q4.padded_cols();
        const size_t words = static_cast<size_t>(rows) * (pad_cols / 8u);
        const size_t blocks = static_cast<size_t>(rows) * (pad_cols / 32u);
        out.q4.packed_q4.resize(words);
        out.q4.scales.resize(blocks);
        if (words > 0) {
            ifs.read(reinterpret_cast<char*>(out.q4.packed_q4.data()),
                     static_cast<std::streamsize>(words * sizeof(uint32_t)));
        }
        if (blocks > 0) {
            ifs.read(reinterpret_cast<char*>(out.q4.scales.data()),
                     static_cast<std::streamsize>(blocks * sizeof(float)));
        }
        if (!ifs) {
            set_err(err, "Unexpected EOF reading Q4_0 matrix from .exyllm file");
            return false;
        }
        return true;
    }
    if (quant_type == 3) {
        out.quant = QuantType::FP16;
        out.f16.rows = rows;
        out.f16.cols = cols;
        const uint32_t pad_cols = out.f16.padded_cols();
        const size_t words = static_cast<size_t>(rows) * (pad_cols / 2u);
        out.f16.packed_f16.resize(words);
        if (words > 0) {
            ifs.read(reinterpret_cast<char*>(out.f16.packed_f16.data()),
                     static_cast<std::streamsize>(words * sizeof(uint32_t)));
        }
        if (!ifs) {
            set_err(err, "Unexpected EOF reading FP16 matrix from .exyllm file");
            return false;
        }
        return true;
    }
    set_err(err, "Unsupported quantization type in .exyllm file");
    return false;
}

bool file_exists(const std::string& p) {
    std::ifstream ifs(p, std::ios::binary);
    return ifs.is_open();
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

}  // namespace

uint16_t float32_to_float16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Inf or NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13);
    return static_cast<uint16_t>(half);
}

float float16_to_float32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f32_bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            f32_bits = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127 - 15 + 1;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            f32_bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f32_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        f32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &f32_bits, sizeof(out));
    return out;
}

Q8_0Matrix quantize_q8_0(const float* src, uint32_t rows, uint32_t cols) {
    Q8_0Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    if (rows == 0 || cols == 0 || !src) {
        return mat;
    }

    const uint32_t pad_cols = mat.padded_cols();
    const uint32_t blocks_per_row = pad_cols / 32u;
    const uint32_t words_per_row = pad_cols / 4u;

    mat.packed_i8.assign(static_cast<size_t>(rows) * words_per_row, 0);
    mat.scales.assign(static_cast<size_t>(rows) * blocks_per_row, 0.0f);

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row_ptr = src + static_cast<size_t>(r) * cols;
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint32_t k_start = b * 32u;
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32u; ++i) {
                const uint32_t k = k_start + i;
                const float v = (k < cols) ? row_ptr[k] : 0.0f;
                amax = std::max(amax, std::fabs(v));
            }
            const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
            mat.scales[static_cast<size_t>(r) * blocks_per_row + b] = scale;
            const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            for (uint32_t g = 0; g < 8u; ++g) {
                uint32_t word = 0u;
                for (uint32_t byte_i = 0; byte_i < 4u; ++byte_i) {
                    const uint32_t k = k_start + g * 4u + byte_i;
                    const float v = (k < cols) ? row_ptr[k] : 0.0f;
                    long q = std::lround(v * inv_scale);
                    q = std::max(-127L, std::min(127L, q));
                    const uint8_t u8 = static_cast<uint8_t>(static_cast<int8_t>(q));
                    word |= (static_cast<uint32_t>(u8) << (byte_i * 8u));
                }
                mat.packed_i8[static_cast<size_t>(r) * words_per_row + b * 8u + g] =
                    static_cast<int32_t>(word);
            }
        }
    }

    return mat;
}

std::vector<float> dequantize_q8_0(const Q8_0Matrix& mat) {
    std::vector<float> out(static_cast<size_t>(mat.rows) * mat.cols, 0.0f);
    if (mat.rows == 0 || mat.cols == 0) {
        return out;
    }

    const uint32_t pad_cols = mat.padded_cols();
    const uint32_t blocks_per_row = pad_cols / 32u;
    const uint32_t words_per_row = pad_cols / 4u;

    for (uint32_t r = 0; r < mat.rows; ++r) {
        float* row_out = out.data() + static_cast<size_t>(r) * mat.cols;
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const float scale = mat.scales[static_cast<size_t>(r) * blocks_per_row + b];
            const uint32_t k_start = b * 32u;
            for (uint32_t g = 0; g < 8u; ++g) {
                const uint32_t word = static_cast<uint32_t>(
                    mat.packed_i8[static_cast<size_t>(r) * words_per_row + b * 8u + g]);
                for (uint32_t byte_i = 0; byte_i < 4u; ++byte_i) {
                    const uint32_t k = k_start + g * 4u + byte_i;
                    if (k < mat.cols) {
                        const int8_t q = static_cast<int8_t>((word >> (byte_i * 8u)) & 0xFFu);
                        row_out[k] = static_cast<float>(q) * scale;
                    }
                }
            }
        }
    }
    return out;
}

Q4_0Matrix quantize_q4_0(const float* src, uint32_t rows, uint32_t cols) {
    Q4_0Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    if (rows == 0 || cols == 0 || !src) {
        return mat;
    }

    const uint32_t pad_cols = mat.padded_cols();
    const uint32_t blocks_per_row = pad_cols / 32u;
    const uint32_t words_per_row = pad_cols / 8u;

    mat.packed_q4.assign(static_cast<size_t>(rows) * words_per_row, 0u);
    mat.scales.assign(static_cast<size_t>(rows) * blocks_per_row, 0.0f);

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row_ptr = src + static_cast<size_t>(r) * cols;
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint32_t k_start = b * 32u;
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32u; ++i) {
                const uint32_t k = k_start + i;
                const float v = (k < cols) ? row_ptr[k] : 0.0f;
                amax = std::max(amax, std::fabs(v));
            }
            const float scale = (amax > 0.0f) ? (amax / 7.0f) : 0.0f;
            mat.scales[static_cast<size_t>(r) * blocks_per_row + b] = scale;
            const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            for (uint32_t g = 0; g < 4u; ++g) {
                uint32_t word = 0u;
                for (uint32_t nib = 0; nib < 8u; ++nib) {
                    const uint32_t k = k_start + g * 8u + nib;
                    const float v = (k < cols) ? row_ptr[k] : 0.0f;
                    long q = std::lround(v * inv_scale);
                    q = std::max(-8L, std::min(7L, q));
                    const uint32_t u4 = static_cast<uint32_t>(q + 8L) & 0xFu;
                    word |= (u4 << (nib * 4u));
                }
                mat.packed_q4[static_cast<size_t>(r) * words_per_row + b * 4u + g] = word;
            }
        }
    }
    return mat;
}

std::vector<float> dequantize_q4_0(const Q4_0Matrix& mat) {
    std::vector<float> out(static_cast<size_t>(mat.rows) * mat.cols, 0.0f);
    if (mat.rows == 0 || mat.cols == 0) {
        return out;
    }

    const uint32_t pad_cols = mat.padded_cols();
    const uint32_t blocks_per_row = pad_cols / 32u;
    const uint32_t words_per_row = pad_cols / 8u;

    for (uint32_t r = 0; r < mat.rows; ++r) {
        float* row_out = out.data() + static_cast<size_t>(r) * mat.cols;
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const float scale = mat.scales[static_cast<size_t>(r) * blocks_per_row + b];
            const uint32_t k_start = b * 32u;
            for (uint32_t g = 0; g < 4u; ++g) {
                const uint32_t word =
                    mat.packed_q4[static_cast<size_t>(r) * words_per_row + b * 4u + g];
                for (uint32_t nib = 0; nib < 8u; ++nib) {
                    const uint32_t k = k_start + g * 8u + nib;
                    if (k < mat.cols) {
                        const int32_t q = static_cast<int32_t>((word >> (nib * 4u)) & 0xFu) - 8;
                        row_out[k] = static_cast<float>(q) * scale;
                    }
                }
            }
        }
    }
    return out;
}

FP16Matrix quantize_f16(const float* src, uint32_t rows, uint32_t cols) {
    FP16Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    if (rows == 0 || cols == 0 || !src) {
        return mat;
    }

    const uint32_t pad_cols = mat.padded_cols();
    const uint32_t words_per_row = pad_cols / 2u;
    mat.packed_f16.assign(static_cast<size_t>(rows) * words_per_row, 0u);

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row_ptr = src + static_cast<size_t>(r) * cols;
        for (uint32_t p = 0; p < words_per_row; ++p) {
            const uint32_t k0 = p * 2u;
            const uint32_t k1 = k0 + 1u;
            const uint16_t h0 = float32_to_float16(row_ptr[k0]);
            const uint16_t h1 = (k1 < cols) ? float32_to_float16(row_ptr[k1]) : 0u;
            mat.packed_f16[static_cast<size_t>(r) * words_per_row + p] =
                static_cast<uint32_t>(h0) | (static_cast<uint32_t>(h1) << 16);
        }
    }
    return mat;
}

std::vector<float> dequantize_f16(const FP16Matrix& mat) {
    std::vector<float> out(static_cast<size_t>(mat.rows) * mat.cols, 0.0f);
    if (mat.rows == 0 || mat.cols == 0) {
        return out;
    }

    const uint32_t pad_cols = mat.padded_cols();
    const uint32_t words_per_row = pad_cols / 2u;
    for (uint32_t r = 0; r < mat.rows; ++r) {
        float* row_out = out.data() + static_cast<size_t>(r) * mat.cols;
        for (uint32_t p = 0; p < words_per_row; ++p) {
            const uint32_t word = mat.packed_f16[static_cast<size_t>(r) * words_per_row + p];
            const uint32_t k0 = p * 2u;
            const uint32_t k1 = k0 + 1u;
            row_out[k0] = float16_to_float32(static_cast<uint16_t>(word & 0xFFFFu));
            if (k1 < mat.cols) {
                row_out[k1] = float16_to_float32(static_cast<uint16_t>((word >> 16) & 0xFFFFu));
            }
        }
    }
    return out;
}

LlmEngine::~LlmEngine() {
    if (vk_) {
        vk_->clear_weight_cache();
    }
}

size_t LlmEngine::model_bytes() const {
    size_t total = token_embedding_.size() * sizeof(float) +
                   token_embedding_f16_.size() * sizeof(uint16_t) +
                   final_norm_.size() * sizeof(float) +
                   lm_head_.storage_bytes();
    for (const auto& lw : layers_) {
        total += lw.attn_norm.size() * sizeof(float) + lw.ffn_norm.size() * sizeof(float);
        total += lw.wq.storage_bytes() + lw.wk.storage_bytes() +
                 lw.wv.storage_bytes() + lw.wo.storage_bytes();
        total += lw.w1.storage_bytes() + lw.w2.storage_bytes() + lw.w3.storage_bytes();
    }
    return total;
}

bool LlmEngine::validate_config(const LlmConfig& cfg, std::string* err) const {
    if (cfg.vocab_size == 0 || cfg.dim == 0 || cfg.hidden_dim == 0 ||
        cfg.n_layers == 0 || cfg.n_heads == 0 || cfg.n_kv_heads == 0 ||
        cfg.max_seq_len == 0) {
        set_err(err, "LlmConfig fields must all be positive");
        return false;
    }
    if (cfg.dim % cfg.n_heads != 0) {
        set_err(err, "dim must be divisible by n_heads");
        return false;
    }
    if (cfg.n_heads % cfg.n_kv_heads != 0) {
        set_err(err, "n_heads must be divisible by n_kv_heads");
        return false;
    }
    return true;
}

void LlmEngine::allocate_state() {
    const uint32_t head_dim = cfg_.dim / cfg_.n_heads;
    const uint32_t kv_dim = cfg_.n_kv_heads * head_dim;

    k_cache_.assign(cfg_.n_layers, std::vector<float>(static_cast<size_t>(cfg_.max_seq_len) * kv_dim, 0.0f));
    v_cache_.assign(cfg_.n_layers, std::vector<float>(static_cast<size_t>(cfg_.max_seq_len) * kv_dim, 0.0f));

    x_.assign(cfg_.dim, 0.0f);
    xb_.assign(cfg_.dim, 0.0f);
    xb2_.assign(cfg_.dim, 0.0f);
    q_.assign(cfg_.dim, 0.0f);
    q_rot_.assign(cfg_.dim, 0.0f);
    k_.assign(kv_dim, 0.0f);
    k_rot_.assign(kv_dim, 0.0f);
    v_.assign(kv_dim, 0.0f);
    hb_.assign(cfg_.hidden_dim, 0.0f);
    hb_act_.assign(cfg_.hidden_dim, 0.0f);
    hb2_.assign(cfg_.hidden_dim, 0.0f);
    att_scores_.assign(static_cast<size_t>(cfg_.n_heads) * cfg_.max_seq_len, 0.0f);
    att_probs_.assign(static_cast<size_t>(cfg_.n_heads) * cfg_.max_seq_len, 0.0f);
}

void LlmEngine::reset_kv_cache() {
    for (auto& kc : k_cache_) {
        std::fill(kc.begin(), kc.end(), 0.0f);
    }
    for (auto& vc : v_cache_) {
        std::fill(vc.begin(), vc.end(), 0.0f);
    }
    cached_tokens_.clear();
}

bool LlmEngine::init_demo_tiny_lm(const LlmConfig& cfg,
                                  gpu::VulkanComputeBackend* vk,
                                  std::string* err,
                                  QuantType quant) {
    if (!validate_config(cfg, err)) {
        return false;
    }
    if (vk_) {
        vk_->clear_weight_cache();
    }

    cfg_ = cfg;
    vk_ = vk;
    is_demo_ = true;
    has_vocab_ = false;
    vocab_.clear();
    piece_to_id_.clear();
    cached_tokens_.clear();
    quant_type_ = quant;

    const uint32_t head_dim = cfg_.dim / cfg_.n_heads;
    const uint32_t kv_dim = cfg_.n_kv_heads * head_dim;
    const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(cfg_.dim));
    const float inv_sqrt_hid = 1.0f / std::sqrt(static_cast<float>(cfg_.hidden_dim));

    uint32_t rng = 0x13572468u;
    token_embedding_f16_.clear();
    token_embedding_ = make_seeded_weights(
        static_cast<size_t>(cfg_.vocab_size) * cfg_.dim, rng, 0.25f);

    layers_.resize(cfg_.n_layers);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        auto& lw = layers_[l];
        lw.attn_norm.assign(cfg_.dim, 1.0f);
        lw.ffn_norm.assign(cfg_.dim, 1.0f);

        auto fq = make_seeded_weights(static_cast<size_t>(cfg_.dim) * cfg_.dim, rng, inv_sqrt_dim);
        auto fk = make_seeded_weights(static_cast<size_t>(kv_dim) * cfg_.dim, rng, inv_sqrt_dim);
        auto fv = make_seeded_weights(static_cast<size_t>(kv_dim) * cfg_.dim, rng, inv_sqrt_dim);
        auto fo = make_seeded_weights(static_cast<size_t>(cfg_.dim) * cfg_.dim, rng, inv_sqrt_dim);
        lw.wq = make_weight_matrix(fq.data(), cfg_.dim, cfg_.dim, quant);
        lw.wk = make_weight_matrix(fk.data(), kv_dim, cfg_.dim, quant);
        lw.wv = make_weight_matrix(fv.data(), kv_dim, cfg_.dim, quant);
        lw.wo = make_weight_matrix(fo.data(), cfg_.dim, cfg_.dim, quant);

        auto f1 = make_seeded_weights(static_cast<size_t>(cfg_.hidden_dim) * cfg_.dim, rng, inv_sqrt_dim);
        auto f2 = make_seeded_weights(static_cast<size_t>(cfg_.dim) * cfg_.hidden_dim, rng, inv_sqrt_hid);
        auto f3 = make_seeded_weights(static_cast<size_t>(cfg_.hidden_dim) * cfg_.dim, rng, inv_sqrt_dim);
        lw.w1 = make_weight_matrix(f1.data(), cfg_.hidden_dim, cfg_.dim, quant);
        lw.w2 = make_weight_matrix(f2.data(), cfg_.dim, cfg_.hidden_dim, quant);
        lw.w3 = make_weight_matrix(f3.data(), cfg_.hidden_dim, cfg_.dim, quant);
    }

    final_norm_.assign(cfg_.dim, 1.0f);
    auto fhead = make_seeded_weights(static_cast<size_t>(cfg_.vocab_size) * cfg_.dim, rng, inv_sqrt_dim);
    lm_head_ = make_weight_matrix(fhead.data(), cfg_.vocab_size, cfg_.dim, quant);

    allocate_state();
    initialized_ = true;
    return true;
}

bool LlmEngine::load_vocab_file(const std::string& tok_path, std::string* err) {
    std::ifstream ifs(tok_path, std::ios::binary);
    if (!ifs.is_open()) {
        set_err(err, "Failed to open tokenizer file: " + tok_path);
        return false;
    }

    char magic[4] = {};
    ifs.read(magic, 4);
    if (!ifs || std::memcmp(magic, "ETOK", 4) != 0) {
        set_err(err, "Invalid tokenizer magic (expected 'ETOK')");
        return false;
    }

    uint32_t ver = 0;
    uint32_t vocab_sz = 0;
    uint32_t bos = 1;
    uint32_t eos = 2;
    uint32_t style = 0;
    if (!read_pod(ifs, ver) || ver != 1 ||
        !read_pod(ifs, vocab_sz) ||
        !read_pod(ifs, bos) ||
        !read_pod(ifs, eos) ||
        !read_pod(ifs, style) ||
        vocab_sz == 0) {
        set_err(err, "Invalid ETOK header");
        return false;
    }

    std::vector<VocabEntry> entries(vocab_sz);
    std::unordered_map<std::string, uint32_t> p2i;
    p2i.reserve(vocab_sz);

    for (uint32_t i = 0; i < vocab_sz; ++i) {
        float score = 0.0f;
        uint32_t len = 0;
        if (!read_pod(ifs, score) || !read_pod(ifs, len) || len > 4096u) {
            set_err(err, "Truncated ETOK vocab entry");
            return false;
        }
        std::string piece(len, '\0');
        if (len > 0) {
            ifs.read(&piece[0], static_cast<std::streamsize>(len));
            if (!ifs) {
                set_err(err, "Unexpected EOF reading ETOK piece");
                return false;
            }
        }
        entries[i] = {piece, score};
        if (p2i.find(piece) == p2i.end()) {
            p2i.emplace(piece, i);
        }
    }

    vocab_ = std::move(entries);
    piece_to_id_ = std::move(p2i);
    cfg_.bos_token_id = bos;
    cfg_.eos_token_id = eos;
    tok_style_ = style;
    has_vocab_ = true;
    return true;
}

bool LlmEngine::load_exyllm_file(const std::string& path,
                                 gpu::VulkanComputeBackend* vk,
                                 std::string* err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        set_err(err, "Failed to open .exyllm file: " + path);
        return false;
    }

    char magic[4] = {};
    ifs.read(magic, 4);
    if (!ifs || std::memcmp(magic, "ELLM", 4) != 0) {
        set_err(err, "Invalid .exyllm magic header (expected 'ELLM')");
        return false;
    }

    uint32_t version = 0;
    uint32_t quant_type = 0;
    LlmConfig cfg;
    if (!read_pod(ifs, version) || (version < 1 || version > 3) ||
        !read_pod(ifs, cfg.vocab_size) ||
        !read_pod(ifs, cfg.dim) ||
        !read_pod(ifs, cfg.hidden_dim) ||
        !read_pod(ifs, cfg.n_layers) ||
        !read_pod(ifs, cfg.n_heads) ||
        !read_pod(ifs, cfg.n_kv_heads) ||
        !read_pod(ifs, cfg.max_seq_len) ||
        !read_pod(ifs, cfg.rms_eps) ||
        !read_pod(ifs, cfg.rope_theta) ||
        !read_pod(ifs, quant_type)) {
        set_err(err, "Failed to read .exyllm header");
        return false;
    }

    if (version >= 2) {
        if (!read_pod(ifs, cfg.rope_mode) ||
            !read_pod(ifs, cfg.bos_token_id) ||
            !read_pod(ifs, cfg.eos_token_id)) {
            set_err(err, "Failed to read .exyllm v2 extended header");
            return false;
        }
    }
    bool fp16_emb = false;
    uint32_t head_quant = quant_type;
    if (version >= 3) {
        uint32_t flags = 0;
        if (!read_pod(ifs, flags) ||
            !read_pod(ifs, cfg.alt_eos_token_id)) {
            set_err(err, "Failed to read .exyllm v3 extended header");
            return false;
        }
        cfg.has_qkv_bias = (flags & 1u) != 0u;
        fp16_emb = (flags & 2u) != 0u;
        if ((flags & 4u) != 0u) {
            head_quant = 1u;  // Mixed Q8_0 output head for cleaner logits
        }
    }

    if (!validate_config(cfg, err)) {
        return false;
    }
    if (quant_type > 3u) {
        set_err(err, "Unsupported quant_type in .exyllm header");
        return false;
    }

    const uint32_t head_dim = cfg.dim / cfg.n_heads;
    const uint32_t kv_dim = cfg.n_kv_heads * head_dim;

    std::vector<float> emb;
    std::vector<uint16_t> emb_f16;
    const size_t emb_elems = static_cast<size_t>(cfg.vocab_size) * cfg.dim;
    if (fp16_emb) {
        emb_f16.resize(emb_elems);
        ifs.read(reinterpret_cast<char*>(emb_f16.data()),
                 static_cast<std::streamsize>(emb_elems * sizeof(uint16_t)));
        if (!ifs) {
            set_err(err, "Failed to read FP16 token embeddings from .exyllm file");
            return false;
        }
    } else if (!read_f32_vec(ifs, emb, emb_elems)) {
        set_err(err, "Failed to read token embeddings from .exyllm file");
        return false;
    }

    std::vector<TransformerLayerWeights> layers(cfg.n_layers);
    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        auto& lw = layers[l];
        if (!read_f32_vec(ifs, lw.attn_norm, cfg.dim) ||
            !read_matrix(ifs, quant_type, cfg.dim, cfg.dim, lw.wq, err) ||
            !read_matrix(ifs, quant_type, kv_dim, cfg.dim, lw.wk, err) ||
            !read_matrix(ifs, quant_type, kv_dim, cfg.dim, lw.wv, err)) {
            return false;
        }
        if (cfg.has_qkv_bias) {
            if (!read_f32_vec(ifs, lw.bq, cfg.dim) ||
                !read_f32_vec(ifs, lw.bk, kv_dim) ||
                !read_f32_vec(ifs, lw.bv, kv_dim)) {
                set_err(err, "Failed to read QKV bias vectors from .exyllm v3 file");
                return false;
            }
        }
        if (!read_matrix(ifs, quant_type, cfg.dim, cfg.dim, lw.wo, err) ||
            !read_f32_vec(ifs, lw.ffn_norm, cfg.dim) ||
            !read_matrix(ifs, quant_type, cfg.hidden_dim, cfg.dim, lw.w1, err) ||
            !read_matrix(ifs, quant_type, cfg.dim, cfg.hidden_dim, lw.w2, err) ||
            !read_matrix(ifs, quant_type, cfg.hidden_dim, cfg.dim, lw.w3, err)) {
            return false;
        }
    }

    std::vector<float> fnorm;
    if (!read_f32_vec(ifs, fnorm, cfg.dim)) {
        set_err(err, "Failed to read final_norm from .exyllm file");
        return false;
    }
    WeightMatrix head;
    if (!read_matrix(ifs, head_quant, cfg.vocab_size, cfg.dim, head, err)) {
        return false;
    }

    if (vk_) {
        vk_->clear_weight_cache();
    }

    cfg_ = cfg;
    vk_ = vk;
    is_demo_ = false;
    has_vocab_ = false;
    vocab_.clear();
    piece_to_id_.clear();
    cached_tokens_.clear();
    quant_type_ = static_cast<QuantType>(quant_type);
    token_embedding_ = std::move(emb);
    token_embedding_f16_ = std::move(emb_f16);
    layers_ = std::move(layers);
    final_norm_ = std::move(fnorm);
    lm_head_ = std::move(head);
    allocate_state();
    initialized_ = true;

    const std::string sidecar_tok = path + ".tok";
    if (file_exists(sidecar_tok)) {
        std::string tok_err;
        load_vocab_file(sidecar_tok, &tok_err);
    }
    return true;
}

std::vector<uint32_t> LlmEngine::encode_prompt(const std::string& prompt) const {
    if (!has_vocab_ || vocab_.empty()) {
        std::vector<uint32_t> tokens;
        if (prompt.empty()) {
            tokens.push_back(1u % cfg_.vocab_size);
        } else {
            tokens.reserve(prompt.size());
            for (unsigned char ch : prompt) {
                tokens.push_back(static_cast<uint32_t>(ch) % cfg_.vocab_size);
            }
        }
        return tokens;
    }

    std::vector<uint32_t> tokens;
    tokens.reserve(prompt.size() + 4);
    // Only prepend BOS automatically when prompt doesn't already start with a ChatML special tag
    const bool starts_with_special = (prompt.rfind("<|", 0) == 0);
    if (!starts_with_special && tok_style_ == 0 && cfg_.bos_token_id < cfg_.vocab_size) {
        tokens.push_back(cfg_.bos_token_id);
    }
    if (prompt.empty()) {
        return tokens;
    }

    auto run_bpe_piece = [&](const std::string& norm) {
        std::vector<uint32_t> seq;
        size_t i = 0;
        while (i < norm.size()) {
            const unsigned char b = static_cast<unsigned char>(norm[i]);
            size_t cp_len = 1;
            if ((b & 0xE0u) == 0xC0u) cp_len = 2;
            else if ((b & 0xF0u) == 0xE0u) cp_len = 3;
            else if ((b & 0xF8u) == 0xF0u) cp_len = 4;
            if (i + cp_len > norm.size()) cp_len = 1;

            std::string ch = norm.substr(i, cp_len);
            auto it = piece_to_id_.find(ch);
            if (it != piece_to_id_.end()) {
                seq.push_back(it->second);
            } else {
                for (size_t k = 0; k < ch.size(); ++k) {
                    char byte_tok[16];
                    std::snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>",
                                  static_cast<unsigned char>(ch[k]));
                    auto bit = piece_to_id_.find(byte_tok);
                    if (bit != piece_to_id_.end()) {
                        seq.push_back(bit->second);
                    } else {
                        auto raw_it = piece_to_id_.find(ch.substr(k, 1));
                        if (raw_it != piece_to_id_.end()) {
                            seq.push_back(raw_it->second);
                        }
                    }
                }
            }
            i += cp_len;
        }

        while (seq.size() >= 2) {
            float best_score = -1e30f;
            uint32_t best_id = 0;
            size_t best_pos = seq.size();
            for (size_t p = 0; p + 1 < seq.size(); ++p) {
                if (seq[p] >= vocab_.size() || seq[p + 1] >= vocab_.size()) continue;
                std::string merged = vocab_[seq[p]].piece + vocab_[seq[p + 1]].piece;
                auto it = piece_to_id_.find(merged);
                if (it != piece_to_id_.end()) {
                    const float sc = vocab_[it->second].score;
                    if (sc > best_score) {
                        best_score = sc;
                        best_id = it->second;
                        best_pos = p;
                    }
                }
            }
            if (best_pos == seq.size()) break;
            seq[best_pos] = best_id;
            seq.erase(seq.begin() + static_cast<std::ptrdiff_t>(best_pos + 1));
        }

        for (uint32_t id : seq) {
            if (id < cfg_.vocab_size) tokens.push_back(id);
        }
    };

    auto bpe_encode_chunk = [&](const std::string& chunk) {
        if (tok_style_ == 2) {
            auto is_letter = [](unsigned char b) -> bool {
                return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80u;
            };
            size_t ci = 0;
            const size_t n = chunk.size();
            while (ci < n) {
                const unsigned char c = static_cast<unsigned char>(chunk[ci]);
                if (c == '\r' || c == '\n') {
                    size_t end = ci + 1;
                    while (end < n && (chunk[end] == '\r' || chunk[end] == '\n')) ++end;
                    run_bpe_piece(chunk.substr(ci, end - ci));
                    ci = end;
                } else if (is_letter(c) || (c == ' ' && ci + 1 < n && is_letter(static_cast<unsigned char>(chunk[ci + 1])))) {
                    size_t end = ci + 1;
                    while (end < n && is_letter(static_cast<unsigned char>(chunk[end]))) ++end;
                    run_bpe_piece(chunk.substr(ci, end - ci));
                    ci = end;
                } else if (c >= '0' && c <= '9') {
                    run_bpe_piece(chunk.substr(ci, 1));
                    ++ci;
                } else if (c == ' ' || c == '\t') {
                    size_t end = ci + 1;
                    while (end < n && (chunk[end] == ' ' || chunk[end] == '\t')) {
                        if (end + 1 < n && is_letter(static_cast<unsigned char>(chunk[end + 1]))) break;
                        ++end;
                    }
                    run_bpe_piece(chunk.substr(ci, end - ci));
                    ci = end;
                } else {
                    run_bpe_piece(chunk.substr(ci, 1));
                    ++ci;
                }
            }
            return;
        }

        std::string norm;
        if (tok_style_ == 0) {
            norm.reserve(chunk.size() * 2 + 4);
            if (piece_to_id_.find("\xE2\x96\x81") != piece_to_id_.end()) {
                norm += "\xE2\x96\x81";
            }
            for (unsigned char c : chunk) {
                if (c == ' ') norm += "\xE2\x96\x81";
                else norm.push_back(static_cast<char>(c));
            }
        } else {
            for (unsigned char c : chunk) {
                if (c == ' ') norm += "\xC4\xA0";
                else if (c == '\n') norm += "\xC4\x8A";
                else norm.push_back(static_cast<char>(c));
            }
        }
        run_bpe_piece(norm);
    };

    // Match known atomic special tokens (<|im_start|>, <|im_end|>, <|endoftext|>, <s>, </s>)
    static const char* kSpecialTags[] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"
    };
    size_t pos = 0;
    while (pos < prompt.size()) {
        size_t next_tag_pos = std::string::npos;
        size_t next_tag_len = 0;
        uint32_t next_tag_id = 0;
        for (const char* tag : kSpecialTags) {
            auto it = piece_to_id_.find(tag);
            if (it == piece_to_id_.end()) continue;
            size_t found = prompt.find(tag, pos);
            if (found != std::string::npos && (next_tag_pos == std::string::npos || found < next_tag_pos)) {
                next_tag_pos = found;
                next_tag_len = std::strlen(tag);
                next_tag_id = it->second;
            }
        }
        if (next_tag_pos == std::string::npos) {
            bpe_encode_chunk(prompt.substr(pos));
            break;
        }
        if (next_tag_pos > pos) {
            bpe_encode_chunk(prompt.substr(pos, next_tag_pos - pos));
        }
        if (next_tag_id < cfg_.vocab_size) {
            tokens.push_back(next_tag_id);
        }
        pos = next_tag_pos + next_tag_len;
    }

    if (tokens.empty()) {
        tokens.push_back(cfg_.bos_token_id < cfg_.vocab_size ? cfg_.bos_token_id : 0u);
    }
    return tokens;
}

std::string LlmEngine::decode_token(uint32_t token_id) const {
    if (!has_vocab_ || token_id >= vocab_.size()) {
        return std::string(1, static_cast<char>(token_id & 0xFFu));
    }
    std::string piece = vocab_[token_id].piece;
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        char* endptr = nullptr;
        unsigned long val = std::strtoul(piece.c_str() + 3, &endptr, 16);
        if (endptr && *endptr == '>') {
            return std::string(1, static_cast<char>(val & 0xFFul));
        }
    }
    replace_all(piece, "\xE2\x96\x81", " ");
    replace_all(piece, "\xC4\xA0", " ");
    replace_all(piece, "\xC4\x8A", "\n");
    return piece;
}

bool LlmEngine::run_rmsnorm(const float* x, const float* w, float* out,
                            uint32_t cols, std::string* err) {
    if (vk_ && vk_->is_initialized()) {
        return vk_->rmsnorm(x, w, out, 1, cols, cfg_.rms_eps, nullptr, err, true);
    }
    double sum_sq = 0.0;
    for (uint32_t c = 0; c < cols; ++c) {
        const double v = static_cast<double>(x[c]);
        sum_sq += v * v;
    }
    const double scale =
        1.0 / std::sqrt(sum_sq / static_cast<double>(cols) + static_cast<double>(cfg_.rms_eps));
    for (uint32_t c = 0; c < cols; ++c) {
        out[c] = static_cast<float>(static_cast<double>(x[c]) * scale * static_cast<double>(w[c]));
    }
    return true;
}

bool LlmEngine::run_matvec(const WeightMatrix& w, const float* x, float* y,
                           std::string* err) {
    if (w.quant == QuantType::Q8_0) {
        const uint32_t pad_cols = w.q8.padded_cols();
        if (vk_ && vk_->is_initialized()) {
            const float* x_ptr = x;
            if (pad_cols != w.cols) {
                x_pad_.assign(pad_cols, 0.0f);
                std::memcpy(x_pad_.data(), x, static_cast<size_t>(w.cols) * sizeof(float));
                x_ptr = x_pad_.data();
            }
            return vk_->matvec_q8_0(w.q8.packed_i8.data(), w.q8.scales.data(), x_ptr, y,
                                    w.rows, pad_cols, nullptr, err, true);
        }
        const uint32_t words_per_row = pad_cols / 4u;
        const uint32_t blocks_per_row = pad_cols / 32u;
        for (uint32_t m = 0; m < w.rows; ++m) {
            double acc = 0.0;
            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                const float scale = w.q8.scales[static_cast<size_t>(m) * blocks_per_row + b];
                const uint32_t k_start = b * 32u;
                double block_dot = 0.0;
                for (uint32_t g = 0; g < 8u; ++g) {
                    const uint32_t word = static_cast<uint32_t>(
                        w.q8.packed_i8[static_cast<size_t>(m) * words_per_row + b * 8u + g]);
                    for (uint32_t byte_i = 0; byte_i < 4u; ++byte_i) {
                        const uint32_t k = k_start + g * 4u + byte_i;
                        if (k < w.cols) {
                            const int8_t q = static_cast<int8_t>((word >> (byte_i * 8u)) & 0xFFu);
                            block_dot += static_cast<double>(q) * static_cast<double>(x[k]);
                        }
                    }
                }
                acc += static_cast<double>(scale) * block_dot;
            }
            y[m] = static_cast<float>(acc);
        }
        return true;
    }

    if (w.quant == QuantType::Q4_0) {
        const uint32_t pad_cols = w.q4.padded_cols();
        if (vk_ && vk_->is_initialized()) {
            const float* x_ptr = x;
            if (pad_cols != w.cols) {
                x_pad_.assign(pad_cols, 0.0f);
                std::memcpy(x_pad_.data(), x, static_cast<size_t>(w.cols) * sizeof(float));
                x_ptr = x_pad_.data();
            }
            return vk_->matvec_q4_0(w.q4.packed_q4.data(), w.q4.scales.data(), x_ptr, y,
                                    w.rows, pad_cols, nullptr, err, true);
        }
        const uint32_t words_per_row = pad_cols / 8u;
        const uint32_t blocks_per_row = pad_cols / 32u;
        for (uint32_t m = 0; m < w.rows; ++m) {
            double acc = 0.0;
            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                const float scale = w.q4.scales[static_cast<size_t>(m) * blocks_per_row + b];
                const uint32_t k_start = b * 32u;
                double block_dot = 0.0;
                for (uint32_t g = 0; g < 4u; ++g) {
                    const uint32_t word =
                        w.q4.packed_q4[static_cast<size_t>(m) * words_per_row + b * 4u + g];
                    for (uint32_t nib = 0; nib < 8u; ++nib) {
                        const uint32_t k = k_start + g * 8u + nib;
                        if (k < w.cols) {
                            const int32_t q = static_cast<int32_t>((word >> (nib * 4u)) & 0xFu) - 8;
                            block_dot += static_cast<double>(q) * static_cast<double>(x[k]);
                        }
                    }
                }
                acc += static_cast<double>(scale) * block_dot;
            }
            y[m] = static_cast<float>(acc);
        }
        return true;
    }

    if (w.quant == QuantType::FP16) {
        const uint32_t pad_cols = w.f16.padded_cols();
        if (vk_ && vk_->is_initialized()) {
            const float* x_ptr = x;
            if (pad_cols != w.cols) {
                x_pad_.assign(pad_cols, 0.0f);
                std::memcpy(x_pad_.data(), x, static_cast<size_t>(w.cols) * sizeof(float));
                x_ptr = x_pad_.data();
            }
            return vk_->matvec_f16(w.f16.packed_f16.data(), x_ptr, y,
                                   w.rows, pad_cols, nullptr, err, true);
        }
        const uint32_t words_per_row = pad_cols / 2u;
        for (uint32_t m = 0; m < w.rows; ++m) {
            double acc = 0.0;
            for (uint32_t p = 0; p < words_per_row; ++p) {
                const uint32_t word = w.f16.packed_f16[static_cast<size_t>(m) * words_per_row + p];
                const uint32_t k0 = p * 2u;
                const uint32_t k1 = k0 + 1u;
                acc += static_cast<double>(float16_to_float32(static_cast<uint16_t>(word & 0xFFFFu))) *
                       static_cast<double>(x[k0]);
                if (k1 < w.cols) {
                    acc += static_cast<double>(float16_to_float32(static_cast<uint16_t>((word >> 16) & 0xFFFFu))) *
                           static_cast<double>(x[k1]);
                }
            }
            y[m] = static_cast<float>(acc);
        }
        return true;
    }

    // F32
    if (vk_ && vk_->is_initialized()) {
        // Compute x * W^T via matmul(x[1, K], W^T)? matmul expects B[K, N]; simpler to keep CPU or transpose,
        // wait: matmul(W[M, K], x[K, 1]) IS matrix * column-vector! Let's call vk_->matmul(w, x, y, M, 1, K)!
        return vk_->matmul(w.f32.data(), x, y, w.rows, 1u, w.cols, nullptr, err);
    }
    for (uint32_t m = 0; m < w.rows; ++m) {
        const float* row_ptr = w.f32.data() + static_cast<size_t>(m) * w.cols;
        double acc = 0.0;
        for (uint32_t k = 0; k < w.cols; ++k) {
            acc += static_cast<double>(row_ptr[k]) * static_cast<double>(x[k]);
        }
        y[m] = static_cast<float>(acc);
    }
    return true;
}

bool LlmEngine::run_rope(const float* in, float* out, uint32_t num_heads,
                         uint32_t head_dim, uint32_t pos, std::string* err) {
    if (vk_ && vk_->is_initialized()) {
        return vk_->rope(in, out, num_heads, head_dim, pos, cfg_.rope_theta,
                         nullptr, err, cfg_.rope_mode);
    }
    const uint32_t half_dim = head_dim / 2u;
    for (uint32_t h = 0; h < num_heads; ++h) {
        const size_t h_base = static_cast<size_t>(h) * head_dim;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const size_t i0 = (cfg_.rope_mode == 1u) ? (h_base + j) : (h_base + 2u * j);
            const size_t i1 = (cfg_.rope_mode == 1u) ? (h_base + half_dim + j) : (i0 + 1u);
            const double exponent = -2.0 * static_cast<double>(j) / static_cast<double>(head_dim);
            const double angle =
                static_cast<double>(pos) * std::pow(static_cast<double>(cfg_.rope_theta), exponent);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const double x0 = static_cast<double>(in[i0]);
            const double x1 = static_cast<double>(in[i1]);
            out[i0] = static_cast<float>(x0 * c - x1 * s);
            out[i1] = static_cast<float>(x0 * s + x1 * c);
        }
        if ((head_dim & 1u) != 0u) {
            out[h_base + head_dim - 1u] = in[h_base + head_dim - 1u];
        }
    }
    return true;
}

bool LlmEngine::run_silu(const float* in, float* out, uint32_t count, std::string* err) {
    if (vk_ && vk_->is_initialized()) {
        return vk_->silu(in, out, count, nullptr, err);
    }
    for (uint32_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(in[i]);
        out[i] = static_cast<float>(v / (1.0 + std::exp(-v)));
    }
    return true;
}

bool LlmEngine::run_softmax(const float* in, float* out, uint32_t rows, uint32_t cols,
                            std::string* err) {
    if (vk_ && vk_->is_initialized()) {
        return vk_->softmax(in, out, rows, cols, nullptr, err);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const size_t base = static_cast<size_t>(r) * cols;
        double row_max = static_cast<double>(in[base]);
        for (uint32_t c = 1; c < cols; ++c) {
            row_max = std::max(row_max, static_cast<double>(in[base + c]));
        }
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; ++c) {
            sum += std::exp(static_cast<double>(in[base + c]) - row_max);
        }
        for (uint32_t c = 0; c < cols; ++c) {
            out[base + c] = static_cast<float>(
                std::exp(static_cast<double>(in[base + c]) - row_max) / sum);
        }
    }
    return true;
}

bool LlmEngine::forward_token(uint32_t token_id, uint32_t pos,
                              std::vector<float>& logits, std::string* err) {
    if (!initialized_) {
        set_err(err, "LlmEngine is not initialized");
        return false;
    }
    if (token_id >= cfg_.vocab_size) {
        set_err(err, "token_id out of range");
        return false;
    }
    if (pos >= cfg_.max_seq_len) {
        set_err(err, "Sequence position exceeds max_seq_len");
        return false;
    }

    const uint32_t dim = cfg_.dim;
    const uint32_t head_dim = dim / cfg_.n_heads;
    const uint32_t kv_dim = cfg_.n_kv_heads * head_dim;
    const uint32_t kv_mul = cfg_.n_heads / cfg_.n_kv_heads;
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const uint32_t seq_len = pos + 1u;

    if (!token_embedding_f16_.empty()) {
        const size_t base = static_cast<size_t>(token_id) * dim;
        for (uint32_t i = 0; i < dim; ++i) {
            x_[i] = float16_to_float32(token_embedding_f16_[base + i]);
        }
    } else {
        std::memcpy(x_.data(),
                    token_embedding_.data() + static_cast<size_t>(token_id) * dim,
                    static_cast<size_t>(dim) * sizeof(float));
    }

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const auto& lw = layers_[l];

        if (!run_rmsnorm(x_.data(), lw.attn_norm.data(), xb_.data(), dim, err)) {
            return false;
        }
        if (!run_matvec(lw.wq, xb_.data(), q_.data(), err) ||
            !run_matvec(lw.wk, xb_.data(), k_.data(), err) ||
            !run_matvec(lw.wv, xb_.data(), v_.data(), err)) {
            return false;
        }
        if (cfg_.has_qkv_bias) {
            for (uint32_t i = 0; i < dim && i < lw.bq.size(); ++i) q_[i] += lw.bq[i];
            for (uint32_t i = 0; i < kv_dim && i < lw.bk.size(); ++i) k_[i] += lw.bk[i];
            for (uint32_t i = 0; i < kv_dim && i < lw.bv.size(); ++i) v_[i] += lw.bv[i];
        }
        if (!run_rope(q_.data(), q_rot_.data(), cfg_.n_heads, head_dim, pos, err) ||
            !run_rope(k_.data(), k_rot_.data(), cfg_.n_kv_heads, head_dim, pos, err)) {
            return false;
        }

        const size_t cache_off = static_cast<size_t>(pos) * kv_dim;
        std::memcpy(k_cache_[l].data() + cache_off, k_rot_.data(),
                    static_cast<size_t>(kv_dim) * sizeof(float));
        std::memcpy(v_cache_[l].data() + cache_off, v_.data(),
                    static_cast<size_t>(kv_dim) * sizeof(float));

        for (uint32_t h = 0; h < cfg_.n_heads; ++h) {
            const float* qh = q_rot_.data() + static_cast<size_t>(h) * head_dim;
            const uint32_t kv_h = h / kv_mul;
            float* att_row = att_scores_.data() + static_cast<size_t>(h) * seq_len;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const float* kh = k_cache_[l].data() +
                                  static_cast<size_t>(t) * kv_dim +
                                  static_cast<size_t>(kv_h) * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(qh[d]) * static_cast<double>(kh[d]);
                }
                att_row[t] = static_cast<float>(dot) * inv_sqrt_hd;
            }
        }

        if (!run_softmax(att_scores_.data(), att_probs_.data(), cfg_.n_heads, seq_len, err)) {
            return false;
        }

        std::fill(xb_.begin(), xb_.end(), 0.0f);
        for (uint32_t h = 0; h < cfg_.n_heads; ++h) {
            float* xbh = xb_.data() + static_cast<size_t>(h) * head_dim;
            const uint32_t kv_h = h / kv_mul;
            const float* probs = att_probs_.data() + static_cast<size_t>(h) * seq_len;
            for (uint32_t t = 0; t < seq_len; ++t) {
                const float p = probs[t];
                const float* vh = v_cache_[l].data() +
                                  static_cast<size_t>(t) * kv_dim +
                                  static_cast<size_t>(kv_h) * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    xbh[d] += p * vh[d];
                }
            }
        }

        if (!run_matvec(lw.wo, xb_.data(), xb2_.data(), err)) {
            return false;
        }
        for (uint32_t i = 0; i < dim; ++i) {
            x_[i] += xb2_[i];
        }

        if (!run_rmsnorm(x_.data(), lw.ffn_norm.data(), xb_.data(), dim, err)) {
            return false;
        }
        if (!run_matvec(lw.w1, xb_.data(), hb_.data(), err) ||
            !run_matvec(lw.w3, xb_.data(), hb2_.data(), err)) {
            return false;
        }
        if (!run_silu(hb_.data(), hb_act_.data(), cfg_.hidden_dim, err)) {
            return false;
        }
        for (uint32_t i = 0; i < cfg_.hidden_dim; ++i) {
            hb_act_[i] *= hb2_[i];
        }
        if (!run_matvec(lw.w2, hb_act_.data(), xb_.data(), err)) {
            return false;
        }
        for (uint32_t i = 0; i < dim; ++i) {
            x_[i] += xb_[i];
        }
    }

    if (!run_rmsnorm(x_.data(), final_norm_.data(), xb_.data(), dim, err)) {
        return false;
    }

    logits.resize(cfg_.vocab_size);
    return run_matvec(lm_head_, xb_.data(), logits.data(), err);
}

uint32_t LlmEngine::sample_token(const std::vector<float>& logits,
                                 float temperature,
                                 uint64_t& rng_state) const {
    if (logits.empty()) {
        return 0;
    }
    if (temperature <= 0.0f) {
        uint32_t best_idx = 0;
        float best_val = logits[0];
        for (uint32_t i = 1; i < static_cast<uint32_t>(logits.size()); ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    const double inv_t = 1.0 / static_cast<double>(temperature);
    double max_v = static_cast<double>(logits[0]) * inv_t;
    for (size_t i = 1; i < logits.size(); ++i) {
        max_v = std::max(max_v, static_cast<double>(logits[i]) * inv_t);
    }

    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double p = std::exp(static_cast<double>(logits[i]) * inv_t - max_v);
        probs[i] = p;
        sum += p;
    }

    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u = static_cast<double>(rng_state >> 11) * (1.0 / 9007199254740992.0);
    double cdf = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cdf += probs[i] / sum;
        if (u <= cdf) {
            return static_cast<uint32_t>(i);
        }
    }
    return static_cast<uint32_t>(probs.size() - 1);
}

namespace {

size_t utf8_complete_prefix_len(const std::string& s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        size_t need = 1;
        if (b < 0x80u) {
            need = 1;
        } else if ((b & 0xE0u) == 0xC0u) {
            need = 2;
        } else if ((b & 0xF0u) == 0xE0u) {
            need = 3;
        } else if ((b & 0xF8u) == 0xF0u) {
            need = 4;
        } else {
            // Unexpected continuation/invalid byte: emit immediately rather than stalling.
            ++i;
            continue;
        }
        if (i + need > n) {
            break;
        }
        bool valid_cont = true;
        for (size_t k = 1; k < need; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) {
                valid_cont = false;
                break;
            }
        }
        i += valid_cont ? need : 1u;
    }
    return i;
}

template <typename T>
bool write_pod_bin(std::ofstream& ofs, const T& v) {
    ofs.write(reinterpret_cast<const char*>(&v), sizeof(T));
    return static_cast<bool>(ofs);
}

bool write_matrix_bin(std::ofstream& ofs, const float* data, uint32_t rows, uint32_t cols,
                      QuantType quant) {
    switch (quant) {
        case QuantType::F32:
            ofs.write(reinterpret_cast<const char*>(data),
                      static_cast<std::streamsize>(static_cast<size_t>(rows) * cols * sizeof(float)));
            return static_cast<bool>(ofs);
        case QuantType::Q8_0: {
            auto q8 = quantize_q8_0(data, rows, cols);
            if (!q8.packed_i8.empty()) {
                ofs.write(reinterpret_cast<const char*>(q8.packed_i8.data()),
                          static_cast<std::streamsize>(q8.packed_i8.size() * sizeof(int32_t)));
            }
            if (!q8.scales.empty()) {
                ofs.write(reinterpret_cast<const char*>(q8.scales.data()),
                          static_cast<std::streamsize>(q8.scales.size() * sizeof(float)));
            }
            return static_cast<bool>(ofs);
        }
        case QuantType::Q4_0: {
            auto q4 = quantize_q4_0(data, rows, cols);
            if (!q4.packed_q4.empty()) {
                ofs.write(reinterpret_cast<const char*>(q4.packed_q4.data()),
                          static_cast<std::streamsize>(q4.packed_q4.size() * sizeof(uint32_t)));
            }
            if (!q4.scales.empty()) {
                ofs.write(reinterpret_cast<const char*>(q4.scales.data()),
                          static_cast<std::streamsize>(q4.scales.size() * sizeof(float)));
            }
            return static_cast<bool>(ofs);
        }
        case QuantType::FP16: {
            auto f16 = quantize_f16(data, rows, cols);
            if (!f16.packed_f16.empty()) {
                ofs.write(reinterpret_cast<const char*>(f16.packed_f16.data()),
                          static_cast<std::streamsize>(f16.packed_f16.size() * sizeof(uint32_t)));
            }
            return static_cast<bool>(ofs);
        }
    }
    return false;
}

}  // namespace

GenerationResult LlmEngine::generate(const std::string& prompt,
                                     uint32_t max_new_tokens,
                                     float temperature,
                                     StreamCallback on_token) {
    GenerationResult res;
    if (!initialized_) {
        return res;
    }

    res.used_vulkan_gpu = (vk_ != nullptr && vk_->is_initialized());
    res.is_demo_weights = is_demo_;
    res.used_bpe_vocab = has_vocab_;
    res.quant_name = quant_type_name(quant_type_);
    res.model_bytes = model_bytes();

    std::vector<uint32_t> prompt_tokens = encode_prompt(prompt);
    if (prompt_tokens.empty()) {
        return res;
    }
    if (prompt_tokens.size() >= cfg_.max_seq_len) {
        const uint32_t keep = (cfg_.max_seq_len > 32u) ? (cfg_.max_seq_len - 32u) : (cfg_.max_seq_len - 1u);
        if (prompt_tokens.size() > keep && keep > 0u) {
            prompt_tokens.erase(prompt_tokens.begin(),
                                prompt_tokens.end() - static_cast<std::ptrdiff_t>(keep));
        } else {
            prompt_tokens.resize(cfg_.max_seq_len - 1u);
        }
        reset_kv_cache();
    }

    // Reuse matching KV-cache prefix across consecutive multi-turn prompts;
    // keep at least the final prompt token uncached so logits reflect it.
    size_t lcp = 0;
    while (lcp < cached_tokens_.size() &&
           lcp + 1u < prompt_tokens.size() &&
           cached_tokens_[lcp] == prompt_tokens[lcp]) {
        ++lcp;
    }
    if (lcp == 0u) {
        reset_kv_cache();
    } else {
        cached_tokens_.resize(lcp);
    }
    res.cached_prefix_tokens = static_cast<uint32_t>(lcp);

    std::vector<float> logits;
    std::string err;
    uint32_t pos = static_cast<uint32_t>(lcp);

    const auto t_prefill_start = std::chrono::steady_clock::now();
    for (size_t i = lcp; i < prompt_tokens.size(); ++i) {
        if (!forward_token(prompt_tokens[i], pos, logits, &err)) {
            reset_kv_cache();
            return res;
        }
        cached_tokens_.push_back(prompt_tokens[i]);
        ++pos;
    }
    const auto t_prefill_end = std::chrono::steady_clock::now();
    res.prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    uint64_t rng_state = 0xCAFEBABE12345678ULL;
    const uint32_t avail = (pos < cfg_.max_seq_len) ? (cfg_.max_seq_len - pos) : 0u;
    const uint32_t to_gen = std::min(max_new_tokens, avail);

    const auto t_decode_start = std::chrono::steady_clock::now();
    std::vector<uint32_t> recent_tokens;
    recent_tokens.reserve(to_gen);
    std::string utf8_pending;
    for (uint32_t step = 0; step < to_gen; ++step) {
        if (!logits.empty()) {
            const size_t hist_start = recent_tokens.size() > 48u ? recent_tokens.size() - 48u : 0u;
            for (size_t hi = hist_start; hi < recent_tokens.size(); ++hi) {
                const uint32_t rt = recent_tokens[hi];
                if (rt < logits.size()) {
                    logits[rt] = (logits[rt] > 0.0f) ? (logits[rt] / 1.12f) : (logits[rt] * 1.12f);
                }
            }
        }
        const uint32_t next_tok = sample_token(logits, temperature, rng_state);
        if (has_vocab_ && (next_tok == cfg_.eos_token_id ||
                           (cfg_.alt_eos_token_id != 0u && next_tok == cfg_.alt_eos_token_id))) {
            break;
        }
        recent_tokens.push_back(next_tok);
        const std::string piece = decode_token(next_tok);
        if (has_vocab_ && (piece == "<|im_end|>" || piece == "<|endoftext|>" || piece == "</s>")) {
            break;
        }
        res.text += piece;
        utf8_pending += piece;
        ++res.tokens_generated;

        if (on_token) {
            const size_t ready = utf8_complete_prefix_len(utf8_pending);
            const std::string emit = (ready > 0u) ? utf8_pending.substr(0, ready) : std::string();
            if (ready > 0u) {
                utf8_pending.erase(0, ready);
            }
            if (!on_token(emit, res.tokens_generated)) {
                res.stopped_early = true;
                break;
            }
        }

        if (pos >= cfg_.max_seq_len || step + 1u >= to_gen) {
            break;
        }
        if (!forward_token(next_tok, pos, logits, &err)) {
            break;
        }
        cached_tokens_.push_back(next_tok);
        ++pos;
    }
    if (on_token && !utf8_pending.empty() && !res.stopped_early) {
        on_token(utf8_pending, res.tokens_generated);
    }
    const auto t_decode_end = std::chrono::steady_clock::now();
    res.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    res.tokens_per_sec = (res.decode_ms > 0.0)
        ? (static_cast<double>(res.tokens_generated) / (res.decode_ms / 1000.0))
        : 0.0;

    return res;
}

bool convert_karpathy_bin_to_exyllm(const std::string& bin_path,
                                    const std::string& tok_bin_path,
                                    const std::string& out_exyllm_path,
                                    QuantType quant,
                                    std::string* err) {
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open()) {
        set_err(err, "Failed to open Karpathy .bin file: " + bin_path);
        return false;
    }

    int32_t hdr[7] = {};
    if (!ifs.read(reinterpret_cast<char*>(hdr), sizeof(hdr))) {
        set_err(err, "Truncated Karpathy .bin header");
        return false;
    }
    const int32_t dim_i = hdr[0];
    const int32_t hidden_i = hdr[1];
    const int32_t n_layers_i = hdr[2];
    const int32_t n_heads_i = hdr[3];
    const int32_t n_kv_heads_i = hdr[4];
    const int32_t vocab_raw = hdr[5];
    const int32_t seq_len_i = hdr[6];

    const bool shared_weights = (vocab_raw > 0);
    const int32_t vocab_i = (vocab_raw < 0) ? -vocab_raw : vocab_raw;
    if (dim_i <= 0 || hidden_i <= 0 || n_layers_i <= 0 || n_heads_i <= 0 ||
        n_kv_heads_i <= 0 || vocab_i <= 0 || seq_len_i <= 0 ||
        (dim_i % n_heads_i) != 0 || (n_heads_i % n_kv_heads_i) != 0) {
        set_err(err, "Invalid hyperparameters in Karpathy .bin header");
        return false;
    }

    const uint32_t dim = static_cast<uint32_t>(dim_i);
    const uint32_t hidden_dim = static_cast<uint32_t>(hidden_i);
    const uint32_t n_layers = static_cast<uint32_t>(n_layers_i);
    const uint32_t n_heads = static_cast<uint32_t>(n_heads_i);
    const uint32_t n_kv_heads = static_cast<uint32_t>(n_kv_heads_i);
    const uint32_t vocab_size = static_cast<uint32_t>(vocab_i);
    const uint32_t head_dim = dim / n_heads;
    const uint32_t kv_dim = n_kv_heads * head_dim;
    const uint32_t max_seq = std::min<uint32_t>(static_cast<uint32_t>(seq_len_i), 256u);

    // Byte offsets of each grouped tensor block after 28-byte header
    const uint64_t f32 = sizeof(float);
    const uint64_t off_emb = 28ULL;
    const uint64_t sz_emb = static_cast<uint64_t>(vocab_size) * dim * f32;
    const uint64_t off_rms_att = off_emb + sz_emb;
    const uint64_t sz_rms = static_cast<uint64_t>(n_layers) * dim * f32;
    const uint64_t off_wq = off_rms_att + sz_rms;
    const uint64_t sz_wq_l = static_cast<uint64_t>(dim) * dim * f32;
    const uint64_t off_wk = off_wq + static_cast<uint64_t>(n_layers) * sz_wq_l;
    const uint64_t sz_wk_l = static_cast<uint64_t>(kv_dim) * dim * f32;
    const uint64_t off_wv = off_wk + static_cast<uint64_t>(n_layers) * sz_wk_l;
    const uint64_t off_wo = off_wv + static_cast<uint64_t>(n_layers) * sz_wk_l;
    const uint64_t off_rms_ffn = off_wo + static_cast<uint64_t>(n_layers) * sz_wq_l;
    const uint64_t off_w1 = off_rms_ffn + sz_rms;
    const uint64_t sz_w1_l = static_cast<uint64_t>(hidden_dim) * dim * f32;
    const uint64_t off_w2 = off_w1 + static_cast<uint64_t>(n_layers) * sz_w1_l;
    const uint64_t sz_w2_l = static_cast<uint64_t>(dim) * hidden_dim * f32;
    const uint64_t off_w3 = off_w2 + static_cast<uint64_t>(n_layers) * sz_w2_l;
    const uint64_t off_rms_final = off_w3 + static_cast<uint64_t>(n_layers) * sz_w1_l;
    const uint64_t sz_freq_cis = static_cast<uint64_t>(seq_len_i) * head_dim * f32;
    const uint64_t off_wcls = off_rms_final + static_cast<uint64_t>(dim) * f32 + sz_freq_cis;

    auto read_slice = [&](uint64_t byte_off, size_t count, std::vector<float>& dst) -> bool {
        dst.resize(count);
        ifs.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
        ifs.read(reinterpret_cast<char*>(dst.data()),
                 static_cast<std::streamsize>(count * sizeof(float)));
        return static_cast<bool>(ifs);
    };

    std::ofstream ofs(out_exyllm_path, std::ios::binary);
    if (!ofs.is_open()) {
        set_err(err, "Failed to create output .exyllm file: " + out_exyllm_path);
        return false;
    }

    // Write .exyllm v3 header with flags=2 (FP16 token embeddings for 50% smaller RAM)
    const uint32_t ver = 3u;
    const uint32_t q_u32 = static_cast<uint32_t>(quant);
    const uint32_t rope_mode = 0u;
    const uint32_t bos = 1u;
    const uint32_t eos = 2u;
    const uint32_t flags = 2u;
    const uint32_t alt_eos = 0u;
    const float rms_eps = 1e-5f;
    const float rope_theta = 10000.0f;
    ofs.write("ELLM", 4);
    if (!write_pod_bin(ofs, ver) ||
        !write_pod_bin(ofs, vocab_size) ||
        !write_pod_bin(ofs, dim) ||
        !write_pod_bin(ofs, hidden_dim) ||
        !write_pod_bin(ofs, n_layers) ||
        !write_pod_bin(ofs, n_heads) ||
        !write_pod_bin(ofs, n_kv_heads) ||
        !write_pod_bin(ofs, max_seq) ||
        !write_pod_bin(ofs, rms_eps) ||
        !write_pod_bin(ofs, rope_theta) ||
        !write_pod_bin(ofs, q_u32) ||
        !write_pod_bin(ofs, rope_mode) ||
        !write_pod_bin(ofs, bos) ||
        !write_pod_bin(ofs, eos) ||
        !write_pod_bin(ofs, flags) ||
        !write_pod_bin(ofs, alt_eos)) {
        set_err(err, "Failed to write .exyllm v3 header");
        return false;
    }

    std::vector<float> buf;
    if (!read_slice(off_emb, static_cast<size_t>(vocab_size) * dim, buf)) {
        set_err(err, "Failed to read embedding weights from .bin");
        return false;
    }
    std::vector<uint16_t> emb_f16(buf.size());
    for (size_t i = 0; i < buf.size(); ++i) {
        emb_f16[i] = float32_to_float16(buf[i]);
    }
    ofs.write(reinterpret_cast<const char*>(emb_f16.data()),
              static_cast<std::streamsize>(emb_f16.size() * sizeof(uint16_t)));

    for (uint32_t l = 0; l < n_layers; ++l) {
        const uint64_t lu = static_cast<uint64_t>(l);
        if (!read_slice(off_rms_att + lu * dim * f32, dim, buf)) return false;
        ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(dim * sizeof(float)));

        if (!read_slice(off_wq + lu * sz_wq_l, dim * dim, buf) || !write_matrix_bin(ofs, buf.data(), dim, dim, quant)) return false;
        if (!read_slice(off_wk + lu * sz_wk_l, kv_dim * dim, buf) || !write_matrix_bin(ofs, buf.data(), kv_dim, dim, quant)) return false;
        if (!read_slice(off_wv + lu * sz_wk_l, kv_dim * dim, buf) || !write_matrix_bin(ofs, buf.data(), kv_dim, dim, quant)) return false;
        if (!read_slice(off_wo + lu * sz_wq_l, dim * dim, buf) || !write_matrix_bin(ofs, buf.data(), dim, dim, quant)) return false;

        if (!read_slice(off_rms_ffn + lu * dim * f32, dim, buf)) return false;
        ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(dim * sizeof(float)));

        if (!read_slice(off_w1 + lu * sz_w1_l, hidden_dim * dim, buf) || !write_matrix_bin(ofs, buf.data(), hidden_dim, dim, quant)) return false;
        if (!read_slice(off_w2 + lu * sz_w2_l, dim * hidden_dim, buf) || !write_matrix_bin(ofs, buf.data(), dim, hidden_dim, quant)) return false;
        if (!read_slice(off_w3 + lu * sz_w1_l, hidden_dim * dim, buf) || !write_matrix_bin(ofs, buf.data(), hidden_dim, dim, quant)) return false;
    }

    if (!read_slice(off_rms_final, dim, buf)) {
        set_err(err, "Failed to read rms_final from .bin");
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(dim * sizeof(float)));

    const uint64_t wcls_off = shared_weights ? off_emb : off_wcls;
    if (!read_slice(wcls_off, static_cast<size_t>(vocab_size) * dim, buf) ||
        !write_matrix_bin(ofs, buf.data(), vocab_size, dim, quant)) {
        set_err(err, "Failed to read/write lm_head from .bin");
        return false;
    }
    if (!ofs) {
        set_err(err, "Write error while creating .exyllm");
        return false;
    }

    if (!tok_bin_path.empty()) {
        std::ifstream tifs(tok_bin_path, std::ios::binary);
        if (!tifs.is_open()) {
            set_err(err, "Failed to open tokenizer .bin: " + tok_bin_path);
            return false;
        }
        int32_t max_tok_len = 0;
        if (!read_pod(tifs, max_tok_len)) {
            set_err(err, "Invalid tokenizer .bin header");
            return false;
        }
        const std::string out_tok = out_exyllm_path + ".tok";
        std::ofstream tofs(out_tok, std::ios::binary);
        if (!tofs.is_open()) {
            set_err(err, "Failed to create sidecar tokenizer: " + out_tok);
            return false;
        }
        tofs.write("ETOK", 4);
        const uint32_t etok_ver = 1u, etok_style = 2u;
        write_pod_bin(tofs, etok_ver);
        write_pod_bin(tofs, vocab_size);
        write_pod_bin(tofs, bos);
        write_pod_bin(tofs, eos);
        write_pod_bin(tofs, etok_style);
        for (uint32_t i = 0; i < vocab_size; ++i) {
            float sc = 0.0f;
            int32_t slen = 0;
            if (!read_pod(tifs, sc) || !read_pod(tifs, slen) || slen < 0 || slen > 4096) {
                set_err(err, "Truncated tokenizer.bin entry");
                return false;
            }
            std::string piece(static_cast<size_t>(slen), '\0');
            if (slen > 0 && !tifs.read(&piece[0], slen)) {
                set_err(err, "Unexpected EOF reading tokenizer.bin piece");
                return false;
            }
            const uint32_t ulen = static_cast<uint32_t>(slen);
            write_pod_bin(tofs, sc);
            write_pod_bin(tofs, ulen);
            if (ulen > 0) {
                tofs.write(piece.data(), slen);
            }
        }
        if (!tofs) {
            set_err(err, "Failed writing sidecar .tok file");
            return false;
        }
    }
    return true;
}

}  // namespace llm
}  // namespace exynova
