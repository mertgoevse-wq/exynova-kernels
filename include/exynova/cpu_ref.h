#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace exynova {
namespace runtime {
namespace cpu_ref {

// ---------------------------------------------------------------------------
// Minimal kernel ops — basis for correctness validation
// ---------------------------------------------------------------------------

struct OpAdd     { static constexpr uint32_t id = 0; };
struct OpMul     { static constexpr uint32_t id = 1; };
struct OpDotProd { static constexpr uint32_t id = 2; };

template<typename Op>
void compute(const float* input_a, const float* input_b,
             float*       output,
             size_t       count);

} // namespace cpu_ref
} // namespace runtime
} // namespace exynova
