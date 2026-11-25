#include "exynova/cpu_ref.h"

#include <algorithm>
#include <cstring>

namespace exynova {
namespace runtime {
namespace cpu_ref {

// ---- dispatch ---------------------------------------------------------------

template<typename Op>
void compute(const float* input_a, const float* input_b,
             float*       output,
             size_t       count)
{
    switch (Op::id) {
    case OpAdd::id:
        std::transform(input_a, input_a + count,
                       input_b, output,
                       [](float a, float b) { return a + b; });
        break;
    case OpMul::id:
        std::transform(input_a, input_a + count,
                       input_b, output,
                       [](float a, float b) { return a * b; });
        break;
    case OpDotProd::id: {
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i)
            sum += static_cast<double>(input_a[i]) * static_cast<double>(input_b[i]);
        *output = static_cast<float>(sum);
        break;
    }
    default:
        throw std::domain_error("unknown cpu_ref op");
    }
}

// explicit instantiations — avoid ODR violations across translation units
template void compute<OpAdd>     (const float*, const float*, float*, size_t);
template void compute<OpMul>     (const float*, const float*, float*, size_t);
template void compute<OpDotProd> (const float*, const float*, float*, size_t);

} // namespace cpu_ref
} // namespace runtime
} // namespace exynova
