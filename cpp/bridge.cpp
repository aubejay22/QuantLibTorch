#include <ql/methods/finitedifferences/tridiagonaloperator.hpp>
#include <torch/library.h>
#include <type_traits>

static_assert(std::is_same_v<QuantLib::Real, quantlibtorch::TorchReal>);

namespace {
QuantLib::Array to_array(const at::Tensor& t) {
    QuantLib::Array result(t.numel());
    for (int64_t i = 0; i < t.numel(); ++i)
        result[i] = quantlibtorch::TorchReal(t.select(0, i));
    return result;
}

at::Tensor solve(const at::Tensor& lower, const at::Tensor& diagonal,
                 const at::Tensor& upper, const at::Tensor& rhs) {
    for (const auto& t : {lower, diagonal, upper, rhs}) {
        TORCH_CHECK(t.device().is_cpu(), "QuantLibTorch currently supports CPU tensors only");
        TORCH_CHECK(t.layout() == at::kStrided, "QuantLibTorch requires dense strided tensors");
        TORCH_CHECK(t.scalar_type() == at::kDouble, "QuantLibTorch requires float64 tensors");
        TORCH_CHECK(t.dim() == 1, "QuantLibTorch requires one-dimensional tensors");
        TORCH_CHECK(at::isfinite(t).all().item<bool>(), "QuantLibTorch requires finite inputs");
    }
    const auto n = diagonal.numel();
    TORCH_CHECK(n >= 2, "diagonal must have at least 2 elements");
    TORCH_CHECK(lower.numel() == n - 1 && upper.numel() == n - 1 && rhs.numel() == n,
                "expected lower/upper length n-1 and rhs length n");
    QuantLib::TridiagonalOperator op(to_array(lower), to_array(diagonal), to_array(upper));
    auto x = op.solveFor(to_array(rhs));
    std::vector<at::Tensor> values;
    values.reserve(n);
    for (QuantLib::Size i = 0; i < x.size(); ++i)
        values.push_back(x[i].tensor());
    auto result = at::stack(values);
    TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite tridiagonal solution");
    return result;
}
} // namespace

TORCH_LIBRARY(quantlibtorch, m) {
    m.def("solve(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs) -> Tensor");
}

// Internal ATen arithmetic records the full reverse-mode graph automatically.
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("solve", TORCH_FN(solve));
}
