// Internal verification entry points for the exact upstream spline/solve hooks.
#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <torch/library.h>

namespace {
void vector(const at::Tensor& t) {
    TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided && t.dim() == 1,
                "expected dense CPU float64 vector");
}
QuantLib::Array array(const at::Tensor& t) {
    QuantLib::Array result(t.numel());
    quantlibtorch::unpack_array(t, result.begin());
    return result;
}
at::Tensor pack(const std::vector<QuantLib::Real>& v) { return quantlibtorch::pack_array(v.data(), v.size()); }

std::vector<at::Tensor> spline_probe(const at::Tensor& knots, const at::Tensor& values,
    const at::Tensor& queries, bool monotonic, int64_t kernels, bool adjoint) {
    for (const auto& t : {knots, values, queries}) vector(t);
    TORCH_CHECK(knots.numel() >= 2 && knots.sizes() == values.sizes() && queries.numel() >= 1, "invalid spline sizes");
    TORCH_CHECK(at::isfinite(knots).all().item<bool>() &&
                (knots.slice(0, 1)-knots.slice(0, 0, -1) > 0).all().item<bool>(), "knots must increase");
    quantlibtorch::TensorKernelMode kernel_guard(static_cast<int>(kernels));
    quantlibtorch::AdjointMode adjoint_guard(adjoint);
    auto x = array(knots), y = array(values), q = array(queries);
    QuantLib::CubicInterpolation spline(x.begin(), x.end(), y.begin(), QuantLib::CubicInterpolation::Spline,
        monotonic, QuantLib::CubicInterpolation::SecondDerivative, 0., QuantLib::CubicInterpolation::SecondDerivative, 0.);
    std::vector<at::Tensor> v, d, dd, primitive;
    for (const auto& point : q) {
        v.push_back(spline(point, true).tensor());
        d.push_back(spline.derivative(point, true).tensor());
        dd.push_back(spline.secondDerivative(point, true).tensor());
        primitive.push_back(spline.primitive(point, true).tensor());
    }
    std::vector<int64_t> adjusted;
    for (bool a : spline.monotonicityAdjustments()) adjusted.push_back(a);
    return {at::stack({at::stack(v), at::stack(d), at::stack(dd), at::stack(primitive)}),
            at::stack({pack(spline.aCoefficients()), pack(spline.bCoefficients()), pack(spline.cCoefficients())}),
            pack(spline.primitiveConstants()), at::tensor(adjusted, at::TensorOptions().dtype(at::kLong))};
}

at::Tensor tridiagonal_hook(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, bool adjoint, bool alias, int64_t kernels) {
    for (const auto& t : {lower, diagonal, upper, rhs}) vector(t);
    const auto n = diagonal.numel();
    TORCH_CHECK(n >= 2 && lower.numel() == n-1 && upper.numel() == n-1 && rhs.numel() == n, "invalid tridiagonal sizes");
    quantlibtorch::TensorKernelMode kernel_guard(static_cast<int>(kernels));
    quantlibtorch::AdjointMode mode(adjoint);
    QuantLib::TridiagonalOperator op(array(lower), array(diagonal), array(upper));
    auto r = array(rhs);
    QuantLib::Array output(n);
    if (alias) {
        op.solveFor(r, r);
        return quantlibtorch::pack_array(r.begin(), n);
    }
    op.solveFor(r, output);
    return quantlibtorch::pack_array(output.begin(), n);
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_spline_probe(Tensor knots, Tensor values, Tensor queries, bool monotonic, int kernels, bool adjoint) -> Tensor[]");
    m.def("_tridiagonal_hook(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, bool adjoint, bool alias, int kernels=2) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_spline_probe", TORCH_FN(spline_probe));
    m.impl("_tridiagonal_hook", TORCH_FN(tridiagonal_hook));
}
