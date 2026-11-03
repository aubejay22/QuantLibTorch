// Internal adapter verification entry points. These do not expose additional
// pricing functionality; they exercise scalar invariants through the same DLL.
#include <ql/utilities/null.hpp>
#include <torch/library.h>
#include <tuple>
#include <type_traits>

static_assert(std::is_same_v<QuantLib::Real, quantlibtorch::TorchReal>);
static_assert(!std::is_convertible_v<QuantLib::Real, double>);
static_assert(!std::is_convertible_v<QuantLib::Real, int64_t>);

namespace {
std::tuple<double, int64_t> scalar_null() {
    return {QuantLib::Real(QuantLib::Null<QuantLib::Real>()).value(),
            QuantLib::Null<QuantLib::Integer>()};
}

std::tuple<at::Tensor, int64_t> scalar_frexp(const at::Tensor& input) {
    int exponent = 0;
    const auto mantissa = quantlibtorch::frexp(quantlibtorch::TorchReal(input), &exponent);
    return {mantissa.tensor(), exponent};
}

at::Tensor scalar_ldexp(const at::Tensor& input, int64_t exponent) {
    TORCH_CHECK(exponent >= (std::numeric_limits<int>::min)()
                && exponent <= (std::numeric_limits<int>::max)(), "exponent is out of int range");
    return quantlibtorch::ldexp(quantlibtorch::TorchReal(input), static_cast<int>(exponent)).tensor();
}

int64_t scalar_integer(const at::Tensor& input) {
    return static_cast<int64_t>(quantlibtorch::TorchReal(input));
}

std::tuple<bool, bool> performance_config() {
    return {quantlibtorch::passive_constants_enabled(), quantlibtorch::sample_cache_enabled()};
}

std::tuple<at::Tensor, bool> scalar_arithmetic(const at::Tensor& input, double constant,
                                             bool tensor_constant) {
    using quantlibtorch::TorchReal;
    const TorchReal x(input);
    const TorchReal c = tensor_constant
        ? TorchReal(at::scalar_tensor(constant, input.options())) : TorchReal(constant);
    auto copy = x;
    const auto alias = copy;
    copy *= c;
    copy += x;
    copy /= c;
    copy -= x;
    return {at::stack({(x+c).tensor(), (c-x).tensor(), (x*c).tensor(),
                      (c/x).tensor(), quantlibtorch::pow(x,c).tensor(),
                      quantlibtorch::pow(c,x).tensor(),
                      quantlibtorch::exp(quantlibtorch::log(x)).tensor(),
                      copy.tensor(), alias.tensor(), (c*c-c).tensor()}), x.is_passive()};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_scalar_null() -> (float, int)");
    m.def("_scalar_frexp(Tensor input) -> (Tensor, int)");
    m.def("_scalar_ldexp(Tensor input, int exponent) -> Tensor");
    m.def("_scalar_integer(Tensor input) -> int");
    m.def("_performance_config() -> (bool, bool)");
    m.def("_scalar_arithmetic(Tensor input, float constant, bool tensor_constant) -> (Tensor, bool)");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_scalar_null", TORCH_FN(scalar_null));
    m.impl("_scalar_frexp", TORCH_FN(scalar_frexp));
    m.impl("_scalar_ldexp", TORCH_FN(scalar_ldexp));
    m.impl("_scalar_integer", TORCH_FN(scalar_integer));
    m.impl("_performance_config", TORCH_FN(performance_config));
    m.impl("_scalar_arithmetic", TORCH_FN(scalar_arithmetic));
}
