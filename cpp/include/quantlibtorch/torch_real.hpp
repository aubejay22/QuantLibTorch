#pragma once

#include <ATen/ATen.h>
#include <quantlibtorch/performance.hpp>
#include <cmath>
#include <limits>
#include <ostream>
#include <type_traits>

namespace quantlibtorch {

template <class T>
inline T checked_integral_cast(double value) {
    TORCH_CHECK(std::isfinite(value), "cannot convert nonfinite TorchReal to an integer");
    if constexpr (!std::is_same_v<T, bool>) {
        // Use an exclusive power-of-two upper bound: converting max(uint64)
        // to double rounds it to 2^64, so a <= max test would admit overflow.
        const double bound = std::ldexp(1.0, std::numeric_limits<T>::digits);
        const double truncated = std::trunc(value);
        const double lower = std::is_signed_v<T> ? -bound : 0.0;
        TORCH_CHECK(truncated >= lower && truncated < bound,
                    "TorchReal integer conversion is out of range");
    }
    return static_cast<T>(value);
}

// Constants originating as C++ doubles stay passive until mixed with tensors.
// Tensor inputs ALWAYS stay tensors, even without requires_grad or in no_grad.
// No implicit conversion back to double can sever an upstream coefficient graph.
class TorchReal {
  public:
    TorchReal() : TorchReal(0.0) {}
    TorchReal(double value) : constant_(value) {
        if (!passive_constants_enabled())
            value_ = at::scalar_tensor(value, at::TensorOptions().dtype(at::kDouble));
    }
    explicit TorchReal(at::Tensor value) : value_(std::move(value)) {
        TORCH_CHECK(value_.device().is_cpu() && value_.scalar_type() == at::kDouble
                    && value_.dim() == 0, "TorchReal requires a CPU float64 scalar");
    }
    bool is_passive() const { return !value_.defined() && !array_owner_.defined(); }
    bool requires_grad() const {
        return value_.defined() ? value_.requires_grad()
            : array_owner_.defined() && array_owner_.requires_grad();
    }
    at::Tensor tensor() const {
        if (value_.defined()) return value_;
        if (array_owner_.defined()) return array_owner_.select(0, array_index_);
        return at::scalar_tensor(constant_, at::TensorOptions().dtype(at::kDouble));
    }
    double value() const {
        if (value_.defined()) return value_.item<double>();
        if (!array_owner_.defined()) return constant_;
        // CPU scalar control-flow read, equivalent to select(...).item().
        // The graph-connected tensor view is materialized only for arithmetic.
        if (array_owner_.is_neg() || array_owner_.is_conj()) return tensor().item<double>();
        return array_owner_.const_data_ptr<double>()[array_index_*array_owner_.stride(0)];
    }

    // Provenance of an unbound array element, used only to reconstruct an
    // unchanged contiguous run. Scalar arithmetic creates a fresh TorchReal
    // and therefore drops this provenance; copies preserve it by value.
    static TorchReal array_element(at::Tensor scalar, const at::Tensor& owner, int64_t index) {
        TorchReal result(std::move(scalar));
        result.array_owner_ = owner;
        result.array_index_ = index;
        return result;
    }
    static TorchReal array_element(const at::Tensor& owner, int64_t index) {
        return TorchReal(ArrayView{}, owner, index);
    }
    const at::Tensor& array_owner() const { return array_owner_; }
    int64_t array_index() const { return array_index_; }

    // Upstream uses explicit integral conversions for mesh sizes / iteration
    // control. Never permit an implicit floating conversion or a differentiable
    // tensor to disappear into an integer calculation.
    template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    explicit operator T() const {
        TORCH_CHECK(!requires_grad(), "cannot convert differentiable TorchReal to an integer");
        return checked_integral_cast<T>(value());
    }

    // Rebind instead of mutating: Array copies share handles, as do saved
    // autograd tensors. In-place arithmetic would break value semantics.
    TorchReal& operator+=(const TorchReal& x) { return *this = *this + x; }
    TorchReal& operator-=(const TorchReal& x) { return *this = *this - x; }
    TorchReal& operator*=(const TorchReal& x) { return *this = *this * x; }
    TorchReal& operator/=(const TorchReal& x) { return *this = *this / x; }
    TorchReal operator-() const { return is_passive() ? TorchReal(-constant_) : TorchReal(-tensor()); }
    TorchReal operator+() const { return *this; }
    TorchReal& operator++() { return *this += 1.0; }
    TorchReal operator++(int) { auto previous = *this; ++*this; return previous; }
    TorchReal& operator--() { return *this -= 1.0; }
    TorchReal operator--(int) { auto previous = *this; --*this; return previous; }

#define QLT_BINARY(OP) \
    friend TorchReal operator OP(const TorchReal& a, const TorchReal& b) { \
        if (a.is_passive() && b.is_passive()) return TorchReal(a.constant_ OP b.constant_); \
        if (a.is_passive()) return TorchReal(a.constant_ OP b.tensor()); \
        if (b.is_passive()) return TorchReal(a.tensor() OP b.constant_); \
        return TorchReal(a.tensor() OP b.tensor()); \
    }
    QLT_BINARY(+)
    QLT_BINARY(-)
    QLT_BINARY(*)
    QLT_BINARY(/)
#undef QLT_BINARY
#define QLT_COMPARE(OP) \
    friend bool operator OP(const TorchReal& a, const TorchReal& b) { \
        return a.value() OP b.value(); \
    }
    QLT_COMPARE(==)
    QLT_COMPARE(!=)
    QLT_COMPARE(<)
    QLT_COMPARE(<=)
    QLT_COMPARE(>)
    QLT_COMPARE(>=)
#undef QLT_COMPARE
    friend std::ostream& operator<<(std::ostream& os, const TorchReal& x) {
        return os << x.value();
    }
  private:
    struct ArrayView {};
    TorchReal(ArrayView, const at::Tensor& owner, int64_t index)
    : array_owner_(owner), array_index_(index) {}
    at::Tensor value_;
    at::Tensor array_owner_;
    int64_t array_index_ = -1;
    double constant_ = 0.0;
};

inline TorchReal fabs(const TorchReal& x) {
    return x.is_passive() ? TorchReal(std::fabs(x.value())) : TorchReal(at::abs(x.tensor()));
}
inline TorchReal abs(const TorchReal& x) { return fabs(x); }
inline TorchReal pow(const TorchReal& x, const TorchReal& y) {
    if (x.is_passive() && y.is_passive()) return TorchReal(std::pow(x.value(), y.value()));
    if (x.is_passive()) return TorchReal(at::pow(x.value(), y.tensor()));
    if (y.is_passive()) return TorchReal(at::pow(x.tensor(), y.value()));
    return TorchReal(at::pow(x.tensor(), y.tensor()));
}
#define QLT_UNARY_MATH(NAME) \
    inline TorchReal NAME(const TorchReal& x) { \
        return x.is_passive() ? TorchReal(std::NAME(x.value())) : TorchReal(at::NAME(x.tensor())); \
    }
QLT_UNARY_MATH(sqrt)
QLT_UNARY_MATH(exp)
QLT_UNARY_MATH(log)
QLT_UNARY_MATH(expm1)
QLT_UNARY_MATH(log1p)
QLT_UNARY_MATH(sin)
QLT_UNARY_MATH(cos)
QLT_UNARY_MATH(tan)
QLT_UNARY_MATH(sinh)
QLT_UNARY_MATH(cosh)
QLT_UNARY_MATH(tanh)
QLT_UNARY_MATH(asinh)
QLT_UNARY_MATH(acos)
QLT_UNARY_MATH(atan)
QLT_UNARY_MATH(floor)
QLT_UNARY_MATH(ceil)
#undef QLT_UNARY_MATH
inline bool isfinite(const TorchReal& x) { return std::isfinite(x.value()); }
inline bool isnan(const TorchReal& x) { return std::isnan(x.value()); }
inline bool isinf(const TorchReal& x) { return std::isinf(x.value()); }
inline long lround(const TorchReal& x) {
    TORCH_CHECK(!x.requires_grad(), "cannot round differentiable TorchReal to an integer");
    return checked_integral_cast<long>(std::round(x.value()));
}
inline TorchReal frexp(const TorchReal& x, int* exponent) {
    if (x.is_passive()) return TorchReal(std::frexp(x.value(), exponent));
    auto result = at::frexp(x.tensor());
    *exponent = std::get<1>(result).item<int>();
    return TorchReal(std::get<0>(result));
}
inline TorchReal ldexp(const TorchReal& x, int exponent) {
    if (x.is_passive()) return TorchReal(std::ldexp(x.value(), exponent));
    // ATen ldexp forms 2**exponent first; that can overflow although the final
    // product is finite (e.g. frexp(DBL_MAX)'s mantissa, exponent 1024).
    // Finite nonzero factors retain the tensor graph and IEEE zero/inf signs.
    // Outside these bounds every nonzero finite binary64 value saturates, so
    // clamping limits work to three multiplications even for INT_MIN/INT_MAX.
    int remaining = std::max(-2099, std::min(2098, exponent));
    at::Tensor result = x.tensor();
    while (remaining > 1023) {
        result = result * std::ldexp(1.0, 1023);
        remaining -= 1023;
    }
    // Retain a single multiplication for representable subnormal factors;
    // splitting e=-1023 prematurely can introduce double rounding.
    while (remaining < -1074) {
        result = result * std::ldexp(1.0, -1022);
        remaining += 1022;
    }
    if (remaining != 0)
        result = result * std::ldexp(1.0, remaining);
    return TorchReal(result);
}

// Mixed constants in upstream min/max prevent std template deduction. ADL
// selects these value-preserving overloads; branch derivatives remain piecewise.
inline TorchReal min(const TorchReal& x, const TorchReal& y) { return y < x ? y : x; }
inline TorchReal max(const TorchReal& x, const TorchReal& y) { return x < y ? y : x; }
template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
inline TorchReal min(const TorchReal& x, T y) { return min(x, TorchReal(y)); }
template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
inline TorchReal min(T x, const TorchReal& y) { return min(TorchReal(x), y); }
template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
inline TorchReal max(const TorchReal& x, T y) { return max(x, TorchReal(y)); }
template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
inline TorchReal max(T x, const TorchReal& y) { return max(TorchReal(x), y); }
} // namespace quantlibtorch

namespace std {
// QuantLib's comparison.hpp needs epsilon() usable as a constexpr double.
// Limits describe the underlying representation; no std math overloads added.
template <> class numeric_limits<quantlibtorch::TorchReal> : public numeric_limits<double> {};
}

#define QL_REAL quantlibtorch::TorchReal
