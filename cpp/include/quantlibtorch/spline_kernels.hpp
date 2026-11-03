#pragma once

#include <quantlibtorch/tensor_kernels.hpp>
#include <vector>

namespace quantlibtorch {
inline bool spline_kernels_enabled() {
    static const bool enabled = performance_flag("QLT_SPLINE_KERNELS");
    return enabled && tensor_kernel_mode() >= 2;
}
inline bool tridiagonal_adjoint_enabled() {
    static const bool enabled = performance_flag("QLT_TRIDIAGONAL_ADJOINT");
    return enabled && native_adjoint_enabled() && tensor_kernel_mode() >= 2;
}

bool spline_slopes(const QuantLib::Array& x, const QuantLib::Array& y,
                   QuantLib::Real* dx, QuantLib::Real* slopes);
template <class I1, class I2>
bool spline_slopes(I1 first, I1 last, I2 values, QuantLib::Real* dx, QuantLib::Real* slopes) {
    if (!spline_kernels_enabled()) return false;
    const QuantLib::Array x(first, last), y(values, values + (last-first));
    return spline_slopes(x, y, dx, slopes);
}
bool spline_rhs(const std::vector<QuantLib::Real>& dx, const std::vector<QuantLib::Real>& slopes,
                QuantLib::Array& rhs);
bool spline_filter(const std::vector<QuantLib::Real>& dx, const std::vector<QuantLib::Real>& slopes,
                   QuantLib::Array& derivatives, std::vector<bool>& adjustments);
bool spline_coefficients(const std::vector<QuantLib::Real>& dx, const std::vector<QuantLib::Real>& slopes,
                         const QuantLib::Array& derivatives, QuantLib::Real* a,
                         QuantLib::Real* b, QuantLib::Real* c);
QuantLib::Array spline_increments(const QuantLib::Array& y, const std::vector<QuantLib::Real>& dx,
    const std::vector<QuantLib::Real>& a, const std::vector<QuantLib::Real>& b,
    const std::vector<QuantLib::Real>& c);
template <class I>
QuantLib::Array spline_increments(I values, QuantLib::Size n, const std::vector<QuantLib::Real>& dx,
    const std::vector<QuantLib::Real>& a, const std::vector<QuantLib::Real>& b,
    const std::vector<QuantLib::Real>& c) {
    if (!spline_kernels_enabled() || n <= 2) return {};
    return spline_increments(QuantLib::Array(values, values+n), dx, a, b, c);
}
}
