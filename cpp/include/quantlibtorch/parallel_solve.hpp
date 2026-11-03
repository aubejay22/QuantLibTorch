#pragma once
#include <ql/math/array.hpp>
#include <ATen/core/grad_mode.h>

namespace quantlibtorch {
// Experimental parallel cyclic reduction, separate from upstream Thomas.
at::Tensor tridiagonal_solve_pcr(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, bool check_pivots);
QuantLib::Array pcr_split(const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal, const QuantLib::Real* upper);
// Conservative CPU dispatch. A rejected candidate leaves output untouched so
// the caller can execute the original QuantLib algorithm with all inputs.
bool try_auto_solve_tensor(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, at::Tensor& output);
bool try_pcr_split(const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal, const QuantLib::Real* upper,
    QuantLib::Array& output);
bool try_pcr_tridiagonal(const QuantLib::Array& lower, const QuantLib::Array& diagonal,
    const QuantLib::Array& upper, const QuantLib::Array& rhs, QuantLib::Array& output);
}
