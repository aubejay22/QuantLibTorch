#pragma once
#include <ATen/core/Tensor.h>

namespace quantlibtorch {
// Predicate-only CPU optimization. The actual solve always receives the
// original graph-connected coefficients; these helpers never construct one.
bool native_solve_checks_enabled();
bool try_native_auto_domain(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, bool& decision);
bool try_native_auto_residual(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, const at::Tensor& solution, bool& decision);
}
