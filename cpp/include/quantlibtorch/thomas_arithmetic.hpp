#pragma once
#include <ATen/core/Tensor.h>

namespace quantlibtorch {
// Checked tensor interface to the same recurrence, including multiple RHS.
// Full AD differentiates elimination arithmetic and retains higher derivatives.
at::Tensor thomas_solve(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs);
// CPU recurrence fusion after the existing auto-domain check. The
// caller retains the residual check and original-solver fallback.
bool try_thomas_arithmetic(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, at::Tensor& output);
}
