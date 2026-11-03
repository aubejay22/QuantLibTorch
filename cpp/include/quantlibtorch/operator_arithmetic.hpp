#pragma once
#include <ATen/core/Tensor.h>

namespace quantlibtorch {
bool arithmetic_native_forward_enabled();

// Internal CPU float64 forwards, called only inside the checked elementary
// autograd boundaries. All original inputs remain saved by the caller.
at::Tensor native_diffusion_arithmetic(const at::Tensor& variance, const at::Tensor& dx,
    const at::Tensor& dxx, const at::Tensor& rate, const at::Tensor& yield);
at::Tensor native_apply_arithmetic(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, const at::Tensor& left, const at::Tensor& right);
}
