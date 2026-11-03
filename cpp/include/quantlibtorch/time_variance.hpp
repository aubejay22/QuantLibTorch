#pragma once
#include <ATen/core/Tensor.h>

namespace quantlibtorch {
// The existing TensorLocalVol interpolation and square, with an optional
// elementary CPU boundary. Original inputs retain all derivative edges.
at::Tensor time_variance(const at::Tensor& low, const at::Tensor& high,
                         double weight, bool native = true);
}
