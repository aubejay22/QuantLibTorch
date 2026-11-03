#pragma once
#include <ATen/core/Tensor.h>
#include <vector>

namespace quantlibtorch {
bool step_native_arithmetic_enabled();
bool adjoint_assembled_bands_enabled();
bool step_arithmetic_eligible(at::TensorList inputs);
// Only the existing one-dimensional, fixed-setting, empty-boundary adjoint
// path calls these. Original coefficient/state tensors remain saved there.
at::Tensor native_step_forward(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& state, double dt, double theta, double time, int kind,
    bool assembled = false);
std::vector<at::Tensor> native_step_backward(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& state, const at::Tensor& result,
    const at::Tensor& incoming, double dt, double theta, int kind, bool assembled = false);
// When assembled is true, lower is the original 3-by-N matrix and the other
// two band arguments are unused. Backward returns one matrix and state VJP.
}
