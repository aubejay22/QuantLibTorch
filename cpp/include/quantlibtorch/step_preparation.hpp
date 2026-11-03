#pragma once
#include <ATen/core/Tensor.h>
#include <cstddef>
#include <vector>

namespace quantlibtorch {
bool assembled_bands_enabled();
// Elementary full-AD preparation only: compact scaled bands and step RHS.
// The caller retains the existing guarded solve and every elimination state.
bool try_step_preparation(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& state, const std::size_t* left,
    const std::size_t* right, double dt, double theta, int kind,
    std::vector<at::Tensor>& output);
// Same elementary expressions, with one original 3-by-N band input and one
// band gradient. Ownership is supplied explicitly by the assembling operator.
bool try_assembled_step_preparation(const at::Tensor& bands, const at::Tensor& state,
    const std::size_t* left, const std::size_t* right, double dt, double theta,
    int kind, std::vector<at::Tensor>& output);
}
