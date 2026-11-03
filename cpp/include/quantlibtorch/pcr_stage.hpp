#pragma once
#include <ATen/core/Tensor.h>
#include <vector>

namespace quantlibtorch {
// CPU implementation of ONE PCR elimination stage. This is the
// derivative of the elimination arithmetic, not an implicit solve adjoint.
// Other devices and forward duals retain the tensor recurrence.
bool try_native_pcr_stage(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, int64_t distance,
    std::vector<at::Tensor>& output);
}
