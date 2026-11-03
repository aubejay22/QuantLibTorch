#pragma once

#include <ql/math/array.hpp>

namespace quantlibtorch {
// Scoped to an engine call: 0=scalar, 1=application, 2=vectorized Thomas,
// 3=explicit PCR splitting, 4=automatic guarded PCR (otherwise Thomas).
int tensor_kernel_mode();
class TensorKernelMode {
  public:
    explicit TensorKernelMode(int mode);
    ~TensorKernelMode();
    TensorKernelMode(const TensorKernelMode&) = delete;
    TensorKernelMode& operator=(const TensorKernelMode&) = delete;
  private:
    int previous_;
};

at::Tensor pack_array(const QuantLib::Real* values, QuantLib::Size n);
void unpack_array(const at::Tensor& values, QuantLib::Real* output);
QuantLib::Array vector_apply(const QuantLib::Array& rhs,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper, const QuantLib::Size* left,
    const QuantLib::Size* right);
void vector_mult(QuantLib::Size n, const QuantLib::Array& scale,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper, QuantLib::Real* out_lower,
    QuantLib::Real* out_diagonal, QuantLib::Real* out_upper);
void vector_axpyb(QuantLib::Size n, const QuantLib::Array& a,
    const QuantLib::Real* xl, const QuantLib::Real* xd, const QuantLib::Real* xu,
    const QuantLib::Real* yl, const QuantLib::Real* yd, const QuantLib::Real* yu,
    const QuantLib::Array& b, QuantLib::Real* lower,
    QuantLib::Real* diagonal, QuantLib::Real* upper);
} // namespace quantlibtorch
