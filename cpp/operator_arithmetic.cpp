// Elementary arithmetic adapted from QuantLib v1.40, commit
// 06edddbdb85f006c43a449b5fc846a00c67477f3:
// ql/methods/finitedifferences/operators/{fdmblackscholesop,triplebandlinearop}.cpp
// (FdmBlackScholesOp::setTime, TripleBandLinearOp::{mult,axpyb,apply}).
// Copyright (C) 2008 Andreas Gaida
// Copyright (C) 2008, 2009 Ralph Schreyer
// Copyright (C) 2008, 2009 Klaus Spanderen
// Copyright (C) 2014 Johannes Göttker-Schnetmann
// Distributed under the QuantLib license; see LICENSES/QuantLib.txt.
// The complete license, conditions, and disclaimer are retained there.
#include <quantlibtorch/operator_arithmetic.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/ATen.h>
#include <ATen/record_function.h>

namespace quantlibtorch {
namespace {
at::Tensor readable(const at::Tensor& tensor) {
    auto values = tensor.is_neg() ? tensor.resolve_neg() : tensor;
    if (values.is_conj()) values = values.resolve_conj();
    return values.contiguous();
}
}

bool arithmetic_native_forward_enabled() {
    static const bool enabled = performance_flag("QLT_ARITHMETIC_NATIVE_FORWARD");
    return enabled;
}

at::Tensor native_diffusion_arithmetic(const at::Tensor& variance, const at::Tensor& dx,
    const at::Tensor& dxx, const at::Tensor& rate, const at::Tensor& yield) {
    RECORD_FUNCTION("qlt::diffusion_arithmetic_native_forward", std::vector<c10::IValue>());
    const auto v = readable(variance), x = readable(dx), xx = readable(dxx);
    const auto n = v.numel();
    const double r = rate.item<double>(), q = yield.item<double>();
    const auto* values = v.const_data_ptr<double>();
    const auto* first = x.const_data_ptr<double>();
    const auto* second = xx.const_data_ptr<double>();
    auto result = at::empty({3, n}, v.options());
    auto* out = result.mutable_data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        const double half_variance = 0.5*values[i];
        const double drift = (r-q)-half_variance;
        for (int64_t k = 0; k < 3; ++k) {
            const auto j = k*n+i;
            // Preserve upstream mult followed by axpyb, including the
            // separately rounded discount addition on the diagonal only.
            const double scaled_second = second[j]*half_variance;
            const double combined = scaled_second+drift*first[j];
            out[j] = k == 1 ? combined+(-r) : combined;
        }
    }
    return result;
}

at::Tensor native_apply_arithmetic(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, const at::Tensor& left, const at::Tensor& right) {
    RECORD_FUNCTION("qlt::apply_arithmetic_native_forward", std::vector<c10::IValue>());
    const auto lo = readable(lower), di = readable(diagonal), up = readable(upper), r = readable(rhs);
    const auto li = left.contiguous(), ri = right.contiguous();
    const auto n = r.numel();
    const auto* l = lo.const_data_ptr<double>(); const auto* d = di.const_data_ptr<double>();
    const auto* u = up.const_data_ptr<double>(); const auto* values = r.const_data_ptr<double>();
    const auto* before = li.const_data_ptr<int64_t>(); const auto* after = ri.const_data_ptr<int64_t>();
    auto result = at::empty({n}, r.options());
    auto* out = result.mutable_data_ptr<double>();
    for (int64_t i = 0; i < n; ++i)
        out[i] = (values[before[i]]*l[i]+values[i]*d[i])+values[after[i]]*u[i];
    return result;
}
}
