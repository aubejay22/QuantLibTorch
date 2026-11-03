// Forward arithmetic adapted from QuantLib v1.40, commit
// 06edddbdb85f006c43a449b5fc846a00c67477f3:
// ql/methods/finitedifferences/schemes/
// {douglasscheme,impliciteulerscheme,expliciteulerscheme}.cpp
// (DouglasScheme::step, ImplicitEulerScheme::step, ExplicitEulerScheme::step),
// operators/triplebandlinearop.cpp (apply, solve_splitting),
// tridiagonaloperator.cpp (solveFor), and ql/math/comparison.hpp (close).
// Copyright (C) 2008, 2009 Andreas Gaida
// Copyright (C) 2008, 2009 Ralph Schreyer
// Copyright (C) 2008, 2009, 2017 Klaus Spanderen
// Copyright (C) 2014 Johannes Göttker-Schnetmann
// Copyright (C) 2002, 2003, 2011 Ferdinando Ametrano
// Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl
// Copyright (C) 2003, 2004, 2005, 2007 StatPro Italia srl
// Distributed under the QuantLib license; see LICENSES/QuantLib.txt for
// the complete license, conditions, and disclaimer.
// The backward retains this project's existing first-order step adjoint.
#include <quantlibtorch/step_arithmetic.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/ATen.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace quantlibtorch {
namespace {
at::Tensor readable(const at::Tensor& t) {
    auto values = t.is_neg() ? t.resolve_neg() : t;
    if (values.is_conj()) values = values.resolve_conj();
    return values.contiguous();
}

bool close_zero(double value) {
    constexpr double tolerance = 42*std::numeric_limits<double>::epsilon();
    return value == 0. || std::abs(value) < tolerance*tolerance;
}

at::Tensor transpose_solve(const double* lower, const double* diagonal, const double* upper,
                          const at::Tensor& incoming, double beta) {
    const auto n = incoming.numel();
    auto result = at::empty_like(incoming), factors = at::empty_like(incoming);
    const auto* r = incoming.const_data_ptr<double>();
    auto* x = result.mutable_data_ptr<double>(); auto* t = factors.mutable_data_ptr<double>();
    double pivot = 1.-beta*diagonal[0];
    TORCH_CHECK(!close_zero(pivot), "diagonal's first element (", pivot, ") cannot be close to zero");
    x[0] = r[0]/pivot;
    for (int64_t i = 1; i < n; ++i) {
        const double transposed_lower = -beta*upper[i-1];
        t[i] = (-beta*lower[i])/pivot;
        pivot = (1.-beta*diagonal[i])-transposed_lower*t[i];
        TORCH_CHECK(!close_zero(pivot), "division by zero");
        x[i] = (r[i]-transposed_lower*x[i-1])/pivot;
    }
    for (int64_t i = n-2; i >= 0; --i) x[i] = x[i]-t[i+1]*x[i+1];
    return result;
}
}

bool step_native_arithmetic_enabled() {
    static const bool enabled = performance_flag("QLT_STEP_NATIVE_ARITHMETIC");
    return enabled;
}

bool adjoint_assembled_bands_enabled() {
    static const bool enabled = performance_flag("QLT_ADJOINT_ASSEMBLED_BANDS");
    return enabled && step_native_arithmetic_enabled();
}

bool step_arithmetic_eligible(at::TensorList inputs) {
    if (at::functorch::functorchTLSAccessor() && !at::functorch::getDynamicLayerStack().empty()) return false;
    for (const auto& t : inputs) {
        const auto keys = t.key_set();
        if (!t.is_cpu() || t.scalar_type() != at::kDouble || t.layout() != at::kStrided
            || keys.has(c10::DispatchKey::Python) || keys.has(c10::DispatchKey::Functionalize)
            || keys.has(c10::DispatchKey::FuncTorchBatched) || keys.has(c10::DispatchKey::FuncTorchGradWrapper)
            || t._fw_grad(0).defined()) return false;
    }
    return true;
}

at::Tensor native_step_forward(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& state, double dt, double theta, double time, int kind, bool assembled) {
    RECORD_FUNCTION("qlt::step_arithmetic_native_forward", std::vector<c10::IValue>());
    TORCH_CHECK(kind >= 0 && kind <= 2, "unknown step kind");
    TORCH_CHECK(time-dt > -1e-8, "a step towards negative time given");
    const auto lo = readable(lower), old = readable(state);
    const auto di = assembled ? at::Tensor() : readable(diagonal);
    const auto up = assembled ? at::Tensor() : readable(upper);
    const auto n = old.numel();
    const auto* l = lo.const_data_ptr<double>();
    const auto* d = assembled ? l+n : di.const_data_ptr<double>();
    const auto* u = assembled ? d+n : up.const_data_ptr<double>();
    const auto* r = old.const_data_ptr<double>();
    auto result = at::empty_like(old);
    auto* x = result.mutable_data_ptr<double>();
    const double beta = theta*dt;
    for (int64_t i = 0; i < n; ++i) {
        if (kind == 1) x[i] = r[i];
        else {
            const auto left = i == 0 ? 1 : i-1, right = i+1 == n ? n-2 : i+1;
            const double applied = (r[left]*l[i]+r[i]*d[i])+r[right]*u[i];
            // Douglas applies the same immutable one-dimensional map twice.
            // Reusing its value preserves both original additions/scalings.
            x[i] = kind == 0 ? (r[i]+dt*applied)-beta*applied : r[i]+beta*applied;
        }
    }
    if (kind == 2) return result;
#ifdef QL_EXTRA_SAFETY_CHECKS
    TORCH_CHECK(l[0] == 0. && u[n-1] == 0., "removing non zero entry!");
#endif
    auto factors = at::empty_like(old);
    auto* t = factors.mutable_data_ptr<double>();
    const double a = -theta*dt;
    double inverse = 1./(a*d[0]+1.);
    // Preserve the upstream inverse-pivot first check and direct-pivot later
    // checks, including their distinct behavior for zeros and infinities.
    TORCH_CHECK(inverse != 0., "division by zero");
    x[0] = x[0]*inverse;
    for (int64_t i = 1; i < n; ++i) {
        t[i] = (a*u[i-1])*inverse;
        const double pivot = 1.+a*(d[i]-t[i]*l[i]);
        TORCH_CHECK(pivot != 0., "division by zero");
        inverse = 1./pivot;
        x[i] = (x[i]-(a*l[i])*x[i-1])*inverse;
    }
    for (int64_t i = n-2; i >= 0; --i) x[i] = x[i]-t[i+1]*x[i+1];
    return result;
}

std::vector<at::Tensor> native_step_backward(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& state, const at::Tensor& result,
    const at::Tensor& incoming, double dt, double theta, int kind, bool assembled) {
    RECORD_FUNCTION("qlt::step_arithmetic_native_backward", std::vector<c10::IValue>());
    const auto lo = readable(lower);
    const auto di = assembled ? at::Tensor() : readable(diagonal);
    const auto up = assembled ? at::Tensor() : readable(upper);
    const auto old = readable(state), next = readable(result), weight = readable(incoming);
    const auto n = old.numel();
    const double beta = kind == 2 ? 0. : theta*dt;
    const double alpha = kind == 0 ? dt-beta : (kind == 2 ? theta*dt : 0.);
    const auto* l = lo.const_data_ptr<double>();
    const auto* d = assembled ? l+n : di.const_data_ptr<double>();
    const auto* u = assembled ? d+n : up.const_data_ptr<double>();
    const auto lambda = beta == 0. ? weight : transpose_solve(l, d, u, weight, beta);
    const auto* w = lambda.const_data_ptr<double>();
    const auto* before = old.const_data_ptr<double>(); const auto* after = next.const_data_ptr<double>();
    auto gb = assembled ? at::empty({3, n}, old.options()) : at::Tensor();
    auto gl = assembled ? at::Tensor() : at::empty_like(old);
    auto gd = assembled ? at::Tensor() : at::empty_like(old);
    auto gu = assembled ? at::Tensor() : at::empty_like(old);
    auto gr = at::empty_like(old);
    auto* dl = assembled ? gb.mutable_data_ptr<double>() : gl.mutable_data_ptr<double>();
    auto* dd = assembled ? dl+n : gd.mutable_data_ptr<double>();
    auto* du = assembled ? dd+n : gu.mutable_data_ptr<double>();
    auto* dr = gr.mutable_data_ptr<double>();
    std::fill(dr, dr+n, 0.);
    for (int64_t i = 0; i < n; ++i) {
        TORCH_CHECK(std::isfinite(w[i]), "non-finite step adjoint");
        const auto left = i == 0 ? 1 : i-1, right = i+1 == n ? n-2 : i+1;
        dl[i] = w[i]*(alpha*before[left]+(i == 0 ? 0. : beta*after[i-1]));
        dd[i] = w[i]*(alpha*before[i]+beta*after[i]);
        du[i] = w[i]*(alpha*before[right]+(i+1 == n ? 0. : beta*after[i+1]));
        dr[i] += w[i]+alpha*w[i]*d[i];
        dr[left] += alpha*w[i]*l[i];
        dr[right] += alpha*w[i]*u[i];
    }
    // Every write above targets fresh result/gradient/scratch buffers.
    if (assembled) return {gb, gr};
    return {gl, gd, gu, gr};
}
}
