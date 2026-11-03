// Elementary preparation adapted from QuantLib v1.40, commit
// 06edddbdb85f006c43a449b5fc846a00c67477f3:
// ql/methods/finitedifferences/schemes/{douglasscheme,impliciteulerscheme}.cpp
// and operators/triplebandlinearop.cpp (apply and solve_splitting scaling).
// Copyright (C) 2008, 2009 Andreas Gaida
// Copyright (C) 2008, 2009 Ralph Schreyer
// Copyright (C) 2008, 2009, 2017 Klaus Spanderen
// Copyright (C) 2014 Johannes Goettker-Schnetmann
// Distributed under the QuantLib license; see LICENSES/QuantLib.txt for the
// complete license, conditions, and disclaimer. This boundary differentiates
// only elementary arithmetic. The existing full-AD solver remains outside it.
#include <quantlibtorch/step_preparation.hpp>
#include <quantlibtorch/step_arithmetic.hpp>
#include <quantlibtorch/operator_arithmetic.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace quantlibtorch {
namespace {
using Tensors = std::vector<at::Tensor>;

bool enabled() {
    static const bool active = performance_flag("QLT_STEP_PREPARATION");
    return active && arithmetic_native_forward_enabled();
}

at::Tensor readable(const at::Tensor& t) {
    auto result = t.is_neg() ? t.resolve_neg() : t;
    if (result.is_conj()) result = result.resolve_conj();
    return result.contiguous();
}

class StepPreparation : public torch::autograd::Function<StepPreparation> {
  public:
    static Tensors forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor state,
        at::Tensor left, at::Tensor right, double dt, double theta, int kind) {
        ctx->set_materialize_grads(false);
        ctx->save_for_backward({lower, diagonal, upper, state, left, right});
        ctx->saved_data["dt"] = dt; ctx->saved_data["theta"] = theta; ctx->saved_data["kind"] = kind;
        return forward_values(lower, diagonal, upper, state, left, right, dt, theta, kind);
    }

    static Tensors forward_values(const at::Tensor& lower, const at::Tensor& diagonal,
        const at::Tensor& upper, const at::Tensor& state, const at::Tensor& left,
        const at::Tensor& right, double dt, double theta, int kind, bool assembled = false) {
        RECORD_FUNCTION("qlt::step_preparation_forward", std::vector<c10::IValue>());
        // For an assembled input, lower is the original 3-by-N matrix. One
        // readable logical copy avoids creating three temporary tensor views.
        const auto lo = readable(lower), old = readable(state);
        const auto di = assembled ? at::Tensor() : readable(diagonal);
        const auto up = assembled ? at::Tensor() : readable(upper);
        const auto n = state.numel();
        const auto* l = lo.const_data_ptr<double>();
        const auto* d = assembled ? l+n : di.const_data_ptr<double>();
        const auto* u = assembled ? d+n : up.const_data_ptr<double>();
        const auto* s = old.const_data_ptr<double>();
        const auto* li = kind == 0 ? left.const_data_ptr<int64_t>() : nullptr;
        const auto* ri = kind == 0 ? right.const_data_ptr<int64_t>() : nullptr;
        auto scaled_lo = at::empty({n-1}, state.options()), scaled_di = at::empty({n}, state.options());
        auto scaled_up = at::empty({n-1}, state.options()), rhs = at::empty({n}, state.options());
        auto* a_lo = scaled_lo.mutable_data_ptr<double>(); auto* a_di = scaled_di.mutable_data_ptr<double>();
        auto* a_up = scaled_up.mutable_data_ptr<double>(); auto* r = rhs.mutable_data_ptr<double>();
        const double a = (-theta)*dt, beta = theta*dt;
        for (int64_t i = 0; i < n; ++i) {
            a_di[i] = a*d[i]+1.;
            if (i > 0) a_lo[i-1] = a*l[i];
            if (i+1 < n) a_up[i] = a*u[i];
            if (kind == 1) r[i] = s[i];
            else {
                TORCH_CHECK(li[i] >= 0 && li[i] < n && ri[i] >= 0 && ri[i] < n, "invalid step neighbour index");
                const double applied = (s[li[i]]*l[i]+s[i]*d[i])+s[ri[i]]*u[i];
                // Preserve separately rounded products, sum, then subtraction.
                r[i] = (s[i]+dt*applied)-beta*applied;
            }
        }
        return {scaled_lo, scaled_di, scaled_up, rhs};
    }

    static Tensors backward(torch::autograd::AutogradContext* ctx, Tensors gradients) {
        const auto saved = ctx->get_saved_variables();
        auto result = backward_values(saved, gradients, ctx->saved_data["dt"].toDouble(),
            ctx->saved_data["theta"].toDouble(), ctx->saved_data["kind"].toInt(), false);
        result.resize(9);
        return result;
    }

    static Tensors backward_values(const Tensors& saved, const Tensors& gradients,
        double dt, double theta, int64_t kind, bool assembled) {
        RECORD_FUNCTION("qlt::step_preparation_backward", std::vector<c10::IValue>());
        auto lo = saved[0]; // original matrix for assembled input, row otherwise
        auto di = assembled ? at::Tensor() : saved[1];
        auto up = assembled ? at::Tensor() : saved[2];
        const auto& state = saved[assembled ? 1 : 3];
        const auto& left = saved[assembled ? 2 : 4];
        const auto& right = saved[assembled ? 3 : 5];
        const double a = (-theta)*dt, beta = theta*dt;
        const auto n = state.numel();
        bool functional = at::GradMode::is_enabled() || (assembled
            ? !step_arithmetic_eligible({lo, state}) : !step_arithmetic_eligible({lo, di, up, state}));
        for (const auto& g : gradients) if (g.defined() && !step_arithmetic_eligible({g})) functional = true;
        if (functional) {
            RECORD_FUNCTION("qlt::step_preparation_backward_graph", std::vector<c10::IValue>());
            if (assembled) {
                // These views are created while recording the higher/mixed
                // derivative, so their connection to the saved matrix remains.
                di = lo[1]; up = lo[2]; lo = lo[0];
            }
            const auto zero = at::zeros_like(state), edge = zero.slice(0, 0, 1);
            auto gl = gradients[0].defined() ? at::cat({edge, gradients[0]*a}) : zero;
            auto gd = gradients[1].defined() ? gradients[1]*a : zero;
            auto gu = gradients[2].defined() ? at::cat({gradients[2]*a, edge}) : zero;
            auto gr = gradients[3].defined() ? gradients[3] : zero;
            if (kind == 0 && gradients[3].defined()) {
                const auto& g = gradients[3];
                // Retain both product contributions instead of combining
                // dt and beta before multiplication. Originals preserve all
                // coefficient/state cross derivatives and incoming duals.
                const auto ga = (-g)*beta + g*dt;
                gl = gl + ga*state.index_select(0, left);
                gd = gd + ga*state;
                gu = gu + ga*state.index_select(0, right);
                gr = gr + (di*ga).index_add(0, left, lo*ga).index_add(0, right, up*ga);
            }
            if (assembled) return {at::stack({gl, gd, gu}), gr};
            return {gl, gd, gu, gr};
        }
        const auto l_values = readable(lo), s_values = readable(state);
        const auto d_values = assembled ? at::Tensor() : readable(di);
        const auto u_values = assembled ? at::Tensor() : readable(up);
        const auto* l = l_values.const_data_ptr<double>();
        const auto* d = assembled ? l+n : d_values.const_data_ptr<double>();
        const auto* u = assembled ? d+n : u_values.const_data_ptr<double>();
        const auto* s = s_values.const_data_ptr<double>();
        Tensors weights;
        for (const auto& g : gradients) weights.push_back(g.defined() ? readable(g) : at::Tensor());
        const auto pointer = [&](int i) { return weights[i].defined() ? weights[i].const_data_ptr<double>() : nullptr; };
        const auto* wl = pointer(0); const auto* wd = pointer(1); const auto* wu = pointer(2); const auto* wr = pointer(3);
        auto gb = assembled ? at::empty({3, n}, state.options()) : at::Tensor();
        auto gl = assembled ? at::Tensor() : at::empty({n}, state.options());
        auto gd = assembled ? at::Tensor() : at::empty({n}, state.options());
        auto gu = assembled ? at::Tensor() : at::empty({n}, state.options());
        auto gr = at::empty({n}, state.options());
        auto* dl = assembled ? gb.mutable_data_ptr<double>() : gl.mutable_data_ptr<double>();
        auto* dd = assembled ? dl+n : gd.mutable_data_ptr<double>();
        auto* du = assembled ? dd+n : gu.mutable_data_ptr<double>();
        auto* dr = gr.mutable_data_ptr<double>();
        // Only fresh gradient buffers are written. Initialize before scatter
        // accumulation so repeated and reflected neighbours remain correct.
        for (int64_t i = 0; i < n; ++i) {
            dl[i] = wl && i > 0 ? wl[i-1]*a : 0.;
            dd[i] = wd ? wd[i]*a : 0.;
            du[i] = wu && i+1 < n ? wu[i]*a : 0.;
            dr[i] = wr ? wr[i] : 0.;
        }
        if (kind == 0 && wr) {
            const auto* li = left.const_data_ptr<int64_t>(); const auto* ri = right.const_data_ptr<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                const double ga = (-wr[i])*beta + wr[i]*dt;
                dl[i] += ga*s[li[i]]; dd[i] += ga*s[i]; du[i] += ga*s[ri[i]];
                dr[li[i]] += ga*l[i]; dr[i] += ga*d[i]; dr[ri[i]] += ga*u[i];
            }
        }
        if (assembled) return {gb, gr};
        return {gl, gd, gu, gr};
    }
};

class AssembledStepPreparation : public torch::autograd::Function<AssembledStepPreparation> {
  public:
    static Tensors forward(torch::autograd::AutogradContext* ctx, at::Tensor bands,
        at::Tensor state, at::Tensor left, at::Tensor right, double dt, double theta, int kind) {
        RECORD_FUNCTION("qlt::assembled_preparation_forward", std::vector<c10::IValue>());
        ctx->set_materialize_grads(false);
        // Save the original matrix, not row views created under forward's
        // NoGradGuard. Backward reconstructs graph-connected views as needed.
        ctx->save_for_backward({bands, state, left, right});
        ctx->saved_data["dt"] = dt; ctx->saved_data["theta"] = theta; ctx->saved_data["kind"] = kind;
        return StepPreparation::forward_values(bands, at::Tensor(), at::Tensor(), state, left, right, dt, theta, kind, true);
    }

    static Tensors backward(torch::autograd::AutogradContext* ctx, Tensors gradients) {
        RECORD_FUNCTION("qlt::assembled_preparation_backward", std::vector<c10::IValue>());
        const auto saved = ctx->get_saved_variables();
        auto result = StepPreparation::backward_values(
            saved, gradients,
            ctx->saved_data["dt"].toDouble(), ctx->saved_data["theta"].toDouble(),
            ctx->saved_data["kind"].toInt(), true);
        result.resize(7);
        return result;
    }
};
}

bool assembled_bands_enabled() {
    static const bool active = performance_flag("QLT_ASSEMBLED_BANDS");
    return active && enabled();
}

bool try_assembled_step_preparation(const at::Tensor& bands, const at::Tensor& state,
    const std::size_t* left, const std::size_t* right, double dt, double theta,
    int kind, std::vector<at::Tensor>& output) {
    if (!assembled_bands_enabled() || (kind != 0 && kind != 1)
        || !step_arithmetic_eligible({bands, state})) return false;
    const auto n = state.numel();
    TORCH_CHECK(bands.dim() == 2 && bands.size(0) == 3 && bands.size(1) == n,
                "invalid assembled step bands");
    const auto options = at::TensorOptions().dtype(at::kLong);
    const auto li = kind == 0 ? at::tensor(std::vector<int64_t>(left, left+n), options) : at::empty({0}, options);
    const auto ri = kind == 0 ? at::tensor(std::vector<int64_t>(right, right+n), options) : li;
    output = AssembledStepPreparation::apply(bands, state, li, ri, dt, theta, kind);
    return true;
}

bool try_step_preparation(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up,
    const at::Tensor& state, const std::size_t* left, const std::size_t* right,
    double dt, double theta, int kind, std::vector<at::Tensor>& output) {
    if (!enabled() || (kind != 0 && kind != 1) || !step_arithmetic_eligible({lo, di, up, state})) return false;
    const auto n = state.numel();
    const auto options = at::TensorOptions().dtype(at::kLong);
    // C++ autograd input metadata requires defined tensors, even for the
    // implicit step's unused indices. One shared empty tensor is sufficient.
    const auto li = kind == 0 ? at::tensor(std::vector<int64_t>(left, left+n), options) : at::empty({0}, options);
    const auto ri = kind == 0 ? at::tensor(std::vector<int64_t>(right, right+n), options) : li;
    output = StepPreparation::apply(lo, di, up, state, li, ri, dt, theta, kind);
    return true;
}

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> preparation_probe(const at::Tensor& bands, const at::Tensor& state,
    double dt, double theta, int64_t kind, bool native, bool permuted, bool assembled) {
    TORCH_CHECK(bands.dim() == 2 && bands.size(0) == 3 && bands.size(1) >= 2 && state.dim() == 1
                && bands.size(1) == state.numel() && bands.device() == state.device()
                && bands.scalar_type() == state.scalar_type() && bands.layout() == at::kStrided
                && state.layout() == at::kStrided && std::isfinite(dt) && std::isfinite(theta)
                && (kind == 0 || kind == 1), "invalid step-preparation probe inputs");
    const auto n = state.numel();
    std::vector<std::size_t> left(n), right(n);
    for (int64_t i = 0; i < n; ++i) {
        left[i] = permuted ? (3*i+1)%n : (i == 0 ? 1 : i-1);
        right[i] = permuted ? (5*i+2)%n : (i+1 == n ? n-2 : i+1);
    }
    Tensors result;
    if (native && assembled && try_assembled_step_preparation(bands, state, left.data(), right.data(),
                                                             dt, theta, static_cast<int>(kind), result))
        return {result[0], result[1], result[2], result[3]};
    const auto lo = bands[0], di = bands[1], up = bands[2];
    if (native && try_step_preparation(lo, di, up, state, left.data(), right.data(), dt, theta, static_cast<int>(kind), result))
        return {result[0], result[1], result[2], result[3]};
    auto rhs = state;
    if (kind == 0) {
        const auto options = at::TensorOptions().dtype(at::kLong).device(state.device());
        // Tensor factories that copy C++ arrays can read raw storage under
        // transforms. Build the reference indices with functional operations.
        const auto rows = at::arange(n, options);
        const auto li = permuted ? at::remainder(3*rows+1, n)
            : at::cat({rows.slice(0, 1, 2), rows.slice(0, 0, n-1)});
        const auto ri = permuted ? at::remainder(5*rows+2, n)
            : at::cat({rows.slice(0, 1, n), rows.slice(0, n-2, n-1)});
        const auto applied = (state.index_select(0, li)*lo+state*di)+state.index_select(0, ri)*up;
        rhs = (state+dt*applied)-(theta*dt)*applied;
    }
    const double a = (-theta)*dt;
    return {a*lo.slice(0, 1, n), a*di+1., a*up.slice(0, 0, n-1), rhs};
}
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_step_preparation(Tensor bands, Tensor state, float dt, float theta, int kind, bool native=True, bool permuted=False, bool assembled=False) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_step_preparation", TORCH_FN(quantlibtorch::preparation_probe));
}
