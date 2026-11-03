#include <quantlibtorch/step_adjoint.hpp>
#include <quantlibtorch/step_arithmetic.hpp>
#include <quantlibtorch/step_preparation.hpp>
#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <quantlibtorch/parallel_solve.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/schemes/douglasscheme.hpp>
#include <ql/methods/finitedifferences/schemes/impliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/schemes/expliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/tridiagonaloperator.hpp>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <typeinfo>

namespace quantlibtorch {
namespace {
bool enabled() {
    static const bool active = performance_flag("QLT_STEP_ADJOINT");
    return active && native_adjoint_enabled();
}

bool full_step_enabled() {
    static const bool active = performance_flag("QLT_FULL_TENSOR_STEP");
    return active && operator_arithmetic_enabled();
}

// Tensor composition of the pinned QuantLib v1.40 Douglas/implicit/explicit
// steps, commit 06edddbdb85f006c43a449b5fc846a00c67477f3:
// ql/methods/finitedifferences/schemes/{douglasscheme,impliciteulerscheme,
// expliciteulerscheme}.cpp. Copyright (C) 2008, 2009 Andreas Gaida;
// 2008, 2009 Ralph Schreyer; 2008, 2009, 2017 Klaus Spanderen;
// 2014 Johannes Goettker-Schnetmann. See LICENSES/QuantLib.txt.
// The enclosing hook has already performed setTime. Empty boundary callbacks
// and an exact 1D Black-Scholes operator make apply and apply_direction equal.
// All operations stay on the full graph, including the selected solve's
// elimination states; this does not substitute an implicit first-order VJP.
bool try_full_tensor_step(const QuantLib::TripleBandLinearOp& op,
    const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up, const at::Tensor& state,
    const QuantLib::Size* left, const QuantLib::Size* right,
    double dt, double theta, int kind, at::Tensor& output) {
    if (!full_step_enabled() || tensor_kernel_mode() != 4 || !at::GradMode::is_enabled()
        || !step_arithmetic_eligible({lo, di, up, state})) return false;
    TORCH_CHECK(kind >= 0 && kind <= 2, "unknown step kind");
    if (kind != 2 && state.numel() < 8) return false;
    RECORD_FUNCTION("qlt::full_tensor_step", std::vector<c10::IValue>());
    std::vector<at::Tensor> prepared;
    if (try_step_preparation(lo, di, up, state, left, right, dt, theta, kind, prepared))
        return try_auto_solve_tensor(prepared[0], prepared[1], prepared[2], prepared[3], output);
    auto rhs = state;
    if (kind != 1) {
        at::Tensor applied;
        if (!try_full_apply_tensor(op, lo, di, up, state, left, right, applied)) return false;
        if (kind == 2) { output = state+(theta*dt)*applied; return true; }
        // Keep both separately rounded products and the original add/subtract
        // ordering. Only the identical immutable application is reused.
        rhs = (state+dt*applied)-(theta*dt)*applied;
    }
    const auto n = state.numel();
    const double a = (-theta)*dt;
    return try_auto_solve_tensor(a*lo.slice(0, 1, n), a*di+1., a*up.slice(0, 0, n-1), rhs, output);
}

at::Tensor readable(const at::Tensor& t) {
    auto values = t.is_neg() ? t.resolve_neg() : t;
    if (values.is_conj()) values = values.resolve_conj();
    return values.contiguous();
}

QuantLib::Array passive(const at::Tensor& t) {
    const auto values = readable(t);
    const auto* data = values.const_data_ptr<double>();
    QuantLib::Array result(t.numel());
    for (int64_t i = 0; i < t.numel(); ++i) result[i] = TorchReal(data[i]);
    return result;
}

class SnapshotBand : public QuantLib::TripleBandLinearOp {
  public:
    SnapshotBand(const QuantLib::TripleBandLinearOp& op, bool native) : TripleBandLinearOp(op) {
        if (native) for (QuantLib::Size i = 0; i < mesher_->layout()->size(); ++i) {
            lower_[i] = TorchReal(lower_[i].value());
            diag_[i] = TorchReal(diag_[i].value());
            upper_[i] = TorchReal(upper_[i].value());
        }
    }
};

// The actual financial operator has ALREADY executed setTime outside the VJP.
// This immutable snapshot only supplies its assembled matrix to the original
// QuantLib scheme. No rates, volatility, schedules or events are recalculated.
class SnapshotComposite final : public QuantLib::FdmLinearOpComposite {
  public:
    SnapshotComposite(const QuantLib::TripleBandLinearOp& op, bool native) : band_(op, native) {}
    QuantLib::Size size() const override { return 1; }
    void setTime(QuantLib::Time, QuantLib::Time) override {}
    QuantLib::Array apply(const QuantLib::Array& r) const override { return band_.apply(r); }
    QuantLib::Array apply_mixed(const QuantLib::Array& r) const override { return QuantLib::Array(r.size(), 0.); }
    QuantLib::Array apply_direction(QuantLib::Size direction, const QuantLib::Array& r) const override {
        return direction == 0 ? band_.apply(r) : QuantLib::Array(r.size(), 0.);
    }
    QuantLib::Array solve_splitting(QuantLib::Size direction, const QuantLib::Array& r, QuantLib::Real s) const override {
        return direction == 0 ? band_.solve_splitting(r, s) : r;
    }
    QuantLib::Array preconditioner(const QuantLib::Array& r, QuantLib::Real s) const override {
        return band_.solve_splitting(r, s);
    }
  private:
    SnapshotBand band_;
};

template <class Scheme> class FixedThetaScheme : public Scheme {
  public:
    using Scheme::Scheme;
    using Scheme::step; // expose the original protected fixed-theta overload
};

void upstream_step(const QuantLib::ext::shared_ptr<QuantLib::FdmLinearOpComposite>& map,
    QuantLib::Array& state, double dt, double theta, double time, int kind) {
    if (kind == 0) {
        QuantLib::DouglasScheme scheme(theta, map);
        scheme.setStep(dt);
        scheme.step(state, time);
    } else if (kind == 1) {
        FixedThetaScheme<QuantLib::ImplicitEulerScheme> scheme(map);
        scheme.setStep(dt);
        scheme.step(state, time, theta);
    } else {
        TORCH_CHECK(kind == 2, "unknown step kind");
        FixedThetaScheme<QuantLib::ExplicitEulerScheme> scheme(map);
        scheme.setStep(dt);
        scheme.step(state, time, theta);
    }
}

class StepAdjoint : public torch::autograd::Function<StepAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor state,
        const QuantLib::TripleBandLinearOp* op, double dt, double theta, double time, int kind, bool allow_native) {
        RECORD_FUNCTION("qlt::step_adjoint_forward", std::vector<c10::IValue>());
        AdjointMode adjoint_guard(false);
        TensorKernelMode kernel_guard(0);
        const bool native = allow_native && step_native_arithmetic_enabled()
            && step_arithmetic_eligible({lower, diagonal, upper, state});
        at::Tensor result;
        if (native) result = native_step_forward(lower, diagonal, upper, state, dt, theta, time, kind);
        else {
            auto values = passive(state);
            const auto map = QuantLib::ext::make_shared<SnapshotComposite>(*op, true);
            upstream_step(map, values, dt, theta, time, kind);
            result = pack_array(values.begin(), values.size());
        }
        TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite step solution");
        ctx->save_for_backward({lower, diagonal, upper, state, result});
        ctx->saved_data["dt"] = dt;
        ctx->saved_data["theta"] = theta;
        ctx->saved_data["kind"] = kind;
        ctx->saved_data["native_arithmetic"] = native;
        return result;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list gradients) {
        auto result = backward_values(ctx->get_saved_variables(), gradients[0],
            ctx->saved_data["dt"].toDouble(), ctx->saved_data["theta"].toDouble(),
            ctx->saved_data["kind"].toInt(), ctx->saved_data["native_arithmetic"].toBool(), false);
        result.resize(10);
        return result;
    }

    static torch::autograd::variable_list backward_values(
        const torch::autograd::variable_list& saved, const at::Tensor& gradient,
        double dt, double theta, int64_t kind, bool native, bool assembled) {
        RECORD_FUNCTION("qlt::step_adjoint_backward", std::vector<c10::IValue>());
        TORCH_CHECK(!at::GradMode::is_enabled(),
                    "QuantLib step adjoint supports first derivatives only; use mode='full' for higher derivatives");
        AdjointMode adjoint_guard(false);
        TensorKernelMode kernel_guard(0);
        const auto& state = saved[assembled ? 1 : 3];
        const auto& output = saved[assembled ? 2 : 4];
        if (native && step_arithmetic_eligible({gradient}))
            return native_step_backward(saved[0], assembled ? at::Tensor() : saved[1],
                assembled ? at::Tensor() : saved[2], state, output, gradient, dt, theta,
                static_cast<int>(kind), assembled);
        // Preserve the preceding generic backward for unsupported incoming
        // weights. First-order-only validation remains above both paths.
        const auto lo = readable(assembled ? saved[0][0] : saved[0]);
        const auto di = readable(assembled ? saved[0][1] : saved[1]);
        const auto up = readable(assembled ? saved[0][2] : saved[2]);
        const auto u = readable(state), x = readable(output);
        const auto n = u.numel();
        const double beta = kind == 2 ? 0. : theta*dt;
        const double alpha = kind == 0 ? dt-beta : (kind == 2 ? theta*dt : 0.);
        const auto* l = lo.const_data_ptr<double>();
        const auto* d = di.const_data_ptr<double>();
        const auto* h = up.const_data_ptr<double>();
        auto lambda = passive(gradient);
        if (beta != 0.) {
            QuantLib::Array tl(n-1), td(n), tu(n-1);
            for (int64_t i = 0; i < n; ++i) td[i] = TorchReal(1.-beta*d[i]);
            for (int64_t i = 0; i < n-1; ++i) {
                tl[i] = TorchReal(-beta*h[i]);
                tu[i] = TorchReal(-beta*l[i+1]);
            }
            lambda = QuantLib::TridiagonalOperator(tl, td, tu).solveFor(lambda);
        }
        auto gl = at::empty({n}, u.options()), gd = at::empty({n}, u.options());
        auto gu = at::empty({n}, u.options()), gr = at::empty({n}, u.options());
        auto* dl = gl.mutable_data_ptr<double>();
        auto* dd = gd.mutable_data_ptr<double>();
        auto* dh = gu.mutable_data_ptr<double>();
        auto* dr = gr.mutable_data_ptr<double>();
        const auto* old = u.const_data_ptr<double>();
        const auto* next = x.const_data_ptr<double>();
        std::fill(dr, dr+n, 0.);
        for (int64_t i = 0; i < n; ++i) {
            const auto left = i == 0 ? 1 : i-1;
            const auto right = i+1 == n ? n-2 : i+1;
            const double w = lambda[i].value();
            TORCH_CHECK(std::isfinite(w), "non-finite step adjoint");
            // Application reflects endpoints; splitting ignores the unused
            // outer bands. Keep their different coefficient VJPs explicit.
            dl[i] = w*(alpha*old[left] + (i == 0 ? 0. : beta*next[i-1]));
            dd[i] = w*(alpha*old[i] + beta*next[i]);
            dh[i] = w*(alpha*old[right] + (i+1 == n ? 0. : beta*next[i+1]));
            dr[i] += w + alpha*w*d[i];
            dr[left] += alpha*w*l[i];
            dr[right] += alpha*w*h[i];
        }
        // dt, theta and time are fixed settings; their passive-only eligibility
        // is checked before entering this boundary.
        if (assembled) return {at::stack({gl, gd, gu}), gr};
        return {gl, gd, gu, gr};
    }
};

class AssembledStepAdjoint : public torch::autograd::Function<AssembledStepAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor bands,
        at::Tensor state, double dt, double theta, double time, int kind) {
        RECORD_FUNCTION("qlt::step_adjoint_forward", std::vector<c10::IValue>());
        {
            RECORD_FUNCTION("qlt::assembled_step_adjoint_forward", std::vector<c10::IValue>());
            AdjointMode adjoint_guard(false);
            TensorKernelMode kernel_guard(0);
            const auto result = native_step_forward(bands, at::Tensor(), at::Tensor(), state,
                                                    dt, theta, time, kind, true);
            TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite step solution");
            ctx->save_for_backward({bands, state, result});
            ctx->saved_data["dt"] = dt; ctx->saved_data["theta"] = theta; ctx->saved_data["kind"] = kind;
            return result;
        }
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list gradients) {
        RECORD_FUNCTION("qlt::assembled_step_adjoint_backward", std::vector<c10::IValue>());
        auto result = StepAdjoint::backward_values(ctx->get_saved_variables(), gradients[0],
            ctx->saved_data["dt"].toDouble(), ctx->saved_data["theta"].toDouble(),
            ctx->saved_data["kind"].toInt(), true, true);
        result.resize(6);
        return result;
    }
};

bool try_assembled_step_adjoint(const at::Tensor& bands, const at::Tensor& state,
    double dt, double theta, double time, int kind, at::Tensor& output) {
    if (!adjoint_assembled_bands_enabled() || !step_arithmetic_eligible({bands, state})) return false;
    TORCH_CHECK(bands.dim() == 2 && bands.size(0) == 3 && state.dim() == 1
                && bands.size(1) == state.numel() && state.numel() >= 2, "invalid assembled adjoint bands");
    output = AssembledStepAdjoint::apply(bands, state, dt, theta, time, kind);
    return true;
}
}

bool try_step_adjoint(QuantLib::FdmLinearOpComposite& map,
    const QuantLib::BoundaryConditionSchemeHelper& boundaries, QuantLib::Array& state,
    QuantLib::Time dt, QuantLib::Real theta, QuantLib::Time time, int kind) {
    const bool full = !adjoint_enabled();
    if (full ? (!full_step_enabled() || tensor_kernel_mode() != 4 || !at::GradMode::is_enabled())
             : (tensor_kernel_mode() < 2 || !enabled())) return false;
    if (!dt.is_passive() || !theta.is_passive() || !time.is_passive()
        || !boundaries.bcSet_.empty() || typeid(map) != typeid(QuantLib::FdmBlackScholesOp)) return false;
    const auto& bs = static_cast<const QuantLib::FdmBlackScholesOp&>(map);
    const auto& op = bs.mapT_;
    if (bs.direction_ != 0 || op.direction_ != 0 || op.mesher_->layout()->dim().size() != 1) return false;
    const auto n = op.mesher_->layout()->size();
    TORCH_CHECK(state.size() == n && n >= 2, "invalid step state size");
    const auto rhs = pack_array(state.begin(), n);
    at::Tensor result;
    if (!full && bs.tensorBands_.defined())
        try_assembled_step_adjoint(bs.tensorBands_, rhs, dt.value(), theta.value(), time.value(), kind, result);
    if (full && bs.tensorBands_.defined() && kind != 2 && n >= 8
        && step_arithmetic_eligible({bs.tensorBands_, rhs})) {
        RECORD_FUNCTION("qlt::full_tensor_step", std::vector<c10::IValue>());
        std::vector<at::Tensor> prepared;
        if (try_assembled_step_preparation(bs.tensorBands_, rhs, op.i0_.get(), op.i2_.get(),
                                           dt.value(), theta.value(), kind, prepared)) {
            RECORD_FUNCTION("qlt::assembled_bands_step", std::vector<c10::IValue>());
            // A rejected candidate returns to the original QuantLib step,
            // without retrying the same solve or changing its input state.
            if (!try_auto_solve_tensor(prepared[0], prepared[1], prepared[2], prepared[3], result)) return false;
        }
    }
    if (!result.defined()) {
        const auto lo = pack_array(op.lower_.get(), n), di = pack_array(op.diag_.get(), n);
        const auto up = pack_array(op.upper_.get(), n);
        if (full) {
            if (!try_full_tensor_step(op, lo, di, up, rhs, op.i0_.get(), op.i2_.get(),
                                      dt.value(), theta.value(), kind, result)) return false;
        } else {
            result = StepAdjoint::apply(lo, di, up, rhs, &op, dt.value(), theta.value(), time.value(), kind, true);
        }
    }
    QuantLib::Array output(n);
    unpack_array(result, output.begin());
    state = std::move(output);
    return true;
}

namespace {
class ProbeBand final : public QuantLib::TripleBandLinearOp {
  public:
    explicit ProbeBand(const at::Tensor& bands) : TripleBandLinearOp(0,
        QuantLib::ext::make_shared<QuantLib::FdmMesherComposite>(
            QuantLib::ext::make_shared<QuantLib::Uniform1dMesher>(0., 1., bands.size(1)))) {
        for (int64_t i = 0; i < bands.size(1); ++i) {
            lower_[i] = TorchReal(bands[0][i]);
            diag_[i] = TorchReal(bands[1][i]);
            upper_[i] = TorchReal(bands[2][i]);
        }
    }
    const QuantLib::Size* left() const { return i0_.get(); }
    const QuantLib::Size* right() const { return i2_.get(); }
};

at::Tensor step_probe(const at::Tensor& bands, const at::Tensor& state,
                     double dt, double theta, int64_t kind, bool adjoint, bool native,
                     int64_t mode, bool tensor_step, bool assembled) {
    for (const auto& t : {bands, state})
        TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided,
                    "step probe requires dense CPU float64 inputs");
    TORCH_CHECK(bands.dim() == 2 && bands.size(0) == 3 && bands.size(1) >= 2
                && state.dim() == 1 && state.numel() == bands.size(1), "invalid step probe shapes");
    TORCH_CHECK(std::isfinite(dt) && dt > 0. && std::isfinite(theta) && kind >= 0 && kind <= 2,
                "invalid step probe settings");
    AdjointMode adjoint_guard(false);
    TensorKernelMode kernel_guard(static_cast<int>(mode));
    ProbeBand op(bands);
    at::Tensor result;
    if (adjoint && native && assembled && try_assembled_step_adjoint(bands, state, dt, theta, dt,
                                                                    static_cast<int>(kind), result)) return result;
    if (adjoint) return StepAdjoint::apply(bands[0], bands[1], bands[2], state,
                                         &op, dt, theta, dt, static_cast<int>(kind), native);
    if (tensor_step && try_full_tensor_step(op, bands[0], bands[1], bands[2], state,
                                          op.left(), op.right(), dt, theta, static_cast<int>(kind), result)) return result;
    QuantLib::Array values(state.numel());
    unpack_array(state, values.begin());
    upstream_step(QuantLib::ext::make_shared<SnapshotComposite>(op, false), values, dt, theta, dt, static_cast<int>(kind));
    return pack_array(values.begin(), values.size());
}
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_step_probe(Tensor bands, Tensor state, float dt, float theta, int kind, bool adjoint, bool native=True, int mode=0, bool tensor_step=False, bool assembled=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_step_probe", TORCH_FN(quantlibtorch::step_probe));
}
