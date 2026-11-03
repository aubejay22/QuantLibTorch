#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/operator_arithmetic.hpp>
#include <quantlibtorch/step_preparation.hpp>
#include <quantlibtorch/step_arithmetic.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/tridiagonaloperator.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/functorch/DynamicLayer.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>

namespace quantlibtorch {
namespace {
thread_local bool use_adjoint = false;

at::Tensor stack(const QuantLib::Real* values, QuantLib::Size n) {
    std::vector<at::Tensor> tensors;
    tensors.reserve(n);
    for (QuantLib::Size i = 0; i < n; ++i)
        tensors.push_back(values[i].tensor());
    return at::stack(tensors);
}

QuantLib::Array array(const at::Tensor& tensor) {
    QuantLib::Array values(tensor.numel());
    for (int64_t i = 0; i < tensor.numel(); ++i)
        values[i] = TorchReal(tensor.select(0, i));
    return values;
}

// Explicit numerical boundary INSIDE the custom autograd operation. This is
// never used by TorchReal's constructors or arithmetic: tensor coefficients
// remain connected through the saved bands and the analytical solve VJP.
at::Tensor readable(const at::Tensor& tensor) {
    // contiguity alone does not materialize PyTorch's lazy negative/conjugate
    // bits. Raw storage reads must reflect the tensor's logical values.
    auto values = tensor.is_neg() ? tensor.resolve_neg() : tensor;
    if (values.is_conj()) values = values.resolve_conj();
    return values.contiguous();
}

bool arithmetic_eligible(at::TensorList inputs) {
    if (at::functorch::functorchTLSAccessor()
        && !at::functorch::getDynamicLayerStack().empty()) return false;
    for (const auto& t : inputs) {
        const auto keys = t.key_set();
        if (!t.is_cpu() || keys.has(c10::DispatchKey::Python) || keys.has(c10::DispatchKey::Functionalize)
            || keys.has(c10::DispatchKey::FuncTorchBatched) || keys.has(c10::DispatchKey::FuncTorchGradWrapper)
            || t._fw_grad(0).defined()) return false;
    }
    return true;
}

bool adjoint_arithmetic_forward_enabled() {
    static const bool enabled = performance_flag("QLT_ADJOINT_ARITHMETIC_FORWARD");
    return enabled;
}

QuantLib::Array passive_array(const at::Tensor& tensor) {
    const auto contiguous = readable(tensor);
    const auto* data = contiguous.const_data_ptr<double>();
    QuantLib::Array values(tensor.numel());
    for (int64_t i = 0; i < tensor.numel(); ++i) values[i] = TorchReal(data[i]);
    return values;
}

class PassiveBand final : public QuantLib::TripleBandLinearOp {
  public:
    explicit PassiveBand(const QuantLib::TripleBandLinearOp& op)
    : QuantLib::TripleBandLinearOp(op) {
        for (QuantLib::Size i = 0; i < mesher_->layout()->size(); ++i) {
            lower_[i] = TorchReal(lower_[i].value());
            diag_[i] = TorchReal(diag_[i].value());
            upper_[i] = TorchReal(upper_[i].value());
        }
    }
    at::Tensor bands() const {
        const auto n = mesher_->layout()->size();
        return at::stack({pack_array(lower_.get(), n), pack_array(diag_.get(), n), pack_array(upper_.get(), n)});
    }
};

template <bool FullDerivatives>
class DiffusionFunction : public torch::autograd::Function<DiffusionFunction<FullDerivatives>> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor variance, at::Tensor dx, at::Tensor dxx, at::Tensor rate, at::Tensor yield,
        const QuantLib::TripleBandLinearOp* dx_op, const QuantLib::TripleBandLinearOp* dxx_op) {
        RECORD_FUNCTION(FullDerivatives ? "qlt::diffusion_arithmetic_forward" : "qlt::diffusion_adjoint_forward",
                        std::vector<c10::IValue>());
        const bool native_forward = FullDerivatives ? arithmetic_native_forward_enabled()
            : adjoint_arithmetic_forward_enabled() && arithmetic_eligible({variance, dx, dxx, rate, yield});
        if (native_forward) {
            ctx->save_for_backward({variance, dx, dxx, rate, yield});
            return native_diffusion_arithmetic(variance, dx, dxx, rate, yield);
        }
        AdjointMode adjoint_guard(false);
        TensorKernelMode kernel_guard(0);
        const auto v = passive_array(variance);
        const auto r = TorchReal(rate.item<double>()), q = TorchReal(yield.item<double>());
        const PassiveBand first(*dx_op), second(*dxx_op);
        PassiveBand output(first);
        // Reuse the original mult and axpyb algorithms, in their original
        // order, with native values ONLY inside this explicit VJP boundary.
        output.axpyb(r-q-0.5*v, first, second.mult(0.5*v), QuantLib::Array(1, -r));
        ctx->save_for_backward({variance, dx, dxx, rate, yield});
        return output.bands();
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list gradients) {
        RECORD_FUNCTION(FullDerivatives ? "qlt::diffusion_arithmetic_backward" : "qlt::diffusion_adjoint_backward",
                        std::vector<c10::IValue>());
        TORCH_CHECK(FullDerivatives || !at::GradMode::is_enabled(),
                    "QuantLib diffusion adjoint supports first derivatives only; use mode='full' for higher derivatives");
        const auto saved = ctx->get_saved_variables();
        if constexpr (FullDerivatives) {
            if (at::GradMode::is_enabled() || gradients[0]._fw_grad(0).defined()) {
                RECORD_FUNCTION("qlt::diffusion_arithmetic_backward_graph", std::vector<c10::IValue>());
                // Differentiate the elementary assembly expression, using
                // ORIGINAL saved tensors and incoming graph-connected VJPs,
                // including forward tangents supplied without create_graph.
                const auto& v = saved[0]; const auto& x = saved[1]; const auto& xx = saved[2];
                const auto& rate = saved[3]; const auto& yield = saved[4]; const auto& g = gradients[0];
                static const bool selected = performance_flag("QLT_DIFFUSION_SELECTED_GRADS");
                const bool need_v = !selected || ctx->needs_input_grad(0);
                const bool need_x = !selected || ctx->needs_input_grad(1);
                const bool need_xx = !selected || ctx->needs_input_grad(2);
                const bool need_r = !selected || ctx->needs_input_grad(3);
                const bool need_q = !selected || ctx->needs_input_grad(4);
                // Skip only outputs that this backward task does not request.
                // Every requested expression still uses ALL of its original
                // operands, including active operands whose own VJP is skipped:
                // a later mixed derivative can depend on those saved inputs.
                at::Tensor wx, gv, gx, gxx, gr, gq;
                if (need_v || need_r || need_q) wx = (g*x).sum(0);
                if (need_v) gv = 0.5*((g*xx).sum(0)-wx);
                if (need_x) gx = g*(rate-yield-0.5*v).unsqueeze(0);
                if (need_xx) gxx = g*(0.5*v).unsqueeze(0);
                if (need_r) gr = (wx-g.select(0, 1)).sum();
                if (need_q) gq = -wx.sum();
                return {gv, gx, gxx, gr, gq, at::Tensor(), at::Tensor()};
            }
        }
        const auto v = readable(saved[0]), x = readable(saved[1]), xx = readable(saved[2]);
        const auto g = readable(gradients[0]);
        const auto n = v.numel();
        const double rate = saved[3].item<double>(), yield = saved[4].item<double>();
        auto gv = ctx->needs_input_grad(0) ? at::empty({n}, v.options()) : at::Tensor();
        auto gx = ctx->needs_input_grad(1) ? at::empty({3, n}, x.options()) : at::Tensor();
        auto gxx = ctx->needs_input_grad(2) ? at::empty({3, n}, xx.options()) : at::Tensor();
        const auto* values = v.const_data_ptr<double>();
        const auto* first = x.const_data_ptr<double>();
        const auto* second = xx.const_data_ptr<double>();
        const auto* weight = g.const_data_ptr<double>();
        auto* dv = gv.defined() ? gv.mutable_data_ptr<double>() : nullptr;
        auto* dx = gx.defined() ? gx.mutable_data_ptr<double>() : nullptr;
        auto* dxx = gxx.defined() ? gxx.mutable_data_ptr<double>() : nullptr;
        double dr = 0., dq = 0.;
        for (int64_t i = 0; i < n; ++i) {
            double wx = 0., wxx = 0.;
            for (int64_t k = 0; k < 3; ++k) {
                const auto j = k*n+i;
                wx += weight[j]*first[j];
                wxx += weight[j]*second[j];
                if (dx) dx[j] = weight[j]*(rate-yield-0.5*values[i]);
                if (dxx) dxx[j] = weight[j]*(0.5*values[i]);
            }
            if (dv) dv[i] = 0.5*(wxx-wx);
            dr += wx-weight[n+i];
            dq -= wx;
        }
        return {gv, gx, gxx,
            ctx->needs_input_grad(3) ? at::scalar_tensor(dr, saved[3].options()) : at::Tensor(),
            ctx->needs_input_grad(4) ? at::scalar_tensor(dq, saved[4].options()) : at::Tensor(),
            at::Tensor(), at::Tensor()};
    }
};
using DiffusionAdjoint = DiffusionFunction<false>;
using DiffusionArithmetic = DiffusionFunction<true>;

template <bool FullDerivatives>
class ApplyFunction : public torch::autograd::Function<ApplyFunction<FullDerivatives>> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor rhs,
        at::Tensor left, at::Tensor right, const QuantLib::TripleBandLinearOp* op) {
        RECORD_FUNCTION(FullDerivatives ? "qlt::apply_arithmetic_forward" : "qlt::apply_adjoint_forward",
                        std::vector<c10::IValue>());
        const bool native_forward = FullDerivatives ? arithmetic_native_forward_enabled()
            : adjoint_arithmetic_forward_enabled() && arithmetic_eligible({lower, diagonal, upper, rhs});
        if (native_forward) {
            ctx->save_for_backward({lower, diagonal, upper, rhs, left, right});
            return native_apply_arithmetic(lower, diagonal, upper, rhs, left, right);
        }
        AdjointMode adjoint_guard(false);
        TensorKernelMode kernel_guard(0);
        // Re-enter the original QuantLib stencil, with its actual neighbour
        // indices (including reflected endpoints), inside this explicit VJP.
        const auto values = PassiveBand(*op).apply(passive_array(rhs));
        ctx->save_for_backward({lower, diagonal, upper, rhs, left, right});
        return pack_array(values.begin(), values.size());
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list gradients) {
        RECORD_FUNCTION(FullDerivatives ? "qlt::apply_arithmetic_backward" : "qlt::apply_adjoint_backward",
                        std::vector<c10::IValue>());
        TORCH_CHECK(FullDerivatives || !at::GradMode::is_enabled(),
                    "QuantLib operator adjoint supports first derivatives only; use mode='full' for higher derivatives");
        const auto saved = ctx->get_saved_variables();
        if constexpr (FullDerivatives) {
            if (at::GradMode::is_enabled() || gradients[0]._fw_grad(0).defined()) {
                RECORD_FUNCTION("qlt::apply_arithmetic_backward_graph", std::vector<c10::IValue>());
                const auto& lo = saved[0]; const auto& di = saved[1]; const auto& up = saved[2];
                const auto& r = saved[3]; const auto& left = saved[4]; const auto& right = saved[5];
                const auto& g = gradients[0];
                // Functional scatter-add preserves duplicate reflected
                // neighbours and all higher dependencies without mutation.
                const auto gr = (di*g).index_add(0, left, lo*g).index_add(0, right, up*g);
                return {g*r.index_select(0, left), g*r, g*r.index_select(0, right), gr,
                        at::Tensor(), at::Tensor(), at::Tensor()};
            }
        }
        const auto lo = readable(saved[0]), di = readable(saved[1]), up = readable(saved[2]);
        const auto rhs = readable(saved[3]), g = readable(gradients[0]);
        const auto left = saved[4].contiguous(), right = saved[5].contiguous();
        const auto n = rhs.numel();
        auto gl = at::empty({n}, rhs.options()), gd = at::empty({n}, rhs.options());
        auto gu = at::empty({n}, rhs.options()), gr = at::empty({n}, rhs.options());
        const auto* l = lo.const_data_ptr<double>();
        const auto* d = di.const_data_ptr<double>();
        const auto* u = up.const_data_ptr<double>();
        const auto* r = rhs.const_data_ptr<double>();
        const auto* w = g.const_data_ptr<double>();
        const auto* li = left.const_data_ptr<int64_t>();
        const auto* ri = right.const_data_ptr<int64_t>();
        auto* dl = gl.mutable_data_ptr<double>();
        auto* dd = gd.mutable_data_ptr<double>();
        auto* du = gu.mutable_data_ptr<double>();
        auto* dr = gr.mutable_data_ptr<double>();
        // Only fresh gradient buffers are written. Inputs and scalar-adapter
        // tensors are immutable. Scatter-add accounts for repeated neighbours.
        std::fill(dr, dr+n, 0.);
        for (int64_t i = 0; i < n; ++i) {
            dl[i] = w[i]*r[li[i]];
            dd[i] = w[i]*r[i];
            du[i] = w[i]*r[ri[i]];
            dr[li[i]] += w[i]*l[i];
            dr[i] += w[i]*d[i];
            dr[ri[i]] += w[i]*u[i];
        }
        return {gl, gd, gu, gr, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};
using ApplyAdjoint = ApplyFunction<false>;
using ApplyArithmetic = ApplyFunction<true>;

class SplitAdjoint : public torch::autograd::Function<SplitAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor rhs,
        const QuantLib::TripleBandLinearOp* op, QuantLib::Real a, QuantLib::Real b) {
        RECORD_FUNCTION("qlt::adjoint_forward", std::vector<c10::IValue>());
        // Function.forward disables recording. Re-enter the identical upstream
        // TripleBandLinearOp algorithm; the hook is disabled only for this call.
        AdjointMode adjoint_guard(false);
        const bool native = native_adjoint_enabled();
        const auto values = native
            ? PassiveBand(*op).solve_splitting(passive_array(rhs), TorchReal(a.value()), TorchReal(b.value()))
            : op->solve_splitting(array(rhs), a, b);
        auto x = native ? pack_array(values.begin(), values.size()) : stack(values.begin(), values.size());
        TORCH_CHECK(at::isfinite(x).all().item<bool>(), "non-finite splitting solution");
        ctx->save_for_backward({lower, diagonal, upper, x});
        return x;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list gradients) {
        RECORD_FUNCTION("qlt::adjoint_backward", std::vector<c10::IValue>());
        AdjointMode adjoint_guard(false);
        TORCH_CHECK(!at::GradMode::is_enabled(),
            "QuantLib FD adjoint supports first derivatives only; use mode='full' for higher derivatives");
        const auto saved = ctx->get_saved_variables();
        const auto& lower = saved[0];
        const auto& diagonal = saved[1];
        const auto& upper = saved[2];
        const auto& x = saved[3];
        const bool native = native_adjoint_enabled();
        const auto convert = native ? passive_array : array;
        QuantLib::TridiagonalOperator transpose(convert(upper), convert(diagonal), convert(lower));
        const auto values = transpose.solveFor(convert(gradients[0]));
        auto lam = native ? pack_array(values.begin(), values.size()) : stack(values.begin(), values.size());
        TORCH_CHECK(at::isfinite(lam).all().item<bool>(), "non-finite splitting adjoint");
        const auto n = x.numel();
        return {-lam.slice(0, 1, n) * x.slice(0, 0, n-1), -lam*x,
                -lam.slice(0, 0, n-1) * x.slice(0, 1, n), lam,
                at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class TridiagonalAdjoint : public torch::autograd::Function<TridiagonalAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor rhs) {
        RECORD_FUNCTION("qlt::tridiagonal_adjoint_forward", std::vector<c10::IValue>());
        AdjointMode adjoint_guard(false);
        QuantLib::TridiagonalOperator op(passive_array(lower), passive_array(diagonal), passive_array(upper));
        const auto values = op.solveFor(passive_array(rhs));
        auto x = pack_array(values.begin(), values.size());
        TORCH_CHECK(at::isfinite(x).all().item<bool>(), "non-finite tridiagonal solution");
        ctx->save_for_backward({lower, diagonal, upper, x});
        return x;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list gradients) {
        RECORD_FUNCTION("qlt::tridiagonal_adjoint_backward", std::vector<c10::IValue>());
        TORCH_CHECK(!at::GradMode::is_enabled(), "QuantLib tridiagonal adjoint supports first derivatives only");
        AdjointMode adjoint_guard(false);
        const auto saved = ctx->get_saved_variables();
        QuantLib::TridiagonalOperator transpose(passive_array(saved[2]), passive_array(saved[1]), passive_array(saved[0]));
        const auto values = transpose.solveFor(passive_array(gradients[0]));
        const auto lam = pack_array(values.begin(), values.size());
        const auto& x = saved[3];
        TORCH_CHECK(at::isfinite(lam).all().item<bool>(), "non-finite tridiagonal adjoint");
        const auto n = x.numel();
        return {-lam.slice(0, 1, n)*x.slice(0, 0, n-1), -lam*x,
                -lam.slice(0, 0, n-1)*x.slice(0, 1, n), lam};
    }
};
} // namespace

bool adjoint_enabled() { return use_adjoint; }
bool operator_adjoint_enabled() {
    static const bool enabled = performance_flag("QLT_OPERATOR_ADJOINT");
    return enabled && native_adjoint_enabled();
}
bool operator_arithmetic_enabled() {
    static const bool enabled = performance_flag("QLT_OPERATOR_ARITHMETIC");
    return enabled;
}
AdjointMode::AdjointMode(bool enabled) : previous_(use_adjoint) { use_adjoint = enabled; }
AdjointMode::~AdjointMode() { use_adjoint = previous_; }

bool try_diffusion_assembly(QuantLib::TripleBandLinearOp& output,
    const QuantLib::TripleBandLinearOp& dx, const QuantLib::TripleBandLinearOp& dxx,
    const QuantLib::Array& variance, QuantLib::Real rate, QuantLib::Real yield,
    DiffusionBasisCache* immutable_basis, at::Tensor* assembled_bands) {
    // A caller must never reuse a preceding assembly after this attempt fails.
    if (assembled_bands) *assembled_bands = at::Tensor();
    static const bool enabled = performance_flag("QLT_ASSEMBLY_ADJOINT");
    static const bool arithmetic = performance_flag("QLT_ASSEMBLY_ARITHMETIC");
    // The owner may disable reuse for an independent packing control.
    static const bool cached = performance_flag("QLT_BASIS_CACHE");
    const bool full = !adjoint_enabled();
    if (tensor_kernel_mode() < 2 || (full ? !arithmetic : (!enabled || !native_adjoint_enabled()))) return false;
    const auto n = variance.size();
    TORCH_CHECK(dx.mesher_->layout()->size() == n && dxx.mesher_->layout()->size() == n
                && output.mesher_->layout()->size() == n, "invalid diffusion assembly sizes");
    const auto pack = [n](const QuantLib::TripleBandLinearOp& op) {
        return at::stack({pack_array(op.lower_.get(), n), pack_array(op.diag_.get(), n), pack_array(op.upper_.get(), n)});
    };
    const auto v = pack_array(variance.begin(), n);
    const auto r = rate.tensor(), q = yield.tensor();
    at::Tensor x, xx;
    // This cache is supplied only by the owner of const dx/dxx members. Reject
    // aliasing outputs and transformed inputs, and never retain an active basis.
    // Inference tensors must not later be saved by a normal autograd operation.
    auto* cache = cached && immutable_basis && &output != &dx && &output != &dxx
        && arithmetic_eligible({v, r, q}) ? immutable_basis : nullptr;
    if (cache) {
        if (!cache->checked_) {
            cache->passive_ = true;
            for (QuantLib::Size i = 0; cache->passive_ && i < n; ++i)
                cache->passive_ = dx.lower_[i].is_passive() && dx.diag_[i].is_passive()
                    && dx.upper_[i].is_passive() && dxx.lower_[i].is_passive()
                    && dxx.diag_[i].is_passive() && dxx.upper_[i].is_passive();
            cache->checked_ = true;
        }
        if (cache->passive_ && cache->first_.defined()
            && cache->first_.is_inference() == c10::InferenceMode::is_enabled()) {
            RECORD_FUNCTION("qlt::diffusion_basis_reuse", std::vector<c10::IValue>());
            x = cache->first_; xx = cache->second_;
        }
    }
    if (!x.defined()) {
        RECORD_FUNCTION("qlt::diffusion_basis_pack", std::vector<c10::IValue>());
        // Save all operands before rebinding output; aliases preserve value semantics.
        x = pack(dx); xx = pack(dxx);
        if (cache && cache->passive_ && arithmetic_eligible({x, xx})) {
            cache->first_ = x; cache->second_ = xx;
        }
    }
    if (full && !arithmetic_eligible({v, x, xx, r, q})) return false;
    const auto result = full ? DiffusionArithmetic::apply(v, x, xx, r, q, &dx, &dxx)
                             : DiffusionAdjoint::apply(v, x, xx, r, q, &dx, &dxx);
    unpack_array(result[0], output.lower_.get());
    unpack_array(result[1], output.diag_.get());
    unpack_array(result[2], output.upper_.get());
    const bool keep_full = assembled_bands && full && tensor_kernel_mode() == 4 && at::GradMode::is_enabled()
        && assembled_bands_enabled();
    const bool keep_adjoint = assembled_bands && !full && adjoint_assembled_bands_enabled()
        && step_arithmetic_eligible({v, x, xx, r, q, result});
    if (keep_full || keep_adjoint) {
        RECORD_FUNCTION("qlt::assembled_bands_store", std::vector<c10::IValue>());
        // Keep the original active result as well as the materialized Array
        // coefficients. No storage/base inference or detached reconstruction.
        *assembled_bands = result;
    }
    return true;
}

bool try_full_apply_tensor(const QuantLib::TripleBandLinearOp& op,
    const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up, const at::Tensor& rhs,
    const QuantLib::Size* left, const QuantLib::Size* right, at::Tensor& output) {
    if (!operator_arithmetic_enabled() || !arithmetic_eligible({lo, di, up, rhs})) return false;
    const auto n = rhs.numel();
    const auto indices = at::TensorOptions().dtype(at::kLong);
    const auto li = at::tensor(std::vector<int64_t>(left, left+n), indices);
    const auto ri = at::tensor(std::vector<int64_t>(right, right+n), indices);
    output = ApplyArithmetic::apply(lo, di, up, rhs, li, ri, &op);
    return true;
}

QuantLib::Array adjoint_apply(const QuantLib::TripleBandLinearOp& op, const QuantLib::Array& rhs,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal, const QuantLib::Real* upper,
    const QuantLib::Size* left, const QuantLib::Size* right) {
    const auto n = rhs.size();
    const auto lo = pack_array(lower, n), di = pack_array(diagonal, n), up = pack_array(upper, n);
    const auto r = pack_array(rhs.begin(), n);
    const bool full = !adjoint_enabled();
    at::Tensor result;
    if (full) {
        if (!try_full_apply_tensor(op, lo, di, up, r, left, right, result))
            return vector_apply(rhs, lower, diagonal, upper, left, right);
    } else {
        const auto indices = at::TensorOptions().dtype(at::kLong);
        const auto li = at::tensor(std::vector<int64_t>(left, left+n), indices);
        const auto ri = at::tensor(std::vector<int64_t>(right, right+n), indices);
        result = ApplyAdjoint::apply(lo, di, up, r, li, ri, &op);
    }
    QuantLib::Array output(n);
    unpack_array(result, output.begin());
    return output;
}

QuantLib::Array adjoint_tridiagonal(const QuantLib::Array& lower, const QuantLib::Array& diagonal,
    const QuantLib::Array& upper, const QuantLib::Array& rhs) {
    const auto n = rhs.size();
    TORCH_CHECK(n >= 2 && diagonal.size() == n && lower.size() == n-1 && upper.size() == n-1,
                "invalid tridiagonal adjoint sizes");
    const auto result = TridiagonalAdjoint::apply(pack_array(lower.begin(), n-1),
        pack_array(diagonal.begin(), n), pack_array(upper.begin(), n-1), pack_array(rhs.begin(), n));
    QuantLib::Array output(n);
    unpack_array(result, output.begin());
    return output;
}

QuantLib::Array adjoint_split(const QuantLib::TripleBandLinearOp& op,
    const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper) {
    const auto n = rhs.size();
    TORCH_CHECK(n >= 2, "splitting adjoint requires at least two nodes");
    // Assemble A=a*L+b*I OUTSIDE Function.forward so a, b and every operator
    // coefficient retain their original graph. No tensor data is detached.
    const auto pack = native_adjoint_enabled() ? pack_array : stack;
    auto lo = a.tensor() * pack(lower + 1, n-1);
    auto di = a.tensor() * pack(diagonal, n) + b.tensor();
    auto up = a.tensor() * pack(upper, n-1);
    auto result = SplitAdjoint::apply(lo, di, up, pack(rhs.begin(), n), &op, a, b);
    if (!native_adjoint_enabled()) return array(result);
    QuantLib::Array output(n);
    unpack_array(result, output.begin());
    return output;
}
} // namespace quantlibtorch

namespace {
// An internal verification entry point for arbitrary nonsymmetric band systems.
// This uses the exact same hook as FdmBlackScholesOp, independent of PDE data.
class TestBand final : public QuantLib::TripleBandLinearOp {
  public:
    TestBand(const at::Tensor& lower, const at::Tensor& diagonal, const at::Tensor& upper)
    : QuantLib::TripleBandLinearOp(0,
        QuantLib::ext::make_shared<QuantLib::FdmMesherComposite>(
            QuantLib::ext::make_shared<QuantLib::Uniform1dMesher>(0.0, 1.0, diagonal.numel()))) {
        for (int64_t i = 0; i < diagonal.numel(); ++i) {
            diag_[i] = quantlibtorch::TorchReal(diagonal[i]);
            lower_[i] = i > 0 ? quantlibtorch::TorchReal(lower[i-1]) : quantlibtorch::TorchReal(0.0);
            upper_[i] = i+1 < diagonal.numel() ? quantlibtorch::TorchReal(upper[i]) : quantlibtorch::TorchReal(0.0);
        }
    }
};

at::Tensor band_solve(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, const at::Tensor& a,
    const at::Tensor& b, bool adjoint, int64_t kernels) {
    for (const auto& t : {lower, diagonal, upper, rhs, a, b}) {
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided,
                    "band inputs must be dense CPU float64 tensors");
        TORCH_CHECK(at::isfinite(t).all().item<bool>(), "band inputs must be finite");
    }
    TORCH_CHECK(lower.dim() == 1 && diagonal.dim() == 1 && upper.dim() == 1 && rhs.dim() == 1
                && a.dim() == 0 && b.dim() == 0, "band input dimensions invalid");
    const auto n = diagonal.numel();
    TORCH_CHECK(n >= 2 && rhs.numel() == n && lower.numel() == n-1 && upper.numel() == n-1,
                "band input lengths invalid");
    quantlibtorch::AdjointMode mode(adjoint);
    quantlibtorch::TensorKernelMode kernel_mode(static_cast<int>(kernels));
    TestBand op(lower, diagonal, upper);
    QuantLib::Array r(n);
    for (int64_t i = 0; i < n; ++i) r[i] = quantlibtorch::TorchReal(rhs[i]);
    const auto result = op.solve_splitting(r, quantlibtorch::TorchReal(a), quantlibtorch::TorchReal(b));
    std::vector<at::Tensor> scalars;
    for (const auto& v : result) scalars.push_back(v.tensor());
    auto x = at::stack(scalars);
    TORCH_CHECK(at::isfinite(x).all().item<bool>(), "non-finite splitting solution");
    return x;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_band_solve(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, Tensor a, Tensor b, bool adjoint, int kernels=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_band_solve", TORCH_FN(band_solve));
}
