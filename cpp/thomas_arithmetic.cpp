// Forward recurrence adapted from QuantLib v1.40, commit
// 06edddbdb85f006c43a449b5fc846a00c67477f3,
// ql/methods/finitedifferences/tridiagonaloperator.cpp, solveFor.
// Copyright (C) 2002, 2003, 2011 Ferdinando Ametrano
// Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl
// Distributed under the QuantLib license; see LICENSES/QuantLib.txt for the
// complete license, conditions and disclaimer. The reverse recurrence below
// differentiates these elementary operations; it is not an implicit solve VJP.
#include <quantlibtorch/thomas_arithmetic.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cmath>

namespace quantlibtorch {
namespace {
using Tensors = std::vector<at::Tensor>;

at::Tensor readable(const at::Tensor& t) {
    auto values = t.is_neg() ? t.resolve_neg() : t;
    if (values.is_conj()) values = values.resolve_conj();
    return values.contiguous();
}

bool eligible(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up, const at::Tensor& r) {
    if (!d.is_cpu()) return false;
    if (at::functorch::functorchTLSAccessor() && !at::functorch::getDynamicLayerStack().empty()) return false;
    for (const auto& t : {lo, d, up, r}) {
        const auto keys = t.key_set();
        if (keys.has(c10::DispatchKey::Python) || keys.has(c10::DispatchKey::Functionalize)
            || keys.has(c10::DispatchKey::FuncTorchBatched) || keys.has(c10::DispatchKey::FuncTorchGradWrapper)
            || t._fw_grad(0).defined()) return false;
    }
    return true;
}

Tensors normalize(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up, const at::Tensor& r) {
    auto batch = at::infer_size(lo.sizes().slice(0, lo.dim()-1), d.sizes().slice(0, d.dim()-1));
    batch = at::infer_size(batch, up.sizes().slice(0, up.dim()-1));
    batch = at::infer_size(batch, r.sizes().slice(0, r.dim()-2));
    auto off_shape = batch, diag_shape = batch, rhs_shape = batch;
    off_shape.push_back(d.size(-1)-1); diag_shape.push_back(d.size(-1));
    rhs_shape.push_back(d.size(-1)); rhs_shape.push_back(r.size(-1));
    const auto expand = [](const at::Tensor& t, const std::vector<int64_t>& shape) {
        return t.sizes() == at::IntArrayRef(shape) ? t : t.expand(shape);
    };
    // Broadcast reductions remain outside the custom operation.
    return {expand(lo, off_shape), expand(d, diag_shape), expand(up, off_shape), expand(r, rhs_shape)};
}

void check_tensor_pivot(const at::Tensor& p) {
    TORCH_CHECK((at::isfinite(p) & (p != 0.)).all().item<bool>(),
                "Thomas arithmetic requires finite nonzero elimination pivots");
}

struct TensorState { Tensors pivot, factor, forward_rhs, solution; };

TensorState tensor_forward(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up,
                           const at::Tensor& r, bool checked) {
    const auto n = d.size(-1);
    TensorState state;
    auto& p = state.pivot; auto& t = state.factor; auto& y = state.forward_rhs;
    p.push_back(d.select(-1, 0)); t.push_back(at::zeros_like(p[0]));
    if (checked) check_tensor_pivot(p[0]);
    y.push_back(r.select(-2, 0)/p[0].unsqueeze(-1));
    for (int64_t i = 1; i < n; ++i) {
        t.push_back(up.select(-1, i-1)/p[i-1]);
        p.push_back(d.select(-1, i)-lo.select(-1, i-1)*t[i]);
        if (checked) check_tensor_pivot(p[i]);
        y.push_back((r.select(-2, i)-lo.select(-1, i-1).unsqueeze(-1)*y[i-1])/p[i].unsqueeze(-1));
    }
    auto& x = state.solution;
    x = y;
    for (int64_t i = n-2; i >= 0; --i) x[i] = y[i]-t[i+1].unsqueeze(-1)*x[i+1];
    return state;
}

at::Tensor graph_solve(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up, const at::Tensor& r);

at::Tensor gradient_or_zero(const at::Tensor& gradient, const at::Tensor& output) {
    return gradient.defined() ? gradient : at::zeros_like(output);
}

Tensors tensor_backward(const Tensors& saved, const Tensors& gradients) {
    RECORD_FUNCTION("qlt::thomas_arithmetic_backward_graph", std::vector<c10::IValue>());
    const auto& lo = saved[0]; const auto& d = saved[1]; const auto& up = saved[2]; const auto& r = saved[3];
    // The diagnostic serial fallback reconstructs every intermediate from
    // the saved originals, independently of the vector backward's state edges.
    const auto state = tensor_forward(lo, d, up, r, false);
    const auto& p = state.pivot; const auto& t = state.factor;
    const auto& y = state.forward_rhs; const auto& x = state.solution;
    const auto n = d.size(-1);
    auto gy = gradient_or_zero(gradients[0], r).unbind(-2);
    auto gp = gradient_or_zero(gradients[1], d).unbind(-1);
    auto gt = gradient_or_zero(gradients[2], d).unbind(-1);
    Tensors gl(n-1), gd(n), gu(n-1), gr(n);
    for (int64_t i = 0; i < n-1; ++i) {
        gt[i+1] = gt[i+1]-(gy[i]*x[i+1]).sum(-1);
        gy[i+1] = gy[i+1]-t[i+1].unsqueeze(-1)*gy[i];
    }
    if (gradients[3].defined())
        for (int64_t i = 0; i < n; ++i) gy[i] = gy[i]+gradients[3].select(-2, i);
    for (int64_t i = n-1; i >= 1; --i) {
        const auto gz = gy[i]/p[i].unsqueeze(-1);
        gp[i] = gp[i]-(gz*y[i]).sum(-1);
        gr[i] = gz;
        gl[i-1] = -(gz*y[i-1]).sum(-1)-gp[i]*t[i];
        gy[i-1] = gy[i-1]-lo.select(-1, i-1).unsqueeze(-1)*gz;
        gd[i] = gp[i];
        gt[i] = gt[i]-gp[i]*lo.select(-1, i-1);
        gu[i-1] = gt[i]/p[i-1];
        gp[i-1] = gp[i-1]-gu[i-1]*t[i];
    }
    gr[0] = gy[0]/p[0].unsqueeze(-1);
    gd[0] = gp[0]-(gr[0]*y[0]).sum(-1);
    return {at::stack(gl, -1), at::stack(gd, -1), at::stack(gu, -1), at::stack(gr, -2), at::Tensor()};
}

Tensors vector_backward(const Tensors& saved, const Tensors& gradients) {
    RECORD_FUNCTION("qlt::thomas_arithmetic_backward_vector", std::vector<c10::IValue>());
    const auto& lo = saved[0]; const auto& d = saved[1]; const auto& r = saved[3];
    const auto& p = saved[4]; const auto& t = saved[5]; const auto& y = saved[6]; const auto& x = saved[7];
    const auto n = d.size(-1);
    const auto one = at::ones_like(p), zero = at::zeros_like(lo);
    const auto previous_p = p.slice(-1, 0, n-1), next_t = t.slice(-1, 1, n);
    // These three triangular recurrences reverse back substitution, RHS
    // elimination, and pivot/factor elimination respectively. Gradients of
    // ALL intermediate outputs are included, not just the final solution.
    const auto w = graph_solve(next_t, one, zero, gradient_or_zero(gradients[0], r));
    const auto gy = gradients[3].defined() ? w+gradients[3] : w;
    auto gt = at::cat({at::zeros_like(t.slice(-1, 0, 1)),
        -(w.slice(-2, 0, n-1)*x.slice(-2, 1, n)).sum(-1)}, -1);
    if (gradients[2].defined()) gt = gt+gradients[2];
    const auto gr = graph_solve(zero, p, lo, gy);
    auto h = -(gr*y).sum(-1);
    if (gradients[1].defined()) h = h+gradients[1];
    const auto adjusted = at::cat({h.slice(-1, 0, n-1)-(gt.slice(-1, 1, n)/previous_p)*next_t,
                                   h.slice(-1, n-1, n)}, -1);
    const auto gd = graph_solve(zero, one, -(lo/previous_p)*next_t, adjusted.unsqueeze(-1)).squeeze(-1);
    const auto gl = -(gr.slice(-2, 1, n)*y.slice(-2, 0, n-1)).sum(-1)-gd.slice(-1, 1, n)*next_t;
    const auto gu = (gt.slice(-1, 1, n)-gd.slice(-1, 1, n)*lo)/previous_p;
    return {gl, gd, gu, gr, at::Tensor()};
}

class ThomasArithmetic : public torch::autograd::Function<ThomasArithmetic> {
  public:
    static Tensors forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor rhs, bool checked) {
        RECORD_FUNCTION("qlt::thomas_arithmetic_forward", std::vector<c10::IValue>());
        const auto lo = readable(lower), di = readable(diagonal), up = readable(upper), r = readable(rhs);
        const auto n = di.size(-1), columns = r.size(-1), batches = di.numel()/n;
        auto pivot = at::empty(di.sizes(), di.options()), factor = at::empty(di.sizes(), di.options());
        auto forward_rhs = at::empty(r.sizes(), r.options()), result = at::empty(r.sizes(), r.options());
        const auto* l = lo.const_data_ptr<double>(); const auto* d = di.const_data_ptr<double>();
        const auto* u = up.const_data_ptr<double>(); const auto* b = r.const_data_ptr<double>();
        auto* p = pivot.mutable_data_ptr<double>(); auto* t = factor.mutable_data_ptr<double>();
        auto* y = forward_rhs.mutable_data_ptr<double>(); auto* x = result.mutable_data_ptr<double>();
        for (int64_t batch = 0; batch < batches; ++batch) {
            const auto start = batch*n, off = batch*(n-1), base = start*columns;
            p[start] = d[start]; t[start] = 0.;
            if (checked) TORCH_CHECK(std::isfinite(p[start]) && p[start] != 0.,
                                    "Thomas arithmetic requires finite nonzero elimination pivots");
            for (int64_t j = 0; j < columns; ++j) y[base+j] = b[base+j]/p[start];
            for (int64_t i = 1; i < n; ++i) {
                t[start+i] = u[off+i-1]/p[start+i-1];
                p[start+i] = d[start+i]-l[off+i-1]*t[start+i];
                if (checked) TORCH_CHECK(std::isfinite(p[start+i]) && p[start+i] != 0.,
                                        "Thomas arithmetic requires finite nonzero elimination pivots");
                for (int64_t j = 0; j < columns; ++j) {
                    const auto row = (start+i)*columns+j;
                    y[row] = (b[row]-l[off+i-1]*y[row-columns])/p[start+i];
                }
            }
            for (int64_t j = 0; j < columns; ++j) x[(start+n-1)*columns+j] = y[(start+n-1)*columns+j];
            for (int64_t i = n-2; i >= 0; --i)
                for (int64_t j = 0; j < columns; ++j) {
                    const auto row = (start+i)*columns+j;
                    x[row] = y[row]-t[start+i+1]*x[row+columns];
                }
        }
        // Intermediates are outputs so their saved versions retain this
        // operation's gradient edge for higher derivatives. Unused output
        // gradients stay undefined; ordinary pricing needs only result.
        ctx->set_materialize_grads(false);
        ctx->save_for_backward({lower, diagonal, upper, rhs, pivot, factor, forward_rhs, result});
        return {result, pivot, factor, forward_rhs};
    }

    static Tensors backward(torch::autograd::AutogradContext* ctx, Tensors gradients) {
        RECORD_FUNCTION("qlt::thomas_arithmetic_backward", std::vector<c10::IValue>());
        const auto saved = ctx->get_saved_variables();
        bool dual = false;
        for (const auto& g : gradients) if (g.defined() && g._fw_grad(0).defined()) dual = true;
        if (at::GradMode::is_enabled() || dual) {
            static const bool vectorized = performance_flag("QLT_THOMAS_VECTOR_BACKWARD");
            return vectorized ? vector_backward(saved, gradients) : tensor_backward(saved, gradients);
        }
        const auto lo = readable(saved[0]), di = saved[1], rhs = saved[3];
        const auto& pivot = saved[4]; const auto& factor = saved[5];
        const auto& forward_rhs = saved[6]; const auto& result = saved[7];
        const auto n = di.size(-1), columns = rhs.size(-1), batches = di.numel()/n;
        auto gl = at::empty(lo.sizes(), lo.options()), gd = at::empty(di.sizes(), di.options());
        auto gu = at::empty(lo.sizes(), lo.options()), gr = at::empty(rhs.sizes(), rhs.options());
        auto grad_y = readable(gradient_or_zero(gradients[0], rhs)).clone();
        auto grad_p = readable(gradient_or_zero(gradients[1], di)).clone();
        auto grad_t = readable(gradient_or_zero(gradients[2], di)).clone();
        const auto auxiliary_y = gradients[3].defined() ? readable(gradients[3]) : at::Tensor();
        const auto* l = lo.const_data_ptr<double>(); const auto* p = pivot.const_data_ptr<double>();
        const auto* t = factor.const_data_ptr<double>(); const auto* y = forward_rhs.const_data_ptr<double>();
        const auto* x = result.const_data_ptr<double>();
        auto* dl = gl.mutable_data_ptr<double>(); auto* dd = gd.mutable_data_ptr<double>();
        auto* du = gu.mutable_data_ptr<double>(); auto* db = gr.mutable_data_ptr<double>();
        auto* gy = grad_y.mutable_data_ptr<double>(); auto* gp = grad_p.mutable_data_ptr<double>();
        auto* gt = grad_t.mutable_data_ptr<double>();
        const auto* ay = auxiliary_y.defined() ? auxiliary_y.const_data_ptr<double>() : nullptr;
        for (int64_t batch = 0; batch < batches; ++batch) {
            const auto start = batch*n, off = batch*(n-1);
            for (int64_t i = 0; i < n-1; ++i)
                for (int64_t j = 0; j < columns; ++j) {
                    const auto row = (start+i)*columns+j;
                    gt[start+i+1] -= gy[row]*x[row+columns];
                    gy[row+columns] -= t[start+i+1]*gy[row];
                }
            if (ay)
                for (int64_t cell = start*columns; cell < (start+n)*columns; ++cell) gy[cell] += ay[cell];
            for (int64_t i = n-1; i >= 1; --i) {
                const auto row = start+i, k = off+i-1;
                double lower_gradient = 0.;
                for (int64_t j = 0; j < columns; ++j) {
                    const auto cell = row*columns+j;
                    const double gz = gy[cell]/p[row];
                    db[cell] = gz;
                    gp[row] -= gz*y[cell];
                    lower_gradient -= gz*y[cell-columns];
                    gy[cell-columns] -= l[k]*gz;
                }
                dd[row] = gp[row];
                dl[k] = lower_gradient-gp[row]*t[row];
                gt[row] -= gp[row]*l[k];
                du[k] = gt[row]/p[row-1];
                gp[row-1] -= du[k]*t[row];
            }
            for (int64_t j = 0; j < columns; ++j) {
                const auto cell = start*columns+j;
                db[cell] = gy[cell]/p[start];
                gp[start] -= db[cell]*y[cell];
            }
            dd[start] = gp[start];
        }
        // Mutations above affect only fresh gradient/scratch buffers.
        return {gl, gd, gu, gr, at::Tensor()};
    }
};

at::Tensor graph_solve(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up, const at::Tensor& r) {
    if (eligible(lo, d, up, r)) return ThomasArithmetic::apply(lo, d, up, r, false)[0];
    return at::stack(tensor_forward(lo, d, up, r, false).solution, -2);
}

Tensors state_probe(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up,
                 const at::Tensor& r, bool checked, bool native) {
    for (const auto& t : {lo, d, up, r}) {
        TORCH_CHECK(t.layout() == at::kStrided && t.scalar_type() == at::kDouble
                    && t.device() == d.device() && (t.is_cpu() || t.is_cuda()),
                    "Thomas arithmetic expects dense float64 inputs on a common CPU/CUDA device");
        if (checked) TORCH_CHECK(at::isfinite(t).all().item<bool>(), "Thomas arithmetic inputs must be finite");
    }
    TORCH_CHECK(lo.dim() >= 1 && d.dim() >= 1 && up.dim() >= 1 && r.dim() >= 2,
                "Thomas arithmetic expects bands (...,N), RHS (...,N,R)");
    const auto n = d.size(-1);
    TORCH_CHECK(n >= 2 && lo.size(-1) == n-1 && up.size(-1) == n-1 && r.size(-2) == n && r.size(-1) >= 1,
                "invalid Thomas arithmetic input sizes");
    const auto values = normalize(lo, d, up, r);
    Tensors result;
    if (native && eligible(lo, d, up, r)) result = ThomasArithmetic::apply(values[0], values[1], values[2], values[3], checked);
    else {
        const auto state = tensor_forward(values[0], values[1], values[2], values[3], checked);
        result = {at::stack(state.solution, -2), at::stack(state.pivot, -1),
                  at::stack(state.factor, -1), at::stack(state.forward_rhs, -2)};
    }
    if (checked) TORCH_CHECK(at::isfinite(result[0]).all().item<bool>(), "non-finite Thomas arithmetic solution");
    return result;
}

at::Tensor probe(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up,
                 const at::Tensor& r, bool checked, bool native) {
    return state_probe(lo, d, up, r, checked, native)[0];
}
}

bool try_thomas_arithmetic(const at::Tensor& lo, const at::Tensor& d,
    const at::Tensor& up, const at::Tensor& rhs, at::Tensor& output) {
    static const bool enabled = performance_flag("QLT_THOMAS_ARITHMETIC");
    if (!enabled || !eligible(lo, d, up, rhs)) return false;
    const auto values = normalize(lo, d, up, rhs);
    output = ThomasArithmetic::apply(values[0], values[1], values[2], values[3], false)[0];
    return true;
}

at::Tensor thomas_solve(const at::Tensor& lo, const at::Tensor& d,
    const at::Tensor& up, const at::Tensor& rhs) {
    return probe(lo, d, up, rhs, true, true);
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_thomas_arithmetic(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, bool checked=True, bool native=True) -> Tensor");
    m.def("_thomas_arithmetic_state(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, bool checked=True, bool native=True) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_thomas_arithmetic", TORCH_FN(quantlibtorch::probe));
    m.impl("_thomas_arithmetic_state", TORCH_FN(quantlibtorch::state_probe));
}
