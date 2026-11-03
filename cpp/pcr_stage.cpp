#include <quantlibtorch/pcr_stage.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace quantlibtorch {
namespace {
using Tensors = std::vector<at::Tensor>;

at::Tensor readable(const at::Tensor& t) {
    auto values = t.is_neg() ? t.resolve_neg() : t;
    if (values.is_conj()) values = values.resolve_conj();
    return values.contiguous();
}

at::Tensor roll(const at::Tensor& t, int64_t distance, int64_t axis=-1) {
    return at::roll(t, {distance}, {axis});
}

// Expansion happens OUTSIDE the custom operation, so ordinary autograd sums
// coefficient gradients across broadcast axes. The common batch shape also
// includes RHS batches; shared bands may be repeated in the stage outputs.
// The common single-contract and already-matched batch paths need no views.
Tensors normalize(const at::Tensor& lo, const at::Tensor& d,
                  const at::Tensor& up, const at::Tensor& r) {
    if (lo.sizes() == d.sizes() && up.sizes() == d.sizes()
        && r.dim() == d.dim()+1 && r.sizes().slice(0, r.dim()-1) == d.sizes())
        return {lo, d, up, r};
    const auto bands = at::broadcast_tensors({lo, d, up, r.select(-1, 0)});
    auto sizes = bands[1].sizes().vec();
    sizes.push_back(r.size(-1));
    return {bands[0], bands[1], bands[2], r.expand(sizes)};
}

Tensors tensor_stage(const at::Tensor& lo, const at::Tensor& d,
    const at::Tensor& up, const at::Tensor& r, int64_t distance) {
    const auto alpha = -lo / roll(d, distance);
    const auto beta = -up / roll(d, -distance);
    return {alpha*roll(lo, distance),
            d + alpha*roll(up, distance) + beta*roll(lo, -distance),
            beta*roll(up, -distance),
            r + alpha.unsqueeze(-1)*roll(r, distance, -2) + beta.unsqueeze(-1)*roll(r, -distance, -2)};
}

class PCRStage : public torch::autograd::Function<PCRStage> {
  public:
    static Tensors forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor rhs, int64_t distance) {
        RECORD_FUNCTION("qlt::pcr_stage_forward", std::vector<c10::IValue>());
        const auto lo = readable(lower), d = readable(diagonal), up = readable(upper), r = readable(rhs);
        const auto n = d.size(-1), columns = r.size(-1);
        auto nl = at::empty(d.sizes(), d.options()), nd = at::empty(d.sizes(), d.options());
        auto nu = at::empty(d.sizes(), d.options()), nr = at::empty(r.sizes(), r.options());
        const auto* l = lo.const_data_ptr<double>();
        const auto* v = d.const_data_ptr<double>();
        const auto* u = up.const_data_ptr<double>();
        const auto* b = r.const_data_ptr<double>();
        auto* out_l = nl.mutable_data_ptr<double>();
        auto* out_d = nd.mutable_data_ptr<double>();
        auto* out_u = nu.mutable_data_ptr<double>();
        auto* out_r = nr.mutable_data_ptr<double>();
        for (int64_t offset = 0; offset < d.numel(); offset += n) {
          for (int64_t row = 0; row < n; ++row) {
            const auto i = offset+row;
            const auto left = offset+(row >= distance ? row-distance : row+n-distance);
            const auto right = offset+(row+distance < n ? row+distance : row+distance-n);
            const auto alpha = -l[i]/v[left], beta = -u[i]/v[right];
            out_l[i] = alpha*l[left];
            out_d[i] = (v[i]+alpha*u[left])+beta*l[right];
            out_u[i] = beta*u[right];
            for (int64_t j = 0; j < columns; ++j)
                out_r[i*columns+j] = (b[i*columns+j]+alpha*b[left*columns+j])+beta*b[right*columns+j];
          }
        }
        // Original graph-connected operands, not the readable native views.
        ctx->save_for_backward({lower, diagonal, upper, rhs});
        ctx->saved_data["distance"] = distance;
        return {nl, nd, nu, nr};
    }

    static Tensors backward(torch::autograd::AutogradContext* ctx, Tensors weights) {
        RECORD_FUNCTION("qlt::pcr_stage_backward", std::vector<c10::IValue>());
        const auto input = ctx->get_saved_variables();
        const auto distance = ctx->saved_data["distance"].toInt();
        bool dual = false;
        for (size_t k = 0; k < 4; ++k) {
            if (!weights[k].defined()) weights[k] = at::zeros_like(input[k]);
            dual = dual || weights[k]._fw_grad(0).defined();
        }

        if (at::GradMode::is_enabled() || dual) {
            RECORD_FUNCTION("qlt::pcr_stage_backward_graph", std::vector<c10::IValue>());
            // Differentiate the elementary recurrence itself. Recomputing
            // alpha/beta from SAVED INPUTS keeps second and higher derivatives
            // connected, including derivatives with respect to incoming VJPs.
            const auto& lo = input[0]; const auto& d = input[1];
            const auto& up = input[2]; const auto& r = input[3];
            const auto& wl = weights[0]; const auto& wd = weights[1];
            const auto& wu = weights[2]; const auto& wr = weights[3];
            const auto dl = roll(d, distance), dr = roll(d, -distance);
            const auto alpha = -lo/dl, beta = -up/dr;
            const auto ga = wl*roll(lo, distance) + wd*roll(up, distance)
                + (wr*roll(r, distance, -2)).sum(-1);
            const auto gb = wu*roll(up, -distance) + wd*roll(lo, -distance)
                + (wr*roll(r, -distance, -2)).sum(-1);
            const auto gl = -ga/dl + roll(alpha*wl, -distance) + roll(beta*wd, distance);
            const auto gd = wd + roll(-ga*(alpha/dl), -distance) + roll(-gb*(beta/dr), distance);
            const auto gu = -gb/dr + roll(alpha*wd, -distance) + roll(beta*wu, distance);
            const auto gr = wr + roll(alpha.unsqueeze(-1)*wr, -distance, -2)
                               + roll(beta.unsqueeze(-1)*wr, distance, -2);
            return {gl, gd, gu, gr, at::Tensor()};
        }

        const auto lo = readable(input[0]), d = readable(input[1]), up = readable(input[2]), r = readable(input[3]);
        for (auto& weight : weights) weight = readable(weight);
        const auto n = d.size(-1), columns = r.size(-1);
        auto gl = at::zeros(d.sizes(), d.options()), gd = at::zeros(d.sizes(), d.options());
        auto gu = at::zeros(d.sizes(), d.options()), gr = at::zeros(r.sizes(), r.options());
        const auto* l = lo.const_data_ptr<double>();
        const auto* v = d.const_data_ptr<double>();
        const auto* u = up.const_data_ptr<double>();
        const auto* b = r.const_data_ptr<double>();
        const auto* wl = weights[0].const_data_ptr<double>();
        const auto* wd = weights[1].const_data_ptr<double>();
        const auto* wu = weights[2].const_data_ptr<double>();
        const auto* wr = weights[3].const_data_ptr<double>();
        auto* out_l = gl.mutable_data_ptr<double>();
        auto* out_d = gd.mutable_data_ptr<double>();
        auto* out_u = gu.mutable_data_ptr<double>();
        auto* out_r = gr.mutable_data_ptr<double>();
        for (int64_t offset = 0; offset < d.numel(); offset += n) {
          for (int64_t row = 0; row < n; ++row) {
            const auto i = offset+row;
            const auto left = offset+(row >= distance ? row-distance : row+n-distance);
            const auto right = offset+(row+distance < n ? row+distance : row+distance-n);
            const auto alpha = -l[i]/v[left], beta = -u[i]/v[right];
            double ga = wl[i]*l[left]+wd[i]*u[left];
            double gb = wu[i]*u[right]+wd[i]*l[right];
            for (int64_t j = 0; j < columns; ++j) {
                const auto w = wr[i*columns+j];
                ga += w*b[left*columns+j];
                gb += w*b[right*columns+j];
                out_r[i*columns+j] += w;
                out_r[left*columns+j] += alpha*w;
                out_r[right*columns+j] += beta*w;
            }
            out_l[i] += -ga/v[left];
            out_l[left] += alpha*wl[i];
            out_l[right] += beta*wd[i];
            out_d[i] += wd[i];
            // Use the same quotient-rule form as tensor division, avoiding
            // a squared denominator that could overflow or underflow.
            out_d[left] += -ga*(alpha/v[left]);
            out_d[right] += -gb*(beta/v[right]);
            out_u[i] += -gb/v[right];
            out_u[left] += alpha*wd[i];
            out_u[right] += beta*wu[i];
          }
        }
        // Only fresh output buffers were written. Coefficient tensors and
        // incoming weights retain their original values and version counters.
        return {gl, gd, gu, gr, at::Tensor()};
    }
};

bool eligible(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up, const at::Tensor& r) {
    if (!d.is_cpu()) return false;
    // C++ custom Functions do not implement torch.func's transform protocol.
    // Preserve the original tensor decomposition under transforms and wrapper
    // tensors (including fake tensors used by graph tracing).
    if (at::functorch::functorchTLSAccessor()
        && !at::functorch::getDynamicLayerStack().empty()) return false;
    for (const auto& t : {lo, d, up, r}) {
        const auto keys = t.key_set();
        if (keys.has(c10::DispatchKey::Python) || keys.has(c10::DispatchKey::Functionalize)
            || keys.has(c10::DispatchKey::FuncTorchBatched) || keys.has(c10::DispatchKey::FuncTorchGradWrapper)
            || t._fw_grad(0).defined()) return false;
    }
    return true;
}

bool prefer_tensor_single_rhs(const at::Tensor& lo, const at::Tensor& d,
                             const at::Tensor& up, const at::Tensor& r) {
    if (r.size(-1) != 1) return false;
    // Measured single-RHS crossover: native wins at ~2K/8K/16K/32K rows,
    // but loses at ~66K. Keep a conservative 32K-row bound for this path.
    constexpr int64_t row_limit = 32768;
    if (lo.sizes() == d.sizes() && up.sizes() == d.sizes()
        && r.dim() == d.dim()+1 && r.sizes().slice(0, r.dim()-1) == d.sizes()) return d.numel() > row_limit;
    const auto band_shape = at::infer_size(at::infer_size(lo.sizes(), d.sizes()), up.sizes());
    const auto full_shape = at::infer_size(band_shape, r.sizes().slice(0, r.dim()-1));
    int64_t rows = 1;
    for (const auto extent : full_shape) {
        if (extent == 0) return false;
        if (rows > row_limit/extent) return true;
        rows *= extent;
    }
    for (int64_t i = full_shape.size()-1, j = band_shape.size()-1; i >= 0; --i, --j)
        if (full_shape[i] > (j >= 0 ? band_shape[j] : 1)) return true;
    return false;
}

Tensors stage_probe(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up,
                    const at::Tensor& r, int64_t distance, bool native) {
    for (const auto& t : {lo, d, up, r})
        TORCH_CHECK(t.layout() == at::kStrided && t.scalar_type() == at::kDouble
                    && t.device() == d.device() && (t.is_cpu() || t.is_cuda()),
                    "stage inputs must be dense float64 on a common CPU/CUDA device");
    TORCH_CHECK(lo.dim() >= 1 && d.dim() >= 1 && up.dim() >= 1 && r.dim() >= 2,
                "stage probe expects bands (...,N), rhs (...,N,R)");
    const auto n = d.size(-1);
    TORCH_CHECK(n >= 2 && lo.size(-1) == n && up.size(-1) == n && r.size(-2) == n && r.size(-1) >= 1
                && distance >= 1 && distance < n, "invalid stage sizes or distance");
    const auto values = normalize(lo, d, up, r);
    if (native && eligible(lo, d, up, r))
        return PCRStage::apply(values[0], values[1], values[2], values[3], distance);
    return tensor_stage(values[0], values[1], values[2], values[3], distance);
}
}

bool try_native_pcr_stage(const at::Tensor& lo, const at::Tensor& d, const at::Tensor& up,
                         const at::Tensor& r, int64_t distance, Tensors& output) {
    static const bool enabled = performance_flag("QLT_PCR_NATIVE_STAGE");
    if (!enabled || !eligible(lo, d, up, r)) return false;
    // Single-RHS shared or large batches benefit from the tensor path; keep
    // the measured dispatch and avoid expanding shared coefficient batches.
    if (prefer_tensor_single_rhs(lo, d, up, r)) return false;
    const auto values = normalize(lo, d, up, r);
    output = PCRStage::apply(values[0], values[1], values[2], values[3], distance);
    return true;
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_pcr_stage(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, int distance, bool native=True) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_pcr_stage", TORCH_FN(quantlibtorch::stage_probe));
}
