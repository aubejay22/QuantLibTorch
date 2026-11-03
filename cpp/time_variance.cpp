// Elementary arithmetic from this project's TensorLocalVol::sampleGrid:
// sigma = (1-weight)*low + weight*high; variance = sigma*sigma.
// This is an explicit full-AD boundary, not an implicit first-order VJP.
#include <quantlibtorch/time_variance.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/ATen.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace quantlibtorch {
namespace {
bool enabled() {
    static const bool active = performance_flag("QLT_TIME_VARIANCE_ARITHMETIC");
    return active;
}

bool eligible(at::TensorList inputs) {
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

at::Tensor readable(const at::Tensor& input) {
    auto result = input.is_neg() ? input.resolve_neg() : input;
    if (result.is_conj()) result = result.resolve_conj();
    return result.contiguous();
}

class TimeVariance : public torch::autograd::Function<TimeVariance> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                              at::Tensor low, at::Tensor high, double weight) {
        RECORD_FUNCTION("qlt::time_variance_forward", std::vector<c10::IValue>());
        ctx->save_for_backward({low, high});
        ctx->saved_data["weight"] = weight;
        const auto a = readable(low), b = readable(high);
        const auto* l = a.const_data_ptr<double>();
        const auto* h = b.const_data_ptr<double>();
        auto result = at::empty(low.sizes(), low.options());
        auto* out = result.mutable_data_ptr<double>();
        for (int64_t i = 0; i < low.numel(); ++i) {
            // This translation unit disables contraction/reassociation so
            // the two products, sum, and square keep the original rounding.
            const double sigma = (1.-weight)*l[i] + weight*h[i];
            out[i] = sigma*sigma;
        }
        return result;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list gradients) {
        RECORD_FUNCTION("qlt::time_variance_backward", std::vector<c10::IValue>());
        const auto saved = ctx->get_saved_variables();
        const auto& low = saved[0]; const auto& high = saved[1]; const auto& g = gradients[0];
        const double weight = ctx->saved_data["weight"].toDouble();
        if (at::GradMode::is_enabled() || !eligible({low, high, g})) {
            RECORD_FUNCTION("qlt::time_variance_backward_graph", std::vector<c10::IValue>());
            // Reconstruct from ORIGINAL tensors for all higher derivatives.
            // Incoming forward-mode weights also require this functional path.
            const auto sigma = (1.-weight)*low + weight*high;
            const auto gs = g*sigma + g*sigma;
            return {gs*(1.-weight), gs*weight, at::Tensor()};
        }
        const auto a = readable(low), b = readable(high), incoming = readable(g);
        const auto* l = a.const_data_ptr<double>(); const auto* h = b.const_data_ptr<double>();
        const auto* weights = incoming.const_data_ptr<double>();
        auto dl = ctx->needs_input_grad(0) ? at::empty(low.sizes(), low.options()) : at::Tensor();
        auto dh = ctx->needs_input_grad(1) ? at::empty(high.sizes(), high.options()) : at::Tensor();
        auto* gl = dl.defined() ? dl.mutable_data_ptr<double>() : nullptr;
        auto* gh = dh.defined() ? dh.mutable_data_ptr<double>() : nullptr;
        for (int64_t i = 0; i < low.numel(); ++i) {
            const double sigma = (1.-weight)*l[i] + weight*h[i];
            const double gs = weights[i]*sigma + weights[i]*sigma;
            if (gl) gl[i] = gs*(1.-weight);
            if (gh) gh[i] = gs*weight;
        }
        return {dl, dh, at::Tensor()};
    }
};
}

at::Tensor time_variance(const at::Tensor& low, const at::Tensor& high, double weight, bool native) {
    TORCH_CHECK(low.dim() == 1 && high.dim() == 1 && low.sizes() == high.sizes()
                && low.device() == high.device() && low.scalar_type() == high.scalar_type(),
                "time variance requires equal one-dimensional rows on the same device and dtype");
    if (native && enabled() && eligible({low, high})) return TimeVariance::apply(low, high, weight);
    const auto sigma = (1.-weight)*low + weight*high;
    return sigma*sigma;
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_time_variance(Tensor low, Tensor high, float weight, bool native=True) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_time_variance", &quantlibtorch::time_variance);
}
