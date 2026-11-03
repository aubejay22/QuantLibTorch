#include <quantlibtorch/structured_ops.hpp>
#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/tridiagonaloperator.hpp>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <limits>
#include <vector>

namespace quantlibtorch {
namespace {

at::Tensor readable(const at::Tensor& tensor) {
    auto result = tensor.is_neg() ? tensor.resolve_neg() : tensor;
    if (result.is_conj()) result = result.resolve_conj();
    return result.contiguous();
}

// This passive conversion belongs only inside the explicit numerical boundary.
// All original coefficient dependence enters through the custom Function inputs.
QuantLib::Array passive_array(const at::Tensor& tensor) {
    const auto values = readable(tensor);
    const auto* data = values.const_data_ptr<double>();
    QuantLib::Array result(values.numel());
    for (int64_t i = 0; i < values.numel(); ++i) result[i] = TorchReal(data[i]);
    return result;
}

at::Tensor owned_indices(const QuantLib::Size* source, int64_t n) {
    auto result = at::empty({n}, at::TensorOptions().dtype(at::kLong));
    auto* data = result.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) data[i] = source ? static_cast<int64_t>(source[i]) : i;
    return result;
}

class PassiveDirectionalBand final : public QuantLib::TripleBandLinearOp {
  public:
    explicit PassiveDirectionalBand(const QuantLib::TripleBandLinearOp& op)
    : QuantLib::TripleBandLinearOp(op) {
        for (QuantLib::Size i = 0; i < mesher_->layout()->size(); ++i) {
            lower_[i] = TorchReal(lower_[i].value());
            diag_[i] = TorchReal(diag_[i].value());
            upper_[i] = TorchReal(upper_[i].value());
        }
    }
};

class DirectionalSolveAdjoint : public torch::autograd::Function<DirectionalSolveAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor lower, at::Tensor diagonal, at::Tensor upper, at::Tensor rhs,
        at::Tensor permutation, const QuantLib::TripleBandLinearOp* op,
        double a, double b) {
        RECORD_FUNCTION("qlt::directional_adjoint_forward", std::vector<c10::IValue>());
        AdjointMode adjoint_guard(false);
        TensorKernelMode kernel_guard(0);
        // Use the pinned upstream scaled recurrence, permutation and pivot rules.
        // The copied operator exists only during this synchronous forward call.
        const auto values = PassiveDirectionalBand(*op).solve_splitting(
            passive_array(rhs), TorchReal(a), TorchReal(b));
        const auto result = pack_array(values.begin(), values.size());
        TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite directional solution");
        ctx->save_for_backward({lower, diagonal, upper, result, permutation});
        return result;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list gradients) {
        RECORD_FUNCTION("qlt::directional_adjoint_backward", std::vector<c10::IValue>());
        TORCH_CHECK(!at::GradMode::is_enabled(),
            "QuantLib directional adjoint supports first derivatives only; use full AD for higher derivatives");
        TORCH_CHECK(!gradients[0]._fw_grad(0).defined(),
            "QuantLib directional adjoint does not support forward-mode tangents");
        AdjointMode adjoint_guard(false);
        TensorKernelMode kernel_guard(0);
        const auto saved = ctx->get_saved_variables();
        const auto& permutation = saved[4];
        const auto x = saved[3].index_select(0, permutation);
        const auto seed = gradients[0].index_select(0, permutation);
        QuantLib::TridiagonalOperator transpose(
            passive_array(saved[2]), passive_array(saved[1]), passive_array(saved[0]));
        const auto values = transpose.solveFor(passive_array(seed));
        const auto lambda = pack_array(values.begin(), values.size());
        TORCH_CHECK(at::isfinite(lambda).all().item<bool>(), "non-finite directional adjoint");
        const auto n = x.numel();
        // Coefficient gradients use elimination order; the graph-connected
        // gathers outside Function.forward scatter them to original coefficients.
        const auto rhs_gradient = at::zeros_like(lambda).index_add(0, permutation, lambda);
        return {-lambda.slice(0, 1, n)*x.slice(0, 0, n-1), -lambda*x,
                -lambda.slice(0, 0, n-1)*x.slice(0, 1, n), rhs_gradient,
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

void check_stencil(const at::Tensor& coefficients, const at::Tensor& indices,
                   const at::Tensor& state) {
    TORCH_CHECK(coefficients.layout() == at::kStrided && state.layout() == at::kStrided
                && indices.layout() == at::kStrided,
                "stencil inputs must be dense tensors");
    TORCH_CHECK(coefficients.scalar_type() == at::kDouble && state.scalar_type() == at::kDouble
                && indices.scalar_type() == at::kLong, "stencil values must be float64 and indices int64");
    TORCH_CHECK(coefficients.device() == state.device() && indices.device() == state.device(),
                "stencil tensors must use the same device");
    TORCH_CHECK(state.dim() == 1 && coefficients.dim() == 2 && indices.dim() == 2
                && coefficients.sizes() == indices.sizes() && coefficients.size(0) >= 1
                && coefficients.size(1) == state.numel() && state.numel() >= 1,
                "stencil shapes must be coefficients/indices [K,N] and state [N]");
    // index_select/index_add check bounds. No host read of coefficient values
    // is needed, so this elementary boundary keeps functional tensor semantics.
}

at::Tensor stencil_apply(const at::Tensor& coefficients, const at::Tensor& indices,
                         const at::Tensor& state) {
    auto result = coefficients[0]*state.index_select(0, indices[0]);
    for (int64_t k = 1; k < coefficients.size(0); ++k)
        result = result+coefficients[k]*state.index_select(0, indices[k]);
    return result;
}

} // namespace

QuantLib::Array directional_adjoint_split(const QuantLib::TripleBandLinearOp& op,
    const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper, const QuantLib::Size* reverse_index) {
    const auto n = static_cast<int64_t>(rhs.size());
    TORCH_CHECK(n >= 2, "directional splitting adjoint requires at least two nodes");
    const auto permutation = owned_indices(reverse_index, n);
    const auto original_lower = pack_array(lower, n), original_diagonal = pack_array(diagonal, n);
    const auto original_upper = pack_array(upper, n), original_rhs = pack_array(rhs.begin(), n);
    const auto scale = a.tensor(), shift = b.tensor();
    for (const auto& value : {original_lower, original_diagonal, original_upper, original_rhs, scale, shift})
        TORCH_CHECK(!value._fw_grad(0).defined(),
            "QuantLib directional adjoint does not support forward-mode tangents");
    // The upstream recurrence connects successive rows in reverseIndex_ order.
    // Only the GLOBAL first lower / last upper are omitted. Independent lines
    // arise from their operator coefficients being zero, not additional masking.
    const auto lo = scale*original_lower.index_select(0, permutation).slice(0, 1, n);
    const auto di = scale*original_diagonal.index_select(0, permutation)+shift;
    const auto up = scale*original_upper.index_select(0, permutation).slice(0, 0, n-1);
    const auto result = DirectionalSolveAdjoint::apply(lo, di, up, original_rhs,
                                                     permutation, &op, a.value(), b.value());
    QuantLib::Array output(n);
    unpack_array(result, output.begin());
    return output;
}

at::Tensor structured_stencil_apply(const at::Tensor& coefficients,
    const at::Tensor& indices, const at::Tensor& state) {
    RECORD_FUNCTION("qlt::structured_stencil_apply", std::vector<c10::IValue>());
    check_stencil(coefficients, indices, state);
    return stencil_apply(coefficients, indices, state);
}

at::Tensor structured_stencil_transpose(const at::Tensor& coefficients,
    const at::Tensor& indices, const at::Tensor& seed) {
    RECORD_FUNCTION("qlt::structured_stencil_transpose", std::vector<c10::IValue>());
    check_stencil(coefficients, indices, seed);
    return at::zeros_like(seed).index_add(0, indices.reshape({-1}),
                                         (coefficients*seed.unsqueeze(0)).reshape({-1}));
}

std::tuple<at::Tensor, at::Tensor> structured_stencil_vjp(
    const at::Tensor& coefficients, const at::Tensor& indices,
    const at::Tensor& state, const at::Tensor& seed) {
    check_stencil(coefficients, indices, state);
    check_stencil(coefficients, indices, seed);
    const auto sampled = state.index_select(0, indices.reshape({-1})).reshape(indices.sizes());
    return {sampled*seed.unsqueeze(0), structured_stencil_transpose(coefficients, indices, seed)};
}

QuantLib::Array nine_point_apply(const QuantLib::Array& rhs,
    const std::array<const QuantLib::Real*, 9>& coefficients,
    const std::array<const QuantLib::Size*, 9>& indices) {
    const auto n = static_cast<int64_t>(rhs.size());
    std::vector<at::Tensor> values, neighbors;
    values.reserve(9); neighbors.reserve(9);
    for (size_t k = 0; k < 9; ++k) {
        values.push_back(pack_array(coefficients[k], n));
        neighbors.push_back(owned_indices(indices[k], n));
    }
    const auto result = structured_stencil_apply(at::stack(values), at::stack(neighbors),
                                                 pack_array(rhs.begin(), n));
    QuantLib::Array output(n);
    unpack_array(result, output.begin());
    return output;
}

namespace {
class DirectionalProbeBand final : public QuantLib::TripleBandLinearOp {
  public:
    DirectionalProbeBand(const at::Tensor& bands,
        const QuantLib::ext::shared_ptr<QuantLib::FdmMesher>& mesher, QuantLib::Size direction)
    : QuantLib::TripleBandLinearOp(direction, mesher) {
        unpack_array(bands[0], lower_.get());
        unpack_array(bands[1], diag_.get());
        unpack_array(bands[2], upper_.get());
    }
    QuantLib::Array solve(const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b, bool adjoint) const {
        if (adjoint) return directional_adjoint_split(*this, rhs, a, b, lower_.get(),
                                                       diag_.get(), upper_.get(), reverseIndex_.get());
        return solve_splitting(rhs, a, b);
    }
};

at::Tensor directional_solve_probe(const at::Tensor& bands, const at::Tensor& rhs,
    const at::Tensor& scales, const c10::List<int64_t>& dimensions, int64_t direction, bool adjoint) {
    for (const auto& t : {bands, rhs, scales}) {
        TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided,
                    "directional probe inputs must be dense CPU float64 tensors");
        TORCH_CHECK(at::isfinite(t).all().item<bool>(), "directional probe inputs must be finite");
    }
    TORCH_CHECK(!dimensions.empty() && direction >= 0 && direction < static_cast<int64_t>(dimensions.size()),
                "directional probe direction is outside the dimensions");
    int64_t n = 1;
    std::vector<QuantLib::ext::shared_ptr<QuantLib::Fdm1dMesher>> axes;
    for (const auto size : dimensions) {
        TORCH_CHECK(size >= 2 && n <= (std::numeric_limits<int64_t>::max)()/size,
                    "directional probe dimensions must be at least two without overflow");
        n *= size;
        axes.push_back(QuantLib::ext::make_shared<QuantLib::Uniform1dMesher>(0., 1., size));
    }
    TORCH_CHECK(bands.dim() == 2 && bands.size(0) == 3 && bands.size(1) == n
                && rhs.dim() == 1 && rhs.numel() == n && scales.dim() == 1 && scales.numel() == 2,
                "directional probe expects bands [3,N], rhs [N], scales [2]");
    AdjointMode adjoint_guard(false);
    TensorKernelMode kernel_guard(0);
    const auto mesh = QuantLib::ext::make_shared<QuantLib::FdmMesherComposite>(axes);
    const DirectionalProbeBand op(bands, mesh, direction);
    QuantLib::Array state(n);
    unpack_array(rhs, state.begin());
    const auto output = op.solve(state, TorchReal(scales[0]), TorchReal(scales[1]), adjoint);
    const auto result = pack_array(output.begin(), output.size());
    TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite directional solution");
    return result;
}
} // namespace
} // namespace quantlibtorch

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_directional_solve_probe(Tensor bands, Tensor rhs, Tensor scales, int[] dimensions, int direction, bool adjoint=True) -> Tensor");
    m.def("_stencil_apply(Tensor coefficients, Tensor indices, Tensor state) -> Tensor");
    m.def("_stencil_transpose(Tensor coefficients, Tensor indices, Tensor seed) -> Tensor");
    m.def("_stencil_vjp(Tensor coefficients, Tensor indices, Tensor state, Tensor seed) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_directional_solve_probe", TORCH_FN(quantlibtorch::directional_solve_probe));
    m.impl("_stencil_apply", TORCH_FN(quantlibtorch::structured_stencil_apply));
    m.impl("_stencil_transpose", TORCH_FN(quantlibtorch::structured_stencil_transpose));
    m.impl("_stencil_vjp", TORCH_FN(quantlibtorch::structured_stencil_vjp));
}
