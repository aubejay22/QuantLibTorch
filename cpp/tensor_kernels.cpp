#include <quantlibtorch/tensor_kernels.hpp>
#include <quantlibtorch/array_kernels.hpp>
#include <quantlibtorch/adjoint.hpp>
#include <ATen/record_function.h>
#include <torch/library.h>
#include <memory>

namespace quantlibtorch {
namespace {
thread_local int kernel_mode = 0;

// Elementwise operations broadcast only the leading batch dimensions; the
// last two axes of values are always nodes and right-hand sides.
at::Tensor apply_rows(const at::Tensor& lower, const at::Tensor& diagonal,
                     const at::Tensor& upper, const at::Tensor& left_values,
                     const at::Tensor& values, const at::Tensor& right_values) {
    return left_values * lower.unsqueeze(-1) + values * diagonal.unsqueeze(-1)
         + right_values * upper.unsqueeze(-1);
}

void check_tensor(const at::Tensor& t, const at::Tensor& reference) {
    TORCH_CHECK(t.layout() == at::kStrided && t.scalar_type() == at::kDouble,
                "band kernels require dense strided float64 tensors");
    TORCH_CHECK(t.device() == reference.device() && (t.is_cpu() || t.is_cuda()),
                "band inputs must be on the same CPU or CUDA device");
}

at::Tensor tridiagonal_apply(const at::Tensor& lower, const at::Tensor& diagonal,
                            const at::Tensor& upper, const at::Tensor& rhs) {
    for (const auto& t : {lower, diagonal, upper, rhs}) check_tensor(t, diagonal);
    TORCH_CHECK(diagonal.dim() >= 1 && lower.dim() >= 1 && upper.dim() >= 1
                && rhs.dim() >= 2, "expected bands (..., N), rhs (..., N, R)");
    const auto n = diagonal.size(-1);
    TORCH_CHECK(n >= 2 && lower.size(-1) == n-1 && upper.size(-1) == n-1
                && rhs.size(-2) == n && rhs.size(-1) >= 1,
                "expected N >= 2, off-diagonals N-1, rhs (..., N, R) with R >= 1");
    // Zero padding retains compact tridiagonal semantics. No dense matrix,
    // contract loop, scalar extraction, or device synchronization is needed.
    auto lo = at::cat({at::zeros_like(lower.slice(-1, 0, 1)), lower}, -1);
    auto up = at::cat({upper, at::zeros_like(upper.slice(-1, 0, 1))}, -1);
    auto zero = at::zeros_like(rhs.slice(-2, 0, 1));
    return apply_rows(lo, diagonal, up,
        at::cat({zero, rhs.slice(-2, 0, n-1)}, -2), rhs,
        at::cat({rhs.slice(-2, 1, n), zero}, -2));
}

at::Tensor band_axpyb(const at::Tensor& a, const at::Tensor& x,
                      const at::Tensor& y, const at::Tensor& b) {
    for (const auto& t : {a, x, y, b}) check_tensor(t, x);
    TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 && a.dim() >= 1 && b.dim() >= 1,
                "expected bands (..., 3, N), scales (..., 1) or (..., N)");
    const auto n = x.size(-1);
    TORCH_CHECK(n >= 2 && x.size(-2) == 3 && y.size(-2) == 3 && y.size(-1) == n
                && (a.size(-1) == 1 || a.size(-1) == n)
                && (b.size(-1) == 1 || b.size(-1) == n), "invalid band or scale shape");
    auto zero = at::zeros_like(b);
    // Row-packed bands, including the endpoint entries used by QuantLib's
    // reflected neighbourhoods. b shifts only the central diagonal.
    return y + a.unsqueeze(-2)*x + at::stack({zero, b, zero}, -2);
}
} // namespace

int tensor_kernel_mode() { return kernel_mode; }
TensorKernelMode::TensorKernelMode(int mode) : previous_(kernel_mode) {
    TORCH_CHECK(mode >= 0 && mode <= 4, "invalid tensor kernel mode");
    kernel_mode = mode;
}
TensorKernelMode::~TensorKernelMode() { kernel_mode = previous_; }

at::Tensor pack_array(const QuantLib::Real* values, QuantLib::Size n) {
    RECORD_FUNCTION("qlt::pack_array", std::vector<c10::IValue>());
    if (packed_arrays_enabled() && n > 0) {
        const auto& owner = values[0].array_owner();
        const auto start = values[0].array_index();
        // Never reconnect views created across a no_grad boundary. Inference
        // tensors without gradients can still use the backing-storage path.
        bool packed = owner.defined() && (at::GradMode::is_enabled() || !owner.requires_grad());
        for (QuantLib::Size i = 1; packed && i < n; ++i)
            packed = values[i].array_owner().is_same(owner)
                  && values[i].array_index() == start + static_cast<int64_t>(i);
        if (packed)
            return start == 0 && static_cast<int64_t>(n) == owner.numel()
                ? owner : owner.slice(0, start, start + n);
    }
    bool passive = true;
    for (QuantLib::Size i = 0; i < n; ++i) passive = passive && values[i].is_passive();
    if (passive) {
        std::vector<double> constants;
        constants.reserve(n);
        for (QuantLib::Size i = 0; i < n; ++i) constants.push_back(values[i].value());
        return at::tensor(constants, at::TensorOptions().dtype(at::kDouble));
    }
    std::vector<at::Tensor> tensors;
    tensors.reserve(n);
    for (QuantLib::Size i = 0; i < n; ++i) tensors.push_back(values[i].tensor());
    return at::stack(tensors);
}

void unpack_array(const at::Tensor& values, QuantLib::Real* output) {
    RECORD_FUNCTION("qlt::unpack_array", std::vector<c10::IValue>());
    TORCH_CHECK(values.is_cpu() && values.scalar_type() == at::kDouble
                && values.layout() == at::kStrided && values.dim() == 1,
                "QuantLib arrays require one-dimensional dense CPU float64 tensors");
    const bool packed = packed_arrays_enabled() && (at::GradMode::is_enabled() || !values.requires_grad());
    // Full-AD Thomas elimination consumes individual entries many times. An
    // eager unbind gives it a smaller graph than repeated lazy select nodes.
    if (packed && lazy_arrays_enabled() && (adjoint_enabled() || tensor_kernel_mode() >= 3)) {
        for (int64_t i = 0; i < values.numel(); ++i) output[i] = TorchReal::array_element(values, i);
        return;
    }
    // Fallback preserves view creation semantics across no_grad boundaries.
    const auto scalars = values.unbind(0);
    for (size_t i = 0; i < scalars.size(); ++i)
        output[i] = packed ? TorchReal::array_element(scalars[i], values, i) : TorchReal(scalars[i]);
}

namespace {
template <class L, class R>
at::Tensor array_operation(const L& left, const R& right, char op) {
    switch (op) {
        case '+': return left + right;
        case '-': return left - right;
        case '*': return left * right;
        case '/': return left / right;
        default: TORCH_CHECK(false, "unknown array operation");
    }
}
}

bool array_binary(const QuantLib::Real* first, const QuantLib::Real* last,
                  const QuantLib::Real* second, QuantLib::Real* output, char op) {
    if (tensor_kernel_mode() < 2 || !array_kernels_enabled() || first == last) return false;
    const auto n = last - first;
    bool active = false;
    for (int64_t i = 0; !active && i < n; ++i)
        active = !first[i].is_passive() || !second[i].is_passive();
    if (!active) return false;
    RECORD_FUNCTION("qlt::array_binary", std::vector<c10::IValue>());
    auto result = array_operation(pack_array(first, n), pack_array(second, n), op);
    unpack_array(result, output);
    return true;
}

bool array_scalar(const QuantLib::Real* first, const QuantLib::Real* last,
                  QuantLib::Real* output, QuantLib::Real scalar, char op, bool reverse) {
    if (tensor_kernel_mode() < 2 || !array_kernels_enabled() || first == last) return false;
    const auto n = last - first;
    bool active = !scalar.is_passive();
    for (int64_t i = 0; !active && i < n; ++i) active = !first[i].is_passive();
    if (!active) return false;
    RECORD_FUNCTION("qlt::array_scalar", std::vector<c10::IValue>());
    const auto values = pack_array(first, n);
    auto result = scalar.is_passive()
        ? (reverse ? array_operation(scalar.value(), values, op) : array_operation(values, scalar.value(), op))
        : (reverse ? array_operation(scalar.tensor(), values, op) : array_operation(values, scalar.tensor(), op));
    unpack_array(result, output);
    return true;
}

QuantLib::Array vector_apply(const QuantLib::Array& rhs,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper, const QuantLib::Size* left,
    const QuantLib::Size* right) {
    const auto n = rhs.size();
    auto values = pack_array(rhs.begin(), n).unsqueeze(-1);
    const auto indices = [n](const QuantLib::Size* source) {
        // Lift immutable geometry constants into active transforms. The
        // vector factory writes through an empty tensor, which can instead
        // be a storage-less wrapper/fake tensor under those transforms.
        auto owner = std::make_shared<std::vector<int64_t>>(source, source+n);
        return at::lift_fresh(at::from_blob(owner->data(), {static_cast<int64_t>(n)},
            [owner](void*) {}, at::TensorOptions().dtype(at::kLong)));
    };
    auto result = apply_rows(pack_array(lower, n), pack_array(diagonal, n),
        pack_array(upper, n), values.index_select(0, indices(left)),
        values, values.index_select(0, indices(right)));
    QuantLib::Array output(n);
    unpack_array(result.squeeze(-1), output.begin());
    return output;
}

void vector_mult(QuantLib::Size n, const QuantLib::Array& scale,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper, QuantLib::Real* out_lower,
    QuantLib::Real* out_diagonal, QuantLib::Real* out_upper) {
    TORCH_CHECK(scale.size() == n, "inconsistent band scale size");
    const auto bands = at::stack({pack_array(lower, n), pack_array(diagonal, n), pack_array(upper, n)});
    const auto result = bands * pack_array(scale.begin(), n).unsqueeze(0);
    unpack_array(result[0], out_lower);
    unpack_array(result[1], out_diagonal);
    unpack_array(result[2], out_upper);
}

void vector_axpyb(QuantLib::Size n, const QuantLib::Array& a,
    const QuantLib::Real* xl, const QuantLib::Real* xd, const QuantLib::Real* xu,
    const QuantLib::Real* yl, const QuantLib::Real* yd, const QuantLib::Real* yu,
    const QuantLib::Array& b, QuantLib::Real* lower,
    QuantLib::Real* diagonal, QuantLib::Real* upper) {
    TORCH_CHECK((a.empty() || a.size() == 1 || a.size() == n)
                && (b.empty() || b.size() == 1 || b.size() == n), "invalid axpyb scale size");
    // Pack every source before rebinding outputs: upstream allows self-aliases.
    auto result = at::stack({pack_array(yl, n), pack_array(yd, n), pack_array(yu, n)});
    if (!a.empty()) {
        auto x = at::stack({pack_array(xl, n), pack_array(xd, n), pack_array(xu, n)});
        result = result + pack_array(a.begin(), a.size()).unsqueeze(0)*x;
    }
    if (!b.empty()) {
        auto shift = pack_array(b.begin(), b.size());
        auto zero = at::zeros_like(shift);
        result = result + at::stack({zero, shift, zero});
    }
    unpack_array(result[0], lower);
    unpack_array(result[1], diagonal);
    unpack_array(result[2], upper);
}
} // namespace quantlibtorch

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("tridiagonal_apply(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs) -> Tensor");
    m.def("band_axpyb(Tensor a, Tensor x, Tensor y, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("tridiagonal_apply", TORCH_FN(quantlibtorch::tridiagonal_apply));
    m.impl("band_axpyb", TORCH_FN(quantlibtorch::band_axpyb));
}
