// Verification/benchmark entry points exercising the actual upstream hooks.
#include <quantlibtorch/tensor_kernels.hpp>
#include <quantlibtorch/adjoint.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <torch/library.h>

namespace {
class TestBand : public QuantLib::TripleBandLinearOp {
  public:
    explicit TestBand(const at::Tensor& bands)
    : TripleBandLinearOp(0, QuantLib::ext::make_shared<QuantLib::FdmMesherComposite>(
        QuantLib::ext::make_shared<QuantLib::Uniform1dMesher>(0., 1., bands.size(1)))) {
        for (int64_t i = 0; i < bands.size(1); ++i) {
            lower_[i] = quantlibtorch::TorchReal(bands[0][i]);
            diag_[i] = quantlibtorch::TorchReal(bands[1][i]);
            upper_[i] = quantlibtorch::TorchReal(bands[2][i]);
        }
    }
    at::Tensor bands() const {
        const auto n = mesher_->layout()->size();
        return at::stack({quantlibtorch::pack_array(lower_.get(), n),
            quantlibtorch::pack_array(diag_.get(), n), quantlibtorch::pack_array(upper_.get(), n)});
    }
};

void check(const at::Tensor& t) {
    TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided,
                "hook verification requires dense CPU float64 tensors");
}
void check_bands(const at::Tensor& x) {
    check(x);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 3 && x.size(1) >= 2, "expected (3, N) bands");
}

at::Tensor hook_apply(const at::Tensor& bands, const at::Tensor& rhs, int64_t mode, bool adjoint) {
    check_bands(bands);
    check(rhs);
    TORCH_CHECK(rhs.dim() == 1 && rhs.size(0) == bands.size(1), "expected (N,) rhs");
    quantlibtorch::TensorKernelMode guard(static_cast<int>(mode));
    quantlibtorch::AdjointMode adjoint_guard(adjoint);
    TestBand op(bands);
    QuantLib::Array values(rhs.numel());
    quantlibtorch::unpack_array(rhs, values.begin());
    auto result = op.apply(values);
    return quantlibtorch::pack_array(result.begin(), result.size());
}

at::Tensor hook_assembly(const at::Tensor& a, const at::Tensor& x,
    const at::Tensor& y, const at::Tensor& b, const at::Tensor& scale,
    int64_t mode, int64_t alias) {
    check_bands(x);
    check_bands(y);
    TORCH_CHECK(x.sizes() == y.sizes(), "bands must have same shape");
    for (const auto& t : {a, b, scale}) check(t);
    const auto n = x.size(1);
    TORCH_CHECK(a.dim() == 1 && b.dim() == 1 && scale.dim() == 1 && scale.numel() == n
                && (a.numel() == 0 || a.numel() == 1 || a.numel() == n)
                && (b.numel() == 0 || b.numel() == 1 || b.numel() == n), "invalid scales");
    quantlibtorch::TensorKernelMode guard(static_cast<int>(mode));
    TestBand xx(x), yy(y), out(x);
    QuantLib::Array aa(a.numel()), bb(b.numel()), ss(n);
    quantlibtorch::unpack_array(a, aa.begin());
    quantlibtorch::unpack_array(b, bb.begin());
    quantlibtorch::unpack_array(scale, ss.begin());
    // Exercise mult's returned operator and each supported self-alias pattern.
    yy.QuantLib::TripleBandLinearOp::operator=(yy.mult(ss));
    if (alias == 1) { xx.axpyb(aa, xx, yy, bb); return xx.bands(); }
    if (alias == 2) { yy.axpyb(aa, xx, yy, bb); return yy.bands(); }
    out.axpyb(aa, xx, yy, bb);
    return out.bands();
}

at::Tensor diffusion_basis_series(const at::Tensor& variances, const at::Tensor& dx, const at::Tensor& dxx,
    const at::Tensor& rate, const at::Tensor& yield, bool adjoint, int64_t mode) {
    check_bands(dx); check_bands(dxx);
    for (const auto& t : {variances, rate, yield}) check(t);
    const auto n = dx.size(1);
    TORCH_CHECK(dx.sizes() == dxx.sizes() && variances.dim() == 2 && variances.size(1) == n
                && variances.size(0) >= 1 && rate.dim() == 0 && yield.dim() == 0, "invalid diffusion series shapes");
    quantlibtorch::TensorKernelMode kernel_guard(static_cast<int>(mode));
    quantlibtorch::AdjointMode adjoint_guard(adjoint);
    const TestBand first(dx), second(dxx);
    TestBand output(dx);
    quantlibtorch::DiffusionBasisCache cache;
    const auto r = quantlibtorch::TorchReal(rate), q = quantlibtorch::TorchReal(yield);
    std::vector<at::Tensor> results;
    for (int64_t i = 0; i < variances.size(0); ++i) {
        QuantLib::Array values(n);
        quantlibtorch::unpack_array(variances[i], values.begin());
        if (!quantlibtorch::try_diffusion_assembly(output, first, second, values, r, q, &cache))
            output.axpyb(r-q-0.5*values, first, second.mult(0.5*values), QuantLib::Array(1, -r));
        results.push_back(output.bands());
    }
    return at::stack(results);
}

at::Tensor hook_diffusion(const at::Tensor& variance, const at::Tensor& dx, const at::Tensor& dxx,
    const at::Tensor& rate, const at::Tensor& yield, bool adjoint, int64_t alias, int64_t mode) {
    check_bands(dx);
    check_bands(dxx);
    for (const auto& t : {variance, rate, yield}) check(t);
    const auto n = dx.size(1);
    TORCH_CHECK(dx.sizes() == dxx.sizes() && variance.dim() == 1 && variance.numel() == n
                && rate.dim() == 0 && yield.dim() == 0 && alias >= 0 && alias <= 2, "invalid diffusion input shapes");
    // Mode zero forces the original mult/axpyb reference independently of
    // process-wide full/adjoint arithmetic optimization switches.
    quantlibtorch::TensorKernelMode kernel_guard(static_cast<int>(mode));
    quantlibtorch::AdjointMode adjoint_guard(adjoint);
    TestBand first(dx), second(dxx), out(dx);
    QuantLib::Array values(n);
    quantlibtorch::unpack_array(variance, values.begin());
    const auto r = quantlibtorch::TorchReal(rate), q = quantlibtorch::TorchReal(yield);
    auto& result = alias == 1 ? first : (alias == 2 ? second : out);
    if (!quantlibtorch::try_diffusion_assembly(result, first, second, values, r, q))
        result.axpyb(r-q-0.5*values, first, second.mult(0.5*values), QuantLib::Array(1, -r));
    return result.bands();
}

at::Tensor roundtrip(const at::Tensor& input) {
    quantlibtorch::AdjointMode lazy_view_scope(true);
    check(input);
    TORCH_CHECK(input.dim() == 1 && input.numel() >= 2, "expected (N,) values");
    QuantLib::Array values(input.numel());
    quantlibtorch::unpack_array(input, values.begin());
    return quantlibtorch::pack_array(values.begin(), values.size());
}

std::vector<at::Tensor> array_views(const at::Tensor& input, const at::Tensor& other) {
    quantlibtorch::AdjointMode lazy_view_scope(true);
    check(input);
    check(other);
    TORCH_CHECK(input.dim() == 1 && input.numel() >= 4 && other.sizes() == input.sizes(), "expected matching vectors");
    const auto n = input.numel();
    QuantLib::Array values(n);
    quantlibtorch::unpack_array(input, values.begin());
    const auto copy = values;
    values[1] += quantlibtorch::TorchReal(other[1]);
    auto reordered = copy;
    std::swap(reordered[0], reordered[n-1]);
    return {quantlibtorch::pack_array(copy.begin(), n),
            quantlibtorch::pack_array(values.begin(), n),
            quantlibtorch::pack_array(copy.begin()+1, n-2),
            quantlibtorch::pack_array(reordered.begin(), n)};
}

at::Tensor array_grad_boundary(const at::Tensor& input, bool unpack_no_grad, bool pack_no_grad) {
    quantlibtorch::AdjointMode lazy_view_scope(true);
    check(input);
    TORCH_CHECK(input.dim() == 1, "expected vector");
    QuantLib::Array values(input.numel());
    {
        at::AutoGradMode guard(!unpack_no_grad);
        quantlibtorch::unpack_array(input, values.begin());
    }
    at::AutoGradMode guard(!pack_no_grad);
    return quantlibtorch::pack_array(values.begin(), values.size());
}

at::Tensor array_arithmetic(const at::Tensor& input, const at::Tensor& other,
                           const at::Tensor& scalar, int64_t mode) {
    quantlibtorch::AdjointMode lazy_view_scope(true);
    check(input);
    check(other);
    check(scalar);
    TORCH_CHECK(input.dim() == 1 && other.sizes() == input.sizes() && scalar.dim() == 0, "invalid array shapes");
    quantlibtorch::TensorKernelMode guard(static_cast<int>(mode));
    QuantLib::Array a(input.numel()), b(other.numel());
    quantlibtorch::unpack_array(input, a.begin());
    quantlibtorch::unpack_array(other, b.begin());
    const quantlibtorch::TorchReal c(scalar);
    std::vector<at::Tensor> results;
    auto append = [&](const QuantLib::Array& x) { results.push_back(quantlibtorch::pack_array(x.begin(), x.size())); };
    // Each value category selects a distinct upstream overload, including
    // aliases in compound assignment. The reference is independent PyTorch.
#define QLT_ARRAY_TEST(OP) \
    append(a OP b); append(QuantLib::Array(a) OP b); \
    append(a OP QuantLib::Array(b)); append(QuantLib::Array(a) OP QuantLib::Array(b)); \
    append(a OP c); append(QuantLib::Array(a) OP c); \
    append(c OP a); append(c OP QuantLib::Array(a)); \
    { auto x = a; x OP##= b; append(x); } \
    { auto x = a; x OP##= c; append(x); } \
    { auto x = a; x OP##= x; append(x); }
    QLT_ARRAY_TEST(+)
    QLT_ARRAY_TEST(-)
    QLT_ARRAY_TEST(*)
    QLT_ARRAY_TEST(/)
#undef QLT_ARRAY_TEST
    return at::stack(results);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_hook_apply(Tensor bands, Tensor rhs, int mode, bool adjoint=False) -> Tensor");
    m.def("_hook_assembly(Tensor a, Tensor x, Tensor y, Tensor b, Tensor scale, int mode, int alias) -> Tensor");
    m.def("_hook_diffusion(Tensor variance, Tensor dx, Tensor dxx, Tensor rate, Tensor yield, bool adjoint, int alias=0, int mode=4) -> Tensor");
    m.def("_diffusion_basis_series(Tensor variances, Tensor dx, Tensor dxx, Tensor rate, Tensor yield, bool adjoint=False, int mode=4) -> Tensor");
    m.def("_array_roundtrip(Tensor input) -> Tensor");
    m.def("_array_views(Tensor input, Tensor other) -> Tensor[]");
    m.def("_array_grad_boundary(Tensor input, bool unpack_no_grad, bool pack_no_grad) -> Tensor");
    m.def("_array_arithmetic(Tensor input, Tensor other, Tensor scalar, int mode) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_hook_apply", TORCH_FN(hook_apply));
    m.impl("_hook_assembly", TORCH_FN(hook_assembly));
    m.impl("_hook_diffusion", TORCH_FN(hook_diffusion));
    m.impl("_diffusion_basis_series", TORCH_FN(diffusion_basis_series));
    m.impl("_array_roundtrip", TORCH_FN(roundtrip));
    m.impl("_array_views", TORCH_FN(array_views));
    m.impl("_array_grad_boundary", TORCH_FN(array_grad_boundary));
    m.impl("_array_arithmetic", TORCH_FN(array_arithmetic));
}
