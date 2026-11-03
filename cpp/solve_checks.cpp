#include <quantlibtorch/solve_checks.hpp>
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
bool readable_vector(const at::Tensor& t) {
    const auto keys = t.key_set();
    return t.is_cpu() && t.layout() == at::kStrided && t.scalar_type() == at::kDouble && t.dim() == 1
        && !keys.has(c10::DispatchKey::Python) && !keys.has(c10::DispatchKey::Functionalize)
        && !keys.has(c10::DispatchKey::FuncTorchBatched) && !keys.has(c10::DispatchKey::FuncTorchGradWrapper);
}

bool eligible(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up, const at::Tensor& rhs) {
    if (at::functorch::functorchTLSAccessor()
        && !at::functorch::getDynamicLayerStack().empty()) return false;
    for (const auto& t : {lo, di, up, rhs}) if (!readable_vector(t)) return false;
    const auto n = di.numel();
    return n >= 1 && lo.numel() == n-1 && up.numel() == n-1 && rhs.numel() == n;
}

struct VectorView {
    at::Tensor owner;
    const double* data;
    int64_t stride;
    explicit VectorView(const at::Tensor& t)
        : owner(t.is_neg() ? t.resolve_neg() : t), data(owner.const_data_ptr<double>()), stride(owner.stride(0)) {}
    double operator[](int64_t i) const { return data[i*stride]; }
};
}

bool native_solve_checks_enabled() {
    static const bool enabled = performance_flag("QLT_NATIVE_SOLVE_CHECKS");
    return enabled;
}

bool try_native_auto_domain(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, bool& decision) {
    if (!eligible(lower, diagonal, upper, rhs)) return false;
    RECORD_FUNCTION("qlt::auto_domain_native", std::vector<c10::IValue>());
    at::NoGradGuard no_graph;
    decision = false;
    const auto n = diagonal.numel();
    if (n < 8) return true;
    const VectorView lo(lower), di(diagonal), up(upper), r(rhs);
    double smallest = std::numeric_limits<double>::infinity(), largest = 0.;
    for (int64_t i = 0; i < n; ++i) {
        const auto d = di[i], b = r[i];
        if (!std::isfinite(d) || !std::isfinite(b) || std::abs(b) > 1e50) return true;
        const auto magnitude = std::abs(d);
        if (magnitude < 1e-12 || magnitude > 1e12) return true;
        smallest = std::min(smallest, magnitude);
        largest = std::max(largest, magnitude);
        const auto left = i > 0 ? std::abs(lo[i-1]) : 0.;
        const auto right = i+1 < n ? std::abs(up[i]) : 0.;
        if (!std::isfinite(left) || !std::isfinite(right)
            || !(left+right <= (1.-1e-7)*magnitude)) return true;
    }
    decision = largest/smallest <= 1e6;
    return true;
}

bool try_native_auto_residual(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, const at::Tensor& solution, bool& decision) {
    if (!eligible(lower, diagonal, upper, rhs) || !readable_vector(solution)
        || solution.numel() != diagonal.numel()) return false;
    RECORD_FUNCTION("qlt::auto_residual_native", std::vector<c10::IValue>());
    at::NoGradGuard no_graph;
    decision = false;
    const VectorView lo(lower), di(diagonal), up(upper), r(rhs), x(solution);
    const auto n = diagonal.numel();
    const auto tolerance = 64.*n*std::numeric_limits<double>::epsilon();
    for (int64_t i = 0; i < n; ++i) if (!std::isfinite(x[i])) return true;
    for (int64_t i = 0; i < n; ++i) {
        // This TU disables contraction/reassociation. Preserve the separately
        // rounded multiply/add sequence of the original ATen predicate,
        // including its comparisons for nonfinite intermediate arithmetic.
        const double center = di[i]*x[i];
        const double left = i > 0 ? lo[i-1]*x[i-1] : 0.;
        const double right = i+1 < n ? up[i]*x[i+1] : 0.;
        const double residual = std::abs(((center+left)+right)-r[i]);
        const double scale = ((std::abs(center)+std::abs(left))+std::abs(right))+std::abs(r[i]);
        if (!(residual <= tolerance*scale+1e-300)) return true;
    }
    decision = true;
    return true;
}
}
