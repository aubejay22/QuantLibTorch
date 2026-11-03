#include <quantlibtorch/parallel_solve.hpp>
#include <quantlibtorch/pcr_stage.hpp>
#include <quantlibtorch/thomas_arithmetic.hpp>
#include <quantlibtorch/solve_checks.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <quantlibtorch/performance.hpp>
#include <ATen/record_function.h>
#include <torch/library.h>
#include <limits>

namespace quantlibtorch {
namespace {
at::Tensor shift(const at::Tensor& t, int64_t step, bool left, int64_t axis=-1, bool one=false) {
    static const bool use_roll = performance_flag("QLT_PCR_ROLL");
    // The first/last step rows have identically zero lower/upper bands.
    // Thus wrapped neighbours always multiply zero, and the next bands retain
    // those structural zeros. Roll removes padding allocations without adding
    // a periodic coupling. All divisors are checked (or caller-guaranteed)
    // nonzero. The padded formulation stays selectable for measurement.
    if (use_roll) return at::roll(t, {left ? step : -step}, {axis});
    const auto prototype = t.slice(axis, 0, step);
    const auto pad = one ? at::ones_like(prototype) : at::zeros_like(prototype);
    const auto rest = t.slice(axis, left ? 0 : step, left ? t.size(axis)-step : t.size(axis));
    return left ? at::cat({pad, rest}, axis) : at::cat({rest, pad}, axis);
}
void check_pivot(const at::Tensor& d) {
    TORCH_CHECK((at::isfinite(d) & (d != 0.)).all().item<bool>(),
                "PCR requires finite nonzero elimination pivots; use the upstream solver for other pivot patterns");
}

bool tensor_auto_domain(const at::Tensor& lo, const at::Tensor& di,
                 const at::Tensor& up, const at::Tensor& rhs) {
    RECORD_FUNCTION("qlt::auto_domain_tensor", std::vector<c10::IValue>());
    // These comparisons choose an implementation; the selected solve below
    // still uses the ORIGINAL graph-connected tensors. No numerical input is
    // detached. Bounds deliberately restrict an optimization, not the API.
    at::NoGradGuard no_graph;
    if (di.numel() < 8) return false;
    for (const auto& t : {lo, di, up, rhs})
        if (!at::isfinite(t).all().item<bool>()) return false;
    const auto magnitude = at::abs(di);
    const auto smallest = magnitude.min().item<double>(), largest = magnitude.max().item<double>();
    if (smallest < 1e-12 || largest > 1e12 || largest/smallest > 1e6
        || at::abs(rhs).max().item<double>() > 1e50) return false;
    const auto zero = at::zeros_like(di.slice(0, 0, 1));
    const auto off = at::cat({zero, at::abs(lo)}) + at::cat({at::abs(up), zero});
    return (off <= (1.-1e-7)*magnitude).all().item<bool>();
}

bool tensor_acceptable_residual(const at::Tensor& lo, const at::Tensor& di,
                         const at::Tensor& up, const at::Tensor& rhs, const at::Tensor& x) {
    RECORD_FUNCTION("qlt::auto_residual_tensor", std::vector<c10::IValue>());
    at::NoGradGuard no_graph;
    if (!at::isfinite(x).all().item<bool>()) return false;
    const auto zero = at::zeros_like(x.slice(0, 0, 1));
    const auto center = di*x;
    const auto left = at::cat({zero, lo*x.slice(0, 0, -1)});
    const auto right = at::cat({up*x.slice(0, 1), zero});
    const auto residual = at::abs(center+left+right-rhs);
    const auto scale = at::abs(center)+at::abs(left)+at::abs(right)+at::abs(rhs);
    const auto tolerance = 64.*di.numel()*std::numeric_limits<double>::epsilon();
    return (residual <= tolerance*scale + 1e-300).all().item<bool>();
}

bool auto_domain(const at::Tensor& lo, const at::Tensor& di,
                 const at::Tensor& up, const at::Tensor& rhs) {
    bool decision;
    if (native_solve_checks_enabled() && try_native_auto_domain(lo, di, up, rhs, decision)) return decision;
    return tensor_auto_domain(lo, di, up, rhs);
}

bool acceptable_residual(const at::Tensor& lo, const at::Tensor& di,
                         const at::Tensor& up, const at::Tensor& rhs, const at::Tensor& x) {
    bool decision;
    if (native_solve_checks_enabled() && try_native_auto_residual(lo, di, up, rhs, x, decision)) return decision;
    return tensor_acceptable_residual(lo, di, up, rhs, x);
}

bool try_pcr(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up,
             const at::Tensor& rhs, QuantLib::Array& output) {
    at::Tensor x;
    if (!try_auto_solve_tensor(lo, di, up, rhs, x)) return false;
    QuantLib::Array candidate(di.numel());
    unpack_array(x, candidate.begin());
    output = std::move(candidate);
    return true;
}
}

bool try_auto_solve_tensor(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up,
                          const at::Tensor& rhs, at::Tensor& output) {
    if (!auto_domain(lo, di, up, rhs)) return false;
    // Strict dominance with a relative margin and bounded scales excludes the
    // problematic pivot patterns. The residual check still permits fallback
    // without hiding an error from the original QuantLib solver.
    const auto matrix_rhs = rhs.unsqueeze(-1);
    at::Tensor solution;
    if (!try_thomas_arithmetic(lo, di, up, matrix_rhs, solution))
        solution = tridiagonal_solve_pcr(lo, di, up, matrix_rhs, false);
    const auto x = solution.squeeze(-1);
    if (!acceptable_residual(lo, di, up, rhs, x)) return false;
    output = x;
    return true;
}

at::Tensor tridiagonal_solve_pcr(const at::Tensor& lower, const at::Tensor& diagonal,
    const at::Tensor& upper, const at::Tensor& rhs, bool check_pivots) {
    RECORD_FUNCTION("qlt::pcr_solve", std::vector<c10::IValue>());
    // addcmul reduces forward dispatch but its generic backward adds work.
    // Keep this measured experiment opt-in; pricing was slower with it.
    static const bool fused = std::getenv("QLT_PCR_FUSED") && performance_flag("QLT_PCR_FUSED");
    static const bool rolled = performance_flag("QLT_PCR_ROLL");
    for (const auto& t : {lower, diagonal, upper, rhs}) {
        TORCH_CHECK(t.layout() == at::kStrided && t.scalar_type() == at::kDouble
                    && t.device() == diagonal.device() && (t.is_cpu() || t.is_cuda()),
                    "PCR inputs must be dense float64 tensors on a common CPU or CUDA device");
        if (check_pivots) TORCH_CHECK(at::isfinite(t).all().item<bool>(), "PCR inputs must be finite");
    }
    TORCH_CHECK(lower.dim() >= 1 && diagonal.dim() >= 1 && upper.dim() >= 1 && rhs.dim() >= 2,
                "expected bands (..., N), rhs (..., N, R)");
    const auto n = diagonal.size(-1);
    TORCH_CHECK(n >= 2 && lower.size(-1) == n-1 && upper.size(-1) == n-1
                && rhs.size(-2) == n && rhs.size(-1) >= 1, "invalid PCR input sizes");
    auto lo = at::cat({at::zeros_like(lower.slice(-1, 0, 1)), lower}, -1);
    auto up = at::cat({upper, at::zeros_like(upper.slice(-1, 0, 1))}, -1);
    auto d = diagonal, r = rhs;
    for (int64_t step = 1; step < n;) {
        if (check_pivots) check_pivot(d);
        std::vector<at::Tensor> stage;
        if (rolled && !fused && try_native_pcr_stage(lo, d, up, r, step, stage)) {
            lo = stage[0]; d = stage[1]; up = stage[2]; r = stage[3];
            if (step > n/2) break;
            step *= 2;
            continue;
        }
        // Eliminate neighbours at +/-step from every row simultaneously.
        // New bands couple +/-2*step. Missing-neighbour divisors are one,
        // with zero band weights, so no unused 0/0 enters the AD graph.
        const auto alpha = -lo / shift(d, step, true, -1, true);
        const auto beta = -up / shift(d, step, false, -1, true);
        auto next_d = fused
            ? at::addcmul(at::addcmul(d, alpha, shift(up, step, true)), beta, shift(lo, step, false))
            : d + alpha*shift(up, step, true) + beta*shift(lo, step, false);
        auto next_r = fused
            ? at::addcmul(at::addcmul(r, alpha.unsqueeze(-1), shift(r, step, true, -2)),
                          beta.unsqueeze(-1), shift(r, step, false, -2))
            : r + alpha.unsqueeze(-1)*shift(r, step, true, -2)
                + beta.unsqueeze(-1)*shift(r, step, false, -2);
        auto next_lo = alpha*shift(lo, step, true);
        auto next_up = beta*shift(up, step, false);
        d = next_d; r = next_r; lo = next_lo; up = next_up;
        if (step > n/2) break;
        step *= 2;
    }
    if (check_pivots) check_pivot(d);
    auto x = r/d.unsqueeze(-1);
    if (check_pivots) TORCH_CHECK(at::isfinite(x).all().item<bool>(), "non-finite PCR solution");
    return x;
}

QuantLib::Array pcr_split(const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal, const QuantLib::Real* upper) {
    const auto n = rhs.size();
    auto lo = a.tensor()*pack_array(lower+1, n-1);
    auto di = a.tensor()*pack_array(diagonal, n)+b.tensor();
    auto up = a.tensor()*pack_array(upper, n-1);
    auto x = tridiagonal_solve_pcr(lo, di, up, pack_array(rhs.begin(), n).unsqueeze(-1), true).squeeze(-1);
    QuantLib::Array output(n);
    unpack_array(x, output.begin());
    return output;
}

bool try_pcr_split(const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal, const QuantLib::Real* upper,
    QuantLib::Array& output) {
    const auto n = rhs.size();
    return try_pcr(a.tensor()*pack_array(lower+1, n-1),
        a.tensor()*pack_array(diagonal, n)+b.tensor(),
        a.tensor()*pack_array(upper, n-1), pack_array(rhs.begin(), n), output);
}

bool try_pcr_tridiagonal(const QuantLib::Array& lower, const QuantLib::Array& diagonal,
    const QuantLib::Array& upper, const QuantLib::Array& rhs, QuantLib::Array& output) {
    return try_pcr(pack_array(lower.begin(), lower.size()), pack_array(diagonal.begin(), diagonal.size()),
        pack_array(upper.begin(), upper.size()), pack_array(rhs.begin(), rhs.size()), output);
}

namespace {
void check_probe_inputs(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up, const at::Tensor& rhs) {
    for (const auto& t : {lo, di, up, rhs})
        TORCH_CHECK(t.layout() == at::kStrided && t.scalar_type() == at::kDouble && t.dim() == 1
                    && t.device() == di.device() && (t.is_cpu() || t.is_cuda()),
                    "checks expect one-dimensional float64 bands/RHS on a common CPU/CUDA device");
    const auto n = di.numel();
    TORCH_CHECK(n >= 1 && lo.numel() == n-1 && up.numel() == n-1 && rhs.numel() == n,
                "invalid check-probe input sizes");
}
bool domain_probe(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up, const at::Tensor& rhs, bool native) {
    check_probe_inputs(lo, di, up, rhs);
    bool result;
    if (native && try_native_auto_domain(lo, di, up, rhs, result)) return result;
    return tensor_auto_domain(lo, di, up, rhs);
}
bool residual_probe(const at::Tensor& lo, const at::Tensor& di, const at::Tensor& up,
                    const at::Tensor& rhs, const at::Tensor& x, bool native) {
    check_probe_inputs(lo, di, up, rhs);
    TORCH_CHECK(x.layout() == at::kStrided && x.scalar_type() == at::kDouble && x.dim() == 1
                && x.sizes() == di.sizes() && x.device() == di.device(), "invalid residual solution");
    bool result;
    if (native && try_native_auto_residual(lo, di, up, rhs, x, result)) return result;
    return tensor_acceptable_residual(lo, di, up, rhs, x);
}
}
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("tridiagonal_solve_pcr(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, bool check_pivots=True) -> Tensor");
    m.def("_auto_domain_check(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, bool native=True) -> bool");
    m.def("_auto_residual_check(Tensor lower, Tensor diagonal, Tensor upper, Tensor rhs, Tensor solution, bool native=True) -> bool");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("tridiagonal_solve_pcr", TORCH_FN(quantlibtorch::tridiagonal_solve_pcr));
    m.impl("_auto_domain_check", TORCH_FN(quantlibtorch::domain_probe));
    m.impl("_auto_residual_check", TORCH_FN(quantlibtorch::residual_probe));
}
