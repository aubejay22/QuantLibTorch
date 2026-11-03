/* Vector forms of QuantLib 1.40 cubicinterpolation.hpp expressions.
 * Original algorithm copyright:
 * Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl
 * Copyright (C) 2001, 2002, 2003 Nicolas Di Césaré
 * Copyright (C) 2004, 2008, 2009, 2011 Ferdinando Ametrano
 * Copyright (C) 2009 Sylvain Bertrand
 * Copyright (C) 2013 Peter Caspers
 * Copyright (C) 2016 Nicholas Bertocchi
 * Distributed under the QuantLib license (LICENSES/QuantLib.txt).
 * No XAD or QuantLibAAD code is used.
 */
#include <quantlibtorch/spline_kernels.hpp>
#include <ATen/record_function.h>

namespace quantlibtorch {
namespace {
bool active(const QuantLib::Real* first, QuantLib::Size n) {
    for (QuantLib::Size i = 0; i < n; ++i) if (!first[i].is_passive()) return true;
    return false;
}
at::Tensor pack(const std::vector<QuantLib::Real>& values) { return pack_array(values.data(), values.size()); }
bool finite(std::initializer_list<at::Tensor> tensors) {
    for (const auto& t : tensors) if (!at::isfinite(t).all().item<bool>()) return false;
    return true;
}
// Unlike aten::minimum/maximum, std::min/max keep the FIRST operand on a
// tie. This is part of QuantLib's piecewise derivative and branch contract.
at::Tensor ordered_min(const at::Tensor& a, const at::Tensor& b) { return at::where(b < a, b, a); }
at::Tensor ordered_max(const at::Tensor& a, const at::Tensor& b) { return at::where(a < b, b, a); }
}

bool spline_slopes(const QuantLib::Array& x, const QuantLib::Array& y,
                   QuantLib::Real* dx, QuantLib::Real* slopes) {
    const auto n = x.size();
    if (!spline_kernels_enabled() || (!active(x.begin(), n) && !active(y.begin(), n))) return false;
    RECORD_FUNCTION("qlt::spline_slopes", std::vector<c10::IValue>());
    const auto yy = pack_array(y.begin(), n);
    at::Tensor h;
    if (!active(x.begin(), n)) {
        // Keep fixed grid arithmetic passive for the upstream matrix assembly.
        for (QuantLib::Size i = 0; i < n-1; ++i) dx[i] = x[i+1]-x[i];
        h = pack_array(dx, n-1);
    } else {
        const auto xx = pack_array(x.begin(), n);
        h = xx.slice(0, 1, n)-xx.slice(0, 0, n-1);
        unpack_array(h, dx);
    }
    const auto s = (yy.slice(0, 1, n)-yy.slice(0, 0, n-1))/h;
    unpack_array(s, slopes);
    return true;
}

bool spline_rhs(const std::vector<QuantLib::Real>& dx, const std::vector<QuantLib::Real>& slopes,
                QuantLib::Array& rhs) {
    const auto n = rhs.size();
    if (!spline_kernels_enabled() || n <= 2 || (!active(dx.data(), n-1) && !active(slopes.data(), n-1))) return false;
    RECORD_FUNCTION("qlt::spline_rhs", std::vector<c10::IValue>());
    const auto h = pack(dx), s = pack(slopes);
    const auto mid = 3.0*(h.slice(0, 1, n-1)*s.slice(0, 0, n-2)+h.slice(0, 0, n-2)*s.slice(0, 1, n-1));
    unpack_array(mid, rhs.begin()+1);
    return true;
}

bool spline_filter(const std::vector<QuantLib::Real>& dx, const std::vector<QuantLib::Real>& slopes,
                   QuantLib::Array& derivatives, std::vector<bool>& adjustments) {
    const auto n = derivatives.size();
    if (!spline_kernels_enabled() || (!active(derivatives.begin(), n) && !active(slopes.data(), n-1))) return false;
    RECORD_FUNCTION("qlt::spline_filter", std::vector<c10::IValue>());
    const auto h = pack(dx), s = pack(slopes), d = pack_array(derivatives.begin(), n);
    if (!finite({h, s, d})) return false;
    // Avoid 0/0 in unselected branches. Where both corrections coincide,
    // upstream keeps the original derivative, including at a flat-data tie.
    const auto sign_ratio = d / at::where(d != 0., at::abs(d), at::ones_like(d));
    const auto edge = [&](int64_t i, int64_t j) {
        auto dd = d.slice(0, i, i+1), ss = s.slice(0, j, j+1);
        return at::where(dd*ss > 0., sign_ratio.slice(0, i, i+1)*ordered_min(at::abs(dd), at::abs(3.0*ss)), at::zeros_like(dd));
    };
    auto left = edge(0, 0), right = edge(n-1, n-2);
    at::Tensor correction;
    if (n == 2) {
        correction = at::cat({left, right});
    } else {
        const auto sl = s.slice(0, 0, n-2), sr = s.slice(0, 1, n-1);
        const auto hl = h.slice(0, 0, n-2), hr = h.slice(0, 1, n-1);
        const auto pm = (sl*hr+sr*hl)/(hl+hr);
        auto M = 3.0*ordered_min(ordered_min(at::abs(sl), at::abs(sr)), at::abs(pm));
        if (n > 3) {
            const auto previous = s.slice(0, 0, n-3), center = s.slice(0, 1, n-2), next = s.slice(0, 2, n-1);
            const auto previous_h = h.slice(0, 0, n-3), center_h = h.slice(0, 1, n-2), next_h = h.slice(0, 2, n-1);
            const auto pd = (center*(2.0*center_h+previous_h)-previous*center_h)/(previous_h+center_h);
            const auto pu = (center*(2.0*center_h+next_h)-next*center_h)/(center_h+next_h);
            if (!finite({pd, pu})) return false;
            const auto rising = (center-previous)*(next-center) > 0.;
            const auto pmd = pm.slice(0, 1, n-2), pmu = pm.slice(0, 0, n-3);
            auto tail = M.slice(0, 1, n-2);
            tail = at::where(rising & (pmd*pd > 0.) & (pmd*(center-previous) > 0.),
                ordered_max(tail, 1.5*ordered_min(at::abs(pmd), at::abs(pd))), tail);
            M = at::cat({M.slice(0, 0, 1), tail});
            auto head = M.slice(0, 0, n-3);
            head = at::where(rising & (pmu*pu > 0.) & (-pmu*(center-previous) > 0.),
                ordered_max(head, 1.5*ordered_min(at::abs(pmu), at::abs(pu))), head);
            M = at::cat({head, M.slice(0, n-3, n-2)});
        }
        if (!finite({pm, M})) return false;
        const auto mid = d.slice(0, 1, n-1);
        correction = at::cat({left, at::where(mid*pm > 0.,
            sign_ratio.slice(0, 1, n-1)*ordered_min(at::abs(mid), M), at::zeros_like(mid)), right});
    }
    if (!finite({correction})) return false;
    const auto changed = (correction != d).contiguous();
    const auto result = at::where(changed, correction, d);
    const auto* mask = changed.const_data_ptr<bool>();
    for (QuantLib::Size i = 0; i < n; ++i) adjustments[i] = mask[i];
    unpack_array(result, derivatives.begin());
    return true;
}

bool spline_coefficients(const std::vector<QuantLib::Real>& dx, const std::vector<QuantLib::Real>& slopes,
                         const QuantLib::Array& derivatives, QuantLib::Real* a,
                         QuantLib::Real* b, QuantLib::Real* c) {
    const auto n = derivatives.size();
    if (!spline_kernels_enabled() || (!active(derivatives.begin(), n) && !active(slopes.data(), n-1))) return false;
    RECORD_FUNCTION("qlt::spline_coefficients", std::vector<c10::IValue>());
    const auto h = pack(dx), s = pack(slopes), d = pack_array(derivatives.begin(), n);
    const auto l = d.slice(0, 0, n-1), r = d.slice(0, 1, n);
    const auto bb = (3.0*s-r-2.0*l)/h;
    const auto cc = (r+l-2.0*s)/(h*h);
    unpack_array(l, a);
    unpack_array(bb, b);
    unpack_array(cc, c);
    return true;
}

QuantLib::Array spline_increments(const QuantLib::Array& y, const std::vector<QuantLib::Real>& dx,
    const std::vector<QuantLib::Real>& a, const std::vector<QuantLib::Real>& b,
    const std::vector<QuantLib::Real>& c) {
    const auto n = y.size();
    if (!active(y.begin(), n) && !active(a.data(), n-1) && !active(b.data(), n-1) && !active(c.data(), n-1)) return {};
    RECORD_FUNCTION("qlt::spline_primitive_increments", std::vector<c10::IValue>());
    const auto h = pack_array(dx.data(), n-2);
    const auto result = h*(pack_array(y.begin(), n-2)+h*(pack_array(a.data(), n-2)/2.0
        +h*(pack_array(b.data(), n-2)/3.0+h*pack_array(c.data(), n-2)/4.0)));
    QuantLib::Array increments(n-2);
    unpack_array(result, increments.begin());
    // The caller retains upstream's sequential prefix additions (no cumsum
    // reordering), and exposes all primitive constants eagerly as before.
    return increments;
}
}
