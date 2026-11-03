#pragma once
#include <quantlibtorch/torch_real.hpp>
#include <ATen/core/List.h>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

namespace quantlibtorch {
// Per-call curves: graph-connected values must never enter passive plan caches.
inline QuantLib::ext::shared_ptr<QuantLib::YieldTermStructure> financial_curve(
    const QuantLib::Date& ref, const QuantLib::Date& end, const at::Tensor& rate,
    const c10::List<int64_t>& dates, const at::Tensor& discounts) {
    using namespace QuantLib;
    const auto check=[](const at::Tensor& x) {
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kDouble && x.layout()==at::kStrided,
                    "financial inputs require dense CPU float64 tensors");
        TORCH_CHECK(at::isfinite(x).all().item<bool>(), "financial inputs must be finite");
    };
    check(rate); check(discounts);
    TORCH_CHECK(rate.dim()==0 && discounts.dim()==1, "invalid financial input dimensions");
    if (dates.empty()) {
        TORCH_CHECK(discounts.numel()==0, "curve dates missing");
        return ext::make_shared<FlatForward>(ref, Real(rate), Actual365Fixed());
    }
    TORCH_CHECK(!rate.requires_grad() && !rate._fw_grad(0).defined(), "active flat rate is overridden by dated curve");
    TORCH_CHECK(dates.size()==discounts.numel() && dates.size()>=2 &&
        dates.get(0)==ref.serialNumber() && dates.get(dates.size()-1)>=end.serialNumber() &&
        discounts[0].item<double>()==1., "invalid discount curve coverage");
    std::vector<Date> ds; std::vector<Real> dfs;
    for (size_t i=0;i<dates.size();++i) {
        TORCH_CHECK(discounts[i].item<double>()>0 && (i==0 || dates.get(i)>dates.get(i-1)),
                    "discounts must be positive and dates increasing");
        ds.emplace_back(dates.get(i));
        // Reference-date normalization is fixed at one, as in public validation.
        dfs.emplace_back(i==0 ? Real(1.) : Real(discounts[i]));
    }
    return ext::make_shared<DiscountCurve>(ds,dfs,Actual365Fixed());
}
}
