// Included in fd_bridge.cpp's anonymous namespace to share its date lock.
// All financial calculations are performed by the original QuantLib engines.
Handle<YieldTermStructure> tensor_curve(const Date& reference, const Date& maturity,
    const Real& flat, const Ints& serials, const at::Tensor& discounts,
    const DayCounter& day_counter = Actual365Fixed()) {
    TORCH_CHECK(discounts.dim() == 1 && discounts.numel() == serials.size(),
                "curve dates/discounts length mismatch");
    if (serials.empty())
        return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(reference, flat, day_counter));
    const auto ds = dates(serials);
    TORCH_CHECK(ds.size() >= 2 && ds.front() == reference && ds.back() >= maturity,
                "curve must start at valuation and cover maturity");
    TORCH_CHECK(discounts[0].item<double>() == 1., "first discount must be one");
    std::vector<Real> dfs;
    for (size_t i = 0; i < ds.size(); ++i) {
        TORCH_CHECK(discounts[i].item<double>() > 0. && (i == 0 || ds[i] > ds[i-1]),
                    "invalid discount curve");
        dfs.emplace_back(discounts[i]);
    }
    return Handle<YieldTermStructure>(ext::make_shared<DiscountCurve>(ds, dfs, day_counter));
}

at::Tensor analytic_european(const at::Tensor& market, int64_t reference_serial,
    int64_t maturity_serial, int64_t option_type,
    const Ints& risk_dates, const at::Tensor& risk_dfs,
    const Ints& yield_dates, const at::Tensor& yield_dfs,
    const Ints& cash_dates, const at::Tensor& cash) {
    for (const auto& t : {market, risk_dfs, yield_dfs, cash})
        TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided
                    && t.dim() == 1 && at::isfinite(t).all().item<bool>(),
                    "analytic inputs must be finite dense CPU float64 vectors");
    TORCH_CHECK(market.numel() == 5 && (market.slice(0, 0, 3) > 0.).all().item<bool>(),
                "spot, strike and volatility must be positive");
    TORCH_CHECK(option_type == 1 || option_type == -1, "invalid option type");
    Ints endpoint_serials;
    endpoint_serials.push_back(reference_serial); endpoint_serials.push_back(maturity_serial);
    const auto endpoints = dates(endpoint_serials);
    const auto reference = endpoints[0], maturity = endpoints[1];
    TORCH_CHECK(maturity > reference, "maturity must follow valuation");
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved_date;
    Settings::instance().evaluationDate() = reference;
    const auto risk = tensor_curve(reference, maturity, TorchReal(market[3]), risk_dates, risk_dfs);
    const auto yield = tensor_curve(reference, maturity, TorchReal(market[4]), yield_dates, yield_dfs);
    const auto black = ext::make_shared<BlackConstantVol>(reference, NullCalendar(),
                                                        TorchReal(market[2]), Actual365Fixed());
    const auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(TorchReal(market[0]))), yield, risk,
        Handle<BlackVolTermStructure>(black));
    TORCH_CHECK(cash.numel() == cash_dates.size(), "cash dividend lengths mismatch");
    DividendSchedule dividends;
    const auto ds = dates(cash_dates);
    for (size_t i = 0; i < ds.size(); ++i) {
        TORCH_CHECK(ds[i] > reference && ds[i] <= maturity && (i == 0 || ds[i] > ds[i-1])
                    && cash[i].item<double>() >= 0., "invalid cash dividend schedule");
        dividends.push_back(ext::make_shared<FixedDividend>(TorchReal(cash[i]), ds[i]));
    }
    VanillaOption option(ext::make_shared<PlainVanillaPayoff>(
        static_cast<Option::Type>(option_type), TorchReal(market[1])),
        ext::make_shared<EuropeanExercise>(maturity));
    if (dividends.empty()) option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));
    else option.setPricingEngine(ext::make_shared<AnalyticDividendEuropeanEngine>(process, dividends));
    const auto result = option.NPV().tensor();
    TORCH_CHECK(at::isfinite(result).item<bool>(), "non-finite analytic price");
    return result;
}
