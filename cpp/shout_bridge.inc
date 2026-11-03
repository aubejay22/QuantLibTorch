// Original shout engine, with explicit passive geometry and active market data.
struct ShoutObjects {
    ext::shared_ptr<GeneralizedBlackScholesProcess> process;
    DividendSchedule dividends;
    ext::shared_ptr<VanillaOption> option;
    Date reference, maturity;
};

ShoutObjects shout_objects(const std::vector<Real>& p, double strike,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const at::Tensor& risk_dfs,
    const Ints& yield_dates, const at::Tensor& yield_dfs,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials) {
    TORCH_CHECK(p.size() == 4, "shout parameters must have four entries");
    for (const auto& value : p) TORCH_CHECK(std::isfinite(value.value()), "shout parameters must be finite");
    TORCH_CHECK(p[0] > 0. && p[1] > 0. && std::isfinite(strike) && strike > 0., "invalid shout spot, volatility or strike");
    TORCH_CHECK(config.at("time_steps") >= 1 && config.at("x_grid") >= 5 && config.at("damping_steps") >= 0,
                "invalid shout grid settings");
    TORCH_CHECK(config.at("scheme") >= 0 && config.at("scheme") <= 5, "invalid shout scheme");
    TORCH_CHECK(config.at("option_type") == -1 || config.at("option_type") == 1, "invalid option type");
    TORCH_CHECK(config.at("exercise") >= 0 && config.at("exercise") <= 2, "invalid exercise type");
    TORCH_CHECK(config.at("adjoint") == 0 || config.at("adjoint") == 1, "invalid AD mode");
    TORCH_CHECK(config.at("kernels") == 0 || config.at("kernels") == 4, "invalid shout kernel mode");
    Ints endpoints;
    endpoints.push_back(config.at("valuation_date")); endpoints.push_back(config.at("maturity_date"));
    const auto ds = dates(endpoints);
    const Date reference = ds[0], maturity = ds[1];
    TORCH_CHECK(maturity > reference, "maturity must follow valuation");
    Settings::instance().evaluationDate() = reference;
    const auto day_counter = selected_day_counter(config.contains("day_counter") ? config.at("day_counter") : 0);
    const auto risk = tensor_curve(reference, maturity, p[2], risk_dates, risk_dfs, day_counter);
    const auto yield = tensor_curve(reference, maturity, p[3], yield_dates, yield_dfs, day_counter);
    auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(p[0])), yield, risk,
        Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(reference, NullCalendar(), p[1], day_counter)));
    ext::shared_ptr<Exercise> exercise;
    const auto ex_dates = dates(exercise_serials);
    if (config.at("exercise") == 2) {
        TORCH_CHECK(!ex_dates.empty() && ex_dates.front() > reference && ex_dates.back() == maturity,
                    "Bermudan dates must follow valuation and end at maturity");
        for (size_t i = 1; i < ex_dates.size(); ++i)
            TORCH_CHECK(ex_dates[i] > ex_dates[i-1], "exercise dates must increase");
        exercise = ext::make_shared<BermudanExercise>(ex_dates);
    } else {
        TORCH_CHECK(ex_dates.empty(), "exercise dates require Bermudan exercise");
        exercise = config.at("exercise") == 1
            ? ext::static_pointer_cast<Exercise>(ext::make_shared<AmericanExercise>(reference, maturity))
            : ext::static_pointer_cast<Exercise>(ext::make_shared<EuropeanExercise>(maturity));
    }
    TORCH_CHECK(cash_dates.size() == cash_amounts.size(), "cash dividend length mismatch");
    DividendSchedule dividends;
    const auto cash_ds = dates(cash_dates);
    for (size_t i = 0; i < cash_ds.size(); ++i) {
        TORCH_CHECK(cash_ds[i] > reference && cash_ds[i] <= maturity
            && (i == 0 || cash_ds[i] > cash_ds[i-1])
            && std::isfinite(cash_amounts.get(i)) && cash_amounts.get(i) >= 0., "invalid cash dividends");
        dividends.push_back(ext::make_shared<FixedDividend>(cash_amounts.get(i), cash_ds[i]));
    }
    auto option = ext::make_shared<VanillaOption>(
        ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(config.at("option_type")), strike), exercise);
    return {process, dividends, option, reference, maturity};
}

Real shout_adjustment(const ShoutObjects& objects) {
    const auto& process = objects.process;
    const Time maturity = process->time(objects.maturity);
    const EscrowedDividendAdjustment adjustment(objects.dividends, process->riskFreeRate(),
        process->dividendYield(), [process](Date d) { return process->time(d); }, maturity);
    return adjustment.dividendAdjustment(process->time(objects.reference));
}

at::Tensor shout_mesh(const Doubles& parameters, double strike,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const Doubles& risk_dfs, const Ints& yield_dates, const Doubles& yield_dfs,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials) {
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved;
    std::vector<Real> p;
    for (double value : parameters) p.emplace_back(value);
    const auto options = at::TensorOptions().dtype(at::kDouble);
    const auto objects = shout_objects(p, strike, config, risk_dates, at::tensor(risk_dfs.vec(), options),
        yield_dates, at::tensor(yield_dfs.vec(), options), cash_dates, cash_amounts, exercise_serials);
    const auto div_adj = shout_adjustment(objects);
    TORCH_CHECK(objects.process->x0() + div_adj > 0., "spot minus dividends becomes negative");
    // Exact original FdBlackScholesShoutEngine mesher construction, without
    // rollback. The configured tensor graph never enters numerical geometry.
    const FdmBlackScholesMesher mesh(config.at("x_grid"), objects.process,
        objects.process->time(objects.maturity), strike, Null<Real>(), Null<Real>(), .0001, 1.5,
        std::pair<Real,Real>(strike, .1), DividendSchedule(), ext::shared_ptr<FdmQuantoHelper>(), div_adj);
    std::vector<double> x;
    for (const auto& value : mesh.locations()) x.push_back(value.value());
    return at::tensor(x, options);
}

at::Tensor fd_shout(const at::Tensor& parameters, double strike,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const at::Tensor& risk_dfs, const Ints& yield_dates, const at::Tensor& yield_dfs,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials, const Doubles& log_spots) {
    for (const auto& t : {parameters, risk_dfs, yield_dfs}) {
        TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided
                    && t.dim() == 1 && at::isfinite(t).all().item<bool>(),
                    "shout inputs must be finite dense CPU float64 vectors");
        TORCH_CHECK(!config.at("adjoint") || !t._fw_grad(0).defined(),
                    "shout adjoint supports first-order reverse derivatives only");
    }
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved;
    quantlibtorch::AdjointMode mode(config.at("adjoint") != 0);
    quantlibtorch::TensorKernelMode kernels(config.at("kernels"));
    quantlibtorch::PriceOnlyMode npv(true);
    TORCH_CHECK(log_spots.size() == config.at("x_grid"), "shout axis length must match grid setting");
    auto mesh = ext::make_shared<FdmMesherComposite>(ext::make_shared<FixedPricingAxis>(log_spots));
    quantlibtorch::FixedMeshScope mesh_scope(mesh);
    std::vector<Real> p;
    for (int64_t i = 0; i < parameters.numel(); ++i) p.emplace_back(parameters[i]);
    auto objects = shout_objects(p, strike, config, risk_dates, risk_dfs, yield_dates, yield_dfs,
                                 cash_dates, cash_amounts, exercise_serials);
    const auto adjusted_spot = objects.process->x0() + shout_adjustment(objects);
    TORCH_CHECK(adjusted_spot > 0. && std::log(adjusted_spot.value()) >= log_spots.get(0)
                && std::log(adjusted_spot.value()) <= log_spots.get(log_spots.size()-1),
                "adjusted spot must lie inside the fixed shout mesh");
    const auto type = config.at("scheme");
    const auto scheme = type == 0 ? FdmSchemeDesc::Douglas()
        : type == 1 ? FdmSchemeDesc::CrankNicolson() : type == 2 ? FdmSchemeDesc::ImplicitEuler()
        : type == 3 ? FdmSchemeDesc::CraigSneyd() : type == 4 ? FdmSchemeDesc::ModifiedCraigSneyd()
        : FdmSchemeDesc::Hundsdorfer();
    objects.option->setPricingEngine(ext::make_shared<FdBlackScholesShoutEngine>(objects.process, objects.dividends,
        config.at("time_steps"), config.at("x_grid"), config.at("damping_steps"), scheme));
    const auto result = objects.option->NPV().tensor();
    TORCH_CHECK(at::isfinite(result).item<bool>(), "non-finite shout price");
    return result;
}
