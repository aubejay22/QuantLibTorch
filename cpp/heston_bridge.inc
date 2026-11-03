// Included in fd_bridge.cpp to share the financial Settings lock and curves.
class FixedPricingAxis final : public Fdm1dMesher {
  public:
    explicit FixedPricingAxis(const Doubles& xs) : Fdm1dMesher(xs.size()) {
        check_axis(xs.vec(), xs.size(), "pricing axis");
        TORCH_CHECK(xs.size() >= 5, "pricing axes need at least five nodes");
        for (size_t i = 0; i < xs.size(); ++i) locations_[i] = xs.get(i);
        for (size_t i = 1; i < xs.size(); ++i)
            dminus_[i] = dplus_[i-1] = locations_[i] - locations_[i-1];
        dminus_.front() = dplus_.back() = Null<Real>();
    }
};

struct HestonObjects {
    ext::shared_ptr<FdHestonVanillaEngine> engine;
    ext::shared_ptr<VanillaOption> option;
};

HestonObjects heston_objects(const std::vector<Real>& p, double strike,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const at::Tensor& risk_dfs,
    const Ints& yield_dates, const at::Tensor& yield_dfs,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials) {
    TORCH_CHECK(p.size() == 8, "Heston parameters must have eight entries");
    for (const auto& value : p) TORCH_CHECK(std::isfinite(value.value()), "Heston parameters must be finite");
    TORCH_CHECK(p[0] > 0. && p[1] > 0. && p[2] > 0. && p[3] > 0. && p[4] > 0.
                && p[5] > -1. && p[5] < 1. && std::isfinite(strike) && strike > 0.,
                "invalid Heston parameters or strike");
    TORCH_CHECK(config.at("time_steps") >= 1 && config.at("x_grid") >= 5 && config.at("v_grid") >= 5,
                "invalid Heston grid sizes");
    TORCH_CHECK(config.at("damping_steps") >= 0, "damping steps must be nonnegative");
    TORCH_CHECK(!config.at("adjoint") || config.at("damping_steps") == 0,
                "Heston adjoint damping requires an unsupported Krylov adjoint");
    TORCH_CHECK(config.at("scheme") >= 0 && config.at("scheme") <= 3, "invalid Heston scheme");
    TORCH_CHECK(config.at("option_type") == -1 || config.at("option_type") == 1, "invalid option type");
    TORCH_CHECK(config.at("exercise") >= 0 && config.at("exercise") <= 2, "invalid exercise type");
    TORCH_CHECK(config.at("adjoint") == 0 || config.at("adjoint") == 1, "invalid AD mode");
    TORCH_CHECK(config.at("kernels") == 0 || config.at("kernels") == 4, "invalid Heston kernel mode");
    Ints endpoints;
    endpoints.push_back(config.at("valuation_date")); endpoints.push_back(config.at("maturity_date"));
    const auto ds = dates(endpoints);
    const Date reference = ds[0], maturity = ds[1];
    TORCH_CHECK(maturity > reference, "maturity must follow valuation");
    Settings::instance().evaluationDate() = reference;
    const auto day_counter = selected_day_counter(config.contains("day_counter") ? config.at("day_counter") : 0);
    const auto risk = tensor_curve(reference, maturity, p[6], risk_dates, risk_dfs, day_counter);
    const auto yield = tensor_curve(reference, maturity, p[7], yield_dates, yield_dfs, day_counter);
    auto process = ext::make_shared<HestonProcess>(risk, yield,
        Handle<Quote>(ext::make_shared<SimpleQuote>(p[0])), p[1], p[2], p[3], p[4], p[5]);
    auto model = ext::make_shared<HestonModel>(process);
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
    const auto scheme_type = config.at("scheme");
    const auto scheme = scheme_type == 0 ? FdmSchemeDesc::Douglas()
        : scheme_type == 1 ? FdmSchemeDesc::CraigSneyd()
        : scheme_type == 2 ? FdmSchemeDesc::ModifiedCraigSneyd() : FdmSchemeDesc::Hundsdorfer();
    auto engine = ext::make_shared<FdHestonVanillaEngine>(model, dividends,
        config.at("time_steps"), config.at("x_grid"), config.at("v_grid"), config.at("damping_steps"), scheme);
    auto option = ext::make_shared<VanillaOption>(
        ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(config.at("option_type")), strike), exercise);
    option->setPricingEngine(engine);
    option->setupArguments(engine->getArguments());
    engine->getArguments()->validate();
    return {engine, option};
}

std::tuple<at::Tensor, at::Tensor> heston_mesh(const Doubles& parameters, double strike,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const Doubles& risk_dfs, const Ints& yield_dates, const Doubles& yield_dfs,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials) {
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved;
    std::vector<Real> p;
    for (double value : parameters) p.emplace_back(value);
    const auto options = at::TensorOptions().dtype(at::kDouble);
    auto objects = heston_objects(p, strike, config, risk_dates, at::tensor(risk_dfs.vec(), options),
        yield_dates, at::tensor(yield_dfs.vec(), options), cash_dates, cash_amounts, exercise_serials);
    auto mesh = ext::dynamic_pointer_cast<FdmMesherComposite>(objects.engine->getSolverDesc(1.5).mesher);
    std::vector<double> x, v;
    for (const auto& value : mesh->getFdm1dMeshers()[0]->locations()) x.push_back(value.value());
    for (const auto& value : mesh->getFdm1dMeshers()[1]->locations()) v.push_back(value.value());
    return {at::tensor(x, options), at::tensor(v, options)};
}

at::Tensor fd_heston(const at::Tensor& parameters, double strike,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const at::Tensor& risk_dfs, const Ints& yield_dates, const at::Tensor& yield_dfs,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials,
    const Doubles& log_spots, const Doubles& variances) {
    for (const auto& t : {parameters, risk_dfs, yield_dfs}) {
        TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided
                    && t.dim() == 1 && at::isfinite(t).all().item<bool>(),
                    "Heston inputs must be finite dense CPU float64 vectors");
        TORCH_CHECK(!config.at("adjoint") || !t._fw_grad(0).defined(),
                    "Heston adjoint supports first-order reverse derivatives only");
    }
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved;
    quantlibtorch::AdjointMode mode(config.at("adjoint") != 0);
    quantlibtorch::TensorKernelMode kernels(config.at("kernels"));
    quantlibtorch::PriceOnlyMode npv(true);
    TORCH_CHECK(log_spots.size() == config.at("x_grid") && variances.size() == config.at("v_grid"),
                "Heston axis lengths must match grid settings");
    TORCH_CHECK(!variances.empty() && variances.get(0) == 0., "Heston variance axis must start at zero");
    auto mesh = ext::make_shared<FdmMesherComposite>(ext::make_shared<FixedPricingAxis>(log_spots),
                                                    ext::make_shared<FixedPricingAxis>(variances));
    quantlibtorch::FixedMeshScope mesh_scope(mesh);
    std::vector<Real> p;
    for (int64_t i = 0; i < parameters.numel(); ++i) p.emplace_back(parameters[i]);
    auto objects = heston_objects(p, strike, config, risk_dates, risk_dfs, yield_dates, yield_dfs,
                                 cash_dates, cash_amounts, exercise_serials);
    TORCH_CHECK(std::log(p[0].value()) >= log_spots.get(0) && std::log(p[0].value()) <= log_spots.get(log_spots.size()-1)
                && p[1].value() >= variances.get(0) && p[1].value() <= variances.get(variances.size()-1),
                "initial Heston state must lie inside the fixed mesh");
    const auto result = objects.option->NPV().tensor();
    TORCH_CHECK(at::isfinite(result).item<bool>(), "non-finite Heston price");
    return result;
}
