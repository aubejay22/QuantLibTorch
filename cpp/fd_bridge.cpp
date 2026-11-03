#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <quantlibtorch/local_vol.hpp>
#include <quantlibtorch/time_variance.hpp>
#include <quantlibtorch/step_adjoint.hpp>
#include <quantlibtorch/pricing_output.hpp>
#include <ql/cashflows/dividend.hpp>
#include <ql/exercise.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/schemes/douglasscheme.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/pricingengines/vanilla/fdhestonvanillaengine.hpp>
#include <quantlibtorch/fixed_mesh.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesshoutengine.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/utilities/escroweddividendadjustment.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/analyticdividendeuropeanengine.hpp>
#include <ql/pricingengines/basket/fd2dblackscholesvanillaengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/weekendsonly.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <tuple>

static_assert(std::is_same_v<QuantLib::Real, quantlibtorch::TorchReal>);

namespace {
using namespace QuantLib;
using quantlibtorch::TorchReal;
using Ints = c10::List<int64_t>;
using Doubles = c10::List<double>;

// QuantLib Settings and observer registries are not a concurrent pricing API.
// Each call owns its objects; only evaluation-date access is serialized here.
std::mutex pricing_mutex;

bool spatial_cache_enabled() {
    static const bool enabled = quantlibtorch::performance_flag("QLT_SPATIAL_CACHE");
    return enabled;
}

class SavedEvaluationDate {
  public:
    // SavedSettings reads DateProxy through its conversion to Date, which
    // anchors an originally floating/null evaluation date to today's date.
    // Preserve the raw value so the functional API restores the exact state.
    SavedEvaluationDate() : previous_(Settings::instance().evaluationDate().value()) {}
    ~SavedEvaluationDate() {
        try {
            Settings::instance().evaluationDate() = previous_;
        } catch (...) {
            // DateProxy restores its value before notifying any observers.
            // A destructor must not propagate notification exceptions.
        }
    }
    SavedEvaluationDate(const SavedEvaluationDate&) = delete;
    SavedEvaluationDate& operator=(const SavedEvaluationDate&) = delete;
  private:
    Date previous_;
};

int64_t evaluation_date() {
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    return Settings::instance().evaluationDate().value().serialNumber();
}

void check_axis(const std::vector<double>& axis, int64_t size, const char* name) {
    TORCH_CHECK(static_cast<int64_t>(axis.size()) == size && size > 0,
                name, " must match its surface dimension");
    for (size_t i = 0; i < axis.size(); ++i) {
        TORCH_CHECK(std::isfinite(axis[i]), name, " must be finite");
        TORCH_CHECK(i == 0 || axis[i] > axis[i-1], name, " must be strictly increasing");
    }
}

class TensorLocalVol final : public LocalVolTermStructure {
  public:
    TensorLocalVol(const Date& reference, at::Tensor values,
                   std::vector<double> times, std::vector<double> log_spots)
    : LocalVolTermStructure(reference, NullCalendar(), Following, Actual365Fixed()),
      values_(std::move(values)), times_(std::move(times)), xs_(std::move(log_spots)) {}
    Date maxDate() const override { return Date::maxDate(); }
    Real minStrike() const override { return 0.0; }
    Real maxStrike() const override { return (std::numeric_limits<double>::max)(); }
    at::Tensor sampleGrid(Time t, const Array& spots) const {
        RECORD_FUNCTION("qlt::sample_grid", std::vector<c10::IValue>());
        checkRange(t, true);
        // Coordinates are fixed and passive in this path. Graph-connected
        // spatial rows can be reused within this pricing call; coefficients
        // are never detached, mutated, or cached across separate prices.
        std::vector<double> grid;
        grid.reserve(spots.size());
        for (const auto& spot : spots) grid.push_back(spot.value());
        if (grid != sample_spots_) {
            std::vector<int64_t> lo, hi;
            std::vector<double> weights;
            for (const auto spot : grid) {
                const auto xi = interval(xs_, std::log(spot));
                lo.push_back(xi.lo);
                hi.push_back(xi.hi);
                weights.push_back(xi.weight.value());
            }
            grid_lo_ = at::tensor(lo, at::TensorOptions().dtype(at::kLong));
            grid_hi_ = at::tensor(hi, at::TensorOptions().dtype(at::kLong));
            grid_weights_ = at::tensor(weights, values_.options().requires_grad(false));
            sample_spots_ = std::move(grid);
            spatial_rows_.clear();
            spatial_variances_.clear();
        }
        const bool cached = spatial_cache_enabled();
        if (cache_recording_ != at::GradMode::is_enabled()) {
            spatial_rows_.clear();
            spatial_variances_.clear();
            cache_recording_ = at::GradMode::is_enabled();
        }
        const auto row = [&](Size i) -> at::Tensor {
            if (cached) {
                const auto found = spatial_rows_.find(i);
                if (found != spatial_rows_.end()) return found->second;
            }
            RECORD_FUNCTION("qlt::spatial_row", std::vector<c10::IValue>());
            auto values = values_.select(0, i);
            auto interpolated = xs_.size() == 1
                ? values.expand({static_cast<int64_t>(spots.size())})
                : (1.0-grid_weights_)*values.index_select(0, grid_lo_)
                  + grid_weights_*values.index_select(0, grid_hi_);
            if (cached) spatial_rows_.emplace(i, interpolated);
            return interpolated;
        };
        const auto ti = interval(times_, t);
        if (cached && ti.lo == ti.hi) {
            const auto found = spatial_variances_.find(ti.lo);
            if (found != spatial_variances_.end()) return found->second;
            auto sigma = row(ti.lo);
            return spatial_variances_.emplace(ti.lo, sigma*sigma).first->second;
        }
        const auto low = row(ti.lo);
        if (ti.lo == ti.hi) return low*low;
        return quantlibtorch::time_variance(low, row(ti.hi), ti.weight.value());
    }
  protected:
    Volatility localVolImpl(Time t, Real s) const override {
        RECORD_FUNCTION("qlt::local_vol", std::vector<c10::IValue>());
        const bool cached = quantlibtorch::sample_cache_enabled();
        const auto ti = interval(times_, t);
        const auto xi = cached && xs_.size() == 1
            ? Interval{0, 0, 0.0} : interval(xs_, quantlibtorch::log(s));
        const auto sample = [&](Size i, Size j) {
            if (cached) {
                const auto key = i*xs_.size() + j;
                const auto found = samples_.find(key);
                if (found != samples_.end()) return found->second;
                // Cache graph-connected views only for coefficients actually
                // visited by this engine call; never detach or mutate inputs.
                return samples_.emplace(key, TorchReal(values_.select(0, i).select(0, j))).first->second;
            }
            return TorchReal(values_.select(0, i).select(0, j));
        };
        if (cached) {
            const auto row = [&](Size i) {
                if (xi.lo == xi.hi) return sample(i, xi.lo);
                return (1.0-xi.weight)*sample(i, xi.lo) + xi.weight*sample(i, xi.hi);
            };
            const auto low = row(ti.lo);
            if (ti.lo == ti.hi) return low;
            return (1.0-ti.weight)*low + ti.weight*row(ti.hi);
        }
        const auto low = (1.0-xi.weight)*sample(ti.lo, xi.lo)
                         + xi.weight*sample(ti.lo, xi.hi);
        const auto high = (1.0-xi.weight)*sample(ti.hi, xi.lo)
                          + xi.weight*sample(ti.hi, xi.hi);
        return (1.0-ti.weight)*low + ti.weight*high;
    }
  private:
    struct Interval { Size lo, hi; Real weight; };
    static Interval interval(const std::vector<double>& axis, const Real& x) {
        if (axis.size() == 1 || x <= axis.front()) return {0, 0, 0.0};
        if (x >= axis.back()) return {axis.size()-1, axis.size()-1, 0.0};
        // Reading a value selects an interpolation interval, just like upstream
        // comparisons. Interpolation arithmetic and coefficient views keep AD.
        const Size hi = std::upper_bound(axis.begin(), axis.end(), x.value()) - axis.begin();
        return {hi-1, hi, (x-axis[hi-1])/(axis[hi]-axis[hi-1])};
    }
    at::Tensor values_;
    std::vector<double> times_, xs_;
    mutable std::unordered_map<Size, TorchReal> samples_;
    mutable std::vector<double> sample_spots_;
    mutable at::Tensor grid_lo_, grid_hi_, grid_weights_;
    mutable std::unordered_map<Size, at::Tensor> spatial_rows_, spatial_variances_;
    mutable bool cache_recording_ = false;
};

// Internal verification of the wider step boundary's conservative dispatch.
class ProbeBoundary final : public BoundaryCondition<FdmLinearOp> {
  public:
    void applyBeforeApplying(FdmLinearOp&) const override {}
    void applyBeforeSolving(FdmLinearOp&, Array&) const override {}
    void applyAfterApplying(Array& state) const override { state[0] = 0.125; }
    void applyAfterSolving(Array& state) const override { state[0] = 0.125; }
    void setTime(Time) override {}
};
class DerivedProbeOperator final : public FdmBlackScholesOp {
  public:
    using FdmBlackScholesOp::FdmBlackScholesOp;
};

at::Tensor guarded_step_probe(const at::Tensor& volatility, const at::Tensor& state,
                             const at::Tensor& settings, int64_t kind, bool adjoint, int64_t mode) {
    for (const auto& input : {volatility, state, settings})
        TORCH_CHECK(input.is_cpu() && input.scalar_type() == at::kDouble
                    && input.layout() == at::kStrided, "probe requires dense CPU float64 inputs");
    TORCH_CHECK(volatility.dim() == 2 && volatility.size(0) == 2 && volatility.size(1) == 1
                && state.dim() == 1 && state.numel() >= 3 && settings.dim() == 1 && settings.numel() == 3
                && kind >= 0 && kind <= 3, "invalid guarded step probe inputs");
    TORCH_CHECK(kind == 2 || !settings.requires_grad(), "fixed probe settings must be passive");
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved_date;
    const Date reference(1, January, 2025);
    Settings::instance().evaluationDate() = reference;
    quantlibtorch::AdjointMode adjoint_guard(adjoint);
    quantlibtorch::TensorKernelMode kernel_guard(mode >= 0 ? static_cast<int>(mode) : (adjoint ? 4 : 2));
    const auto local = ext::make_shared<TensorLocalVol>(reference, volatility,
        std::vector<double>{0., 1.}, std::vector<double>{std::log(100.)});
    const auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(100.)),
        Handle<YieldTermStructure>(ext::make_shared<FlatForward>(reference, .01, Actual365Fixed())),
        Handle<YieldTermStructure>(ext::make_shared<FlatForward>(reference, .03, Actual365Fixed())),
        Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(reference, NullCalendar(), .2, Actual365Fixed())),
        Handle<LocalVolTermStructure>(local));
    const auto mesh = ext::make_shared<FdmMesherComposite>(
        ext::make_shared<Uniform1dMesher>(std::log(70.), std::log(130.), state.numel()));
    ext::shared_ptr<FdmLinearOpComposite> map;
    if (kind == 3) map = ext::make_shared<DerivedProbeOperator>(mesh, process, 100., true);
    else map = ext::make_shared<FdmBlackScholesOp>(mesh, process, 100., true);
    DouglasScheme::bc_set boundaries;
    if (kind == 1) boundaries.push_back(ext::make_shared<ProbeBoundary>());
    const auto parameter = [&](int64_t i) {
        return kind == 2 ? TorchReal(settings[i]) : TorchReal(settings[i].item<double>());
    };
    DouglasScheme scheme(parameter(1), map, boundaries);
    scheme.setStep(parameter(0));
    Array values(state.numel());
    quantlibtorch::unpack_array(state, values.begin());
    scheme.step(values, parameter(2));
    return quantlibtorch::pack_array(values.begin(), values.size());
}

// Private lifetime probe: rebind coefficients without mutating their tensors,
// copy an assembled operator, and invalidate it through a failing/fallback
// setTime before trying another step. No production surface cache is involved.
class RebindingProbeVol final : public LocalVolTermStructure {
  public:
    RebindingProbeVol(const Date& reference, at::Tensor value)
    : LocalVolTermStructure(reference, NullCalendar(), Following, Actual365Fixed()), value(std::move(value)) {}
    Date maxDate() const override { return Date::maxDate(); }
    Real minStrike() const override { return 0.; }
    Real maxStrike() const override { return (std::numeric_limits<double>::max)(); }
    at::Tensor value;
    bool fail = false;
  protected:
    Volatility localVolImpl(Time, Real) const override {
        QL_REQUIRE(!fail, "intentional owner lifetime probe failure");
        return TorchReal(value);
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> assembled_owner_sequence(
    const at::Tensor& first, const at::Tensor& second, const at::Tensor& state,
    bool use_owner, int64_t transition, bool adjoint) {
    for (const auto& t : {first, second, state})
        TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided,
                    "owner probe requires dense CPU float64 inputs");
    TORCH_CHECK(first.dim() == 0 && second.dim() == 0 && state.dim() == 1 && state.numel() >= 8
                && transition >= 0 && transition <= 2, "invalid owner probe inputs");
    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved_date;
    const Date reference(1, January, 2025);
    Settings::instance().evaluationDate() = reference;
    quantlibtorch::AdjointMode adjoint_guard(adjoint && use_owner);
    quantlibtorch::TensorKernelMode kernel_guard(use_owner ? 4 : 0);
    const auto local = ext::make_shared<RebindingProbeVol>(reference, first);
    const auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(100.)),
        Handle<YieldTermStructure>(ext::make_shared<FlatForward>(reference, .01, Actual365Fixed())),
        Handle<YieldTermStructure>(ext::make_shared<FlatForward>(reference, .03, Actual365Fixed())),
        Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(reference, NullCalendar(), .2, Actual365Fixed())),
        Handle<LocalVolTermStructure>(local));
    const auto mesh = ext::make_shared<FdmMesherComposite>(
        ext::make_shared<Uniform1dMesher>(std::log(70.), std::log(130.), state.numel()));
    auto op = ext::make_shared<FdmBlackScholesOp>(mesh, process, 100., true);
    const double dt = .04, theta = .6;
    const auto step = [&](FdmBlackScholesOp& map) {
        Array values(state.numel());
        quantlibtorch::unpack_array(state, values.begin());
        const BoundaryConditionSchemeHelper boundaries(DouglasScheme::bc_set{});
        if (!use_owner || !quantlibtorch::try_step_adjoint(map, boundaries, values, dt, theta, .8, 0)) {
            const Array applied = map.apply(values);
            const Array rhs = (values + dt*applied) - (theta*dt)*applied;
            values = map.solve_splitting(0, rhs, (-theta)*dt);
        }
        return quantlibtorch::pack_array(values.begin(), values.size());
    };
    op->setTime(.7, .8);
    auto copy = ext::make_shared<FdmBlackScholesOp>(*op);
    const auto initial = step(*op);
    local->value = second;
    op->setTime(.7, .8);
    const auto rebound = step(*op), copied = step(*copy);
    if (transition == 0) {
        local->fail = true;
        bool caught = false;
        try { op->setTime(.6, .7); } catch (const Error&) { caught = true; }
        TORCH_CHECK(caught, "owner probe expected a sampling failure");
    } else if (transition == 1) {
        quantlibtorch::TensorKernelMode fallback_guard(use_owner ? (adjoint ? 1 : 2) : 0);
        op->setTime(.6, .7);
    } else {
        at::NoGradGuard no_grad;
        op->setTime(.6, .7);
    }
    const auto after_invalidation = step(*op);
    return {initial, rebound, copied, after_invalidation};
}

std::vector<Date> dates(const Ints& serials) {
    std::vector<Date> result;
    for (int64_t serial : serials) {
        TORCH_CHECK(serial >= Date::minDate().serialNumber()
                    && serial <= Date::maxDate().serialNumber(), "date outside QuantLib range");
        result.emplace_back(static_cast<Date::serial_type>(serial));
    }
    return result;
}

Handle<YieldTermStructure> curve(const Date& reference, const Date& maturity,
                                 double flat_rate, const Ints& serials,
                                 const Doubles& discounts) {
    TORCH_CHECK(serials.size() == discounts.size(), "curve dates/discounts length mismatch");
    if (serials.empty())
        return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(reference, flat_rate, Actual365Fixed()));
    auto ds = dates(serials);
    TORCH_CHECK(ds.size() >= 2 && ds.front() == reference && ds.back() >= maturity,
                "discount curve must start at valuation and cover maturity");
    TORCH_CHECK(discounts.get(0) == 1.0, "first discount must equal one");
    std::vector<Real> dfs;
    for (size_t i = 0; i < ds.size(); ++i) {
        TORCH_CHECK(std::isfinite(discounts.get(i)) && discounts.get(i) > 0,
                    "discount factors must be finite and positive");
        TORCH_CHECK(i == 0 || ds[i] > ds[i-1], "curve dates must be strictly increasing");
        dfs.emplace_back(discounts.get(i));
    }
    return Handle<YieldTermStructure>(ext::make_shared<DiscountCurve>(ds, dfs, Actual365Fixed()));
}

#include "convention_bridge.inc"
#include "analytic_bridge.inc"
#include "heston_bridge.inc"
#include "shout_bridge.inc"

at::Tensor fd_vanilla(const at::Tensor& surface, const Doubles& times,
    const Doubles& log_spots, const c10::Dict<std::string, double>& market,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const Doubles& risk_discounts,
    const Ints& yield_dates, const Doubles& yield_discounts,
    const Ints& cash_dates, const Doubles& cash_amounts, const Ints& exercise_serials) {
    RECORD_FUNCTION("qlt::bridge", std::vector<c10::IValue>());
    TORCH_CHECK(surface.device().is_cpu() && surface.scalar_type() == at::kDouble,
                "local_vol must use CPU float64");
    TORCH_CHECK(surface.layout() == at::kStrided && surface.dim() == 2,
                "local_vol must be a dense two-dimensional tensor");
    TORCH_CHECK(at::isfinite(surface).all().item<bool>() && (surface > 0).all().item<bool>(),
                "local_vol must be finite and strictly positive");
    const auto ts = times.vec(), xs = log_spots.vec();
    check_axis(ts, surface.size(0), "times");
    check_axis(xs, surface.size(1), "log_spots");
    TORCH_CHECK(ts.front() >= 0.0, "times must be nonnegative");
    const double spot = market.at("spot"), strike = market.at("strike");
    const double rate = market.at("rate"), yield = market.at("dividend_yield");
    const double mesh_vol = market.at("mesh_volatility");
    for (const auto v : {spot, strike, rate, yield, mesh_vol})
        TORCH_CHECK(std::isfinite(v), "market inputs must be finite");
    TORCH_CHECK(spot > 0 && strike > 0 && mesh_vol > 0, "spot, strike, mesh_volatility must be positive");
    const auto steps = config.at("time_steps"), nodes = config.at("grid_points");
    const auto damping = config.at("damping_steps");
    TORCH_CHECK(steps >= 1 && nodes >= 5 && damping >= 0, "invalid finite-difference grid sizes");
    const auto option_type = config.at("option_type"), exercise_type = config.at("exercise");
    const auto scheme_type = config.at("scheme"), adjoint = config.at("adjoint");
    TORCH_CHECK(option_type == -1 || option_type == 1, "invalid option type");
    TORCH_CHECK(exercise_type >= 0 && exercise_type <= 2, "invalid exercise type");
    TORCH_CHECK(scheme_type >= 0 && scheme_type <= 5, "invalid scheme");
    const auto dividend_model = config.contains("cash_dividend_model") ? config.at("cash_dividend_model") : 0;
    TORCH_CHECK(dividend_model == 0 || dividend_model == 1, "invalid cash dividend model");
    TORCH_CHECK(adjoint == 0 || adjoint == 1, "invalid differentiation mode");
    const auto kernel_mode = config.contains("kernels") ? config.at("kernels") : 0;
    TORCH_CHECK(kernel_mode != 3 || adjoint == 0, "PCR pricing supports mode='full' only");
    Ints endpoint_serials;
    endpoint_serials.push_back(config.at("valuation_date"));
    endpoint_serials.push_back(config.at("maturity_date"));
    const auto endpoints = dates(endpoint_serials);
    const auto reference = endpoints[0], maturity = endpoints[1];
    TORCH_CHECK(maturity > reference, "maturity must follow valuation");

    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved_date;
    Settings::instance().evaluationDate() = reference;
    // Graph-disabled inference can use the same explicit native solve boundary
    // without changing the user's derivative mode for graph-enabled calls.
    quantlibtorch::AdjointMode mode(adjoint != 0 ||
        (!at::GradMode::is_enabled() && kernel_mode != 3 && !surface._fw_grad(0).defined()
         && quantlibtorch::native_adjoint_enabled()));
    quantlibtorch::TensorKernelMode kernels(config.contains("kernels") ? config.at("kernels") : 0);
    quantlibtorch::PriceOnlyMode output(quantlibtorch::price_only_enabled());
    auto risk = curve(reference, maturity, rate, risk_dates, risk_discounts);
    auto dividend_yield = curve(reference, maturity, yield, yield_dates, yield_discounts);
    auto local = ext::make_shared<TensorLocalVol>(reference, surface, ts, xs);
    auto black = ext::make_shared<BlackConstantVol>(reference, NullCalendar(), mesh_vol, Actual365Fixed());
    auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(spot)), dividend_yield, risk,
        Handle<BlackVolTermStructure>(black), Handle<LocalVolTermStructure>(local));

    ext::shared_ptr<Exercise> exercise;
    auto ex_dates = dates(exercise_serials);
    if (exercise_type == 2) {
        TORCH_CHECK(!ex_dates.empty() && ex_dates.front() > reference && ex_dates.back() == maturity,
                    "Bermudan exercise dates must follow valuation and end at maturity");
        for (size_t i = 1; i < ex_dates.size(); ++i)
            TORCH_CHECK(ex_dates[i] > ex_dates[i-1], "exercise dates must be strictly increasing");
        exercise = ext::make_shared<BermudanExercise>(ex_dates);
    } else {
        TORCH_CHECK(ex_dates.empty(), "exercise_dates apply only to Bermudan exercise");
        exercise = exercise_type == 1
            ? ext::static_pointer_cast<Exercise>(ext::make_shared<AmericanExercise>(reference, maturity))
            : ext::static_pointer_cast<Exercise>(ext::make_shared<EuropeanExercise>(maturity));
    }
    TORCH_CHECK(cash_dates.size() == cash_amounts.size(), "cash dividend lengths mismatch");
    auto div_dates = dates(cash_dates);
    DividendSchedule dividends;
    double total_cash = 0.0;
    for (size_t i = 0; i < div_dates.size(); ++i) {
        TORCH_CHECK(div_dates[i] > reference && div_dates[i] <= maturity,
                    "cash dividend dates must be after valuation and on/before maturity");
        TORCH_CHECK(i == 0 || div_dates[i] > div_dates[i-1], "cash dividend dates must increase");
        TORCH_CHECK(std::isfinite(cash_amounts.get(i)) && cash_amounts.get(i) >= 0,
                    "cash dividends must be finite and nonnegative");
        total_cash += cash_amounts.get(i);
        dividends.push_back(ext::make_shared<FixedDividend>(cash_amounts.get(i), div_dates[i]));
    }
    TORCH_CHECK(total_cash < spot, "total cash dividends must be less than spot");
    const auto scheme = scheme_type == 0 ? FdmSchemeDesc::Douglas()
        : scheme_type == 1 ? FdmSchemeDesc::CrankNicolson()
        : scheme_type == 2 ? FdmSchemeDesc::ImplicitEuler()
        : scheme_type == 3 ? FdmSchemeDesc::CraigSneyd()
        : scheme_type == 4 ? FdmSchemeDesc::ModifiedCraigSneyd() : FdmSchemeDesc::Hundsdorfer();
    auto engine = ext::make_shared<FdBlackScholesVanillaEngine>(
        process, dividends, steps, nodes, damping, scheme, true, -Null<Real>(),
        dividend_model == 0 ? FdBlackScholesVanillaEngine::Spot : FdBlackScholesVanillaEngine::Escrowed);
    VanillaOption option(ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(option_type), strike), exercise);
    option.setPricingEngine(engine);
    auto value = option.NPV().tensor();
    TORCH_CHECK(at::isfinite(value).all().item<bool>(), "non-finite finite-difference price");
    return value;
}
at::Tensor fd_basket(const at::Tensor& surface1, const at::Tensor& surface2,
    const at::Tensor& correlation, const Doubles& times1, const Doubles& log_spots1,
    const Doubles& times2, const Doubles& log_spots2,
    const c10::Dict<std::string, double>& market,
    const c10::Dict<std::string, int64_t>& config,
    const Ints& risk_dates, const Doubles& risk_discounts,
    const Ints& yield_dates1, const Doubles& yield_discounts1,
    const Ints& yield_dates2, const Doubles& yield_discounts2,
    const Ints& exercise_serials) {
    RECORD_FUNCTION("qlt::basket_bridge", std::vector<c10::IValue>());
    for (const auto& surface : {surface1, surface2}) {
        TORCH_CHECK(surface.device().is_cpu() && surface.scalar_type() == at::kDouble,
                    "local volatility surfaces must use CPU float64");
        TORCH_CHECK(surface.layout() == at::kStrided && surface.dim() == 2,
                    "local volatility surfaces must be dense two-dimensional tensors");
        TORCH_CHECK(at::isfinite(surface).all().item<bool>() && (surface > 0).all().item<bool>(),
                    "local volatility surfaces must be finite and strictly positive");
    }
    TORCH_CHECK(correlation.device().is_cpu() && correlation.scalar_type() == at::kDouble
                && correlation.layout() == at::kStrided && correlation.dim() == 0,
                "correlation must be a CPU float64 scalar tensor");
    TORCH_CHECK(at::isfinite(correlation).item<bool>() && correlation.abs().item<double>() < 1.,
                "correlation must be finite and strictly between -1 and 1");
    const auto ts1 = times1.vec(), xs1 = log_spots1.vec();
    const auto ts2 = times2.vec(), xs2 = log_spots2.vec();
    check_axis(ts1, surface1.size(0), "times1");
    check_axis(xs1, surface1.size(1), "log_spots1");
    check_axis(ts2, surface2.size(0), "times2");
    check_axis(xs2, surface2.size(1), "log_spots2");
    TORCH_CHECK(ts1.front() >= 0. && ts2.front() >= 0., "times must be nonnegative");
    const double s1 = market.at("spot1"), s2 = market.at("spot2"), strike = market.at("strike");
    const double rate = market.at("rate"), q1 = market.at("dividend_yield1"), q2 = market.at("dividend_yield2");
    const double mesh1 = market.at("mesh_volatility1"), mesh2 = market.at("mesh_volatility2");
    for (const auto value : {s1, s2, strike, rate, q1, q2, mesh1, mesh2})
        TORCH_CHECK(std::isfinite(value), "market inputs must be finite");
    TORCH_CHECK(s1 > 0. && s2 > 0. && strike > 0. && mesh1 > 0. && mesh2 > 0.,
                "spots, strike and mesh volatilities must be positive");
    const auto steps = config.at("time_steps"), nx = config.at("x_grid"), ny = config.at("y_grid");
    TORCH_CHECK(steps >= 1 && nx >= 5 && ny >= 5, "invalid finite-difference grid sizes");
    TORCH_CHECK(config.at("damping_steps") == 0,
                "basket damping is unsupported: multidimensional implicit Euler needs a Krylov adjoint");
    const auto option_type = config.at("option_type"), exercise_type = config.at("exercise");
    const auto scheme_type = config.at("scheme"), adjoint = config.at("adjoint");
    const auto basket_type = config.at("basket"), kernel_mode = config.at("kernels");
    TORCH_CHECK(option_type == -1 || option_type == 1, "invalid option type");
    TORCH_CHECK(exercise_type >= 0 && exercise_type <= 2, "invalid exercise type");
    TORCH_CHECK(scheme_type >= 0 && scheme_type <= 3, "invalid basket scheme");
    TORCH_CHECK(basket_type == 0 || basket_type == 1, "invalid basket payoff");
    TORCH_CHECK(adjoint == 0 || adjoint == 1, "invalid differentiation mode");
    TORCH_CHECK(kernel_mode == 0 || kernel_mode == 4, "basket kernels must be scalar or auto");
    Ints endpoint_serials;
    endpoint_serials.push_back(config.at("valuation_date"));
    endpoint_serials.push_back(config.at("maturity_date"));
    const auto endpoints = dates(endpoint_serials);
    const auto reference = endpoints[0], maturity = endpoints[1];
    TORCH_CHECK(maturity > reference, "maturity must follow valuation");

    const std::lock_guard<std::mutex> lock(pricing_mutex);
    SavedEvaluationDate saved_date;
    Settings::instance().evaluationDate() = reference;
    quantlibtorch::AdjointMode mode(adjoint != 0);
    quantlibtorch::TensorKernelMode kernels(kernel_mode);
    const auto risk = curve(reference, maturity, rate, risk_dates, risk_discounts);
    const auto yield1 = curve(reference, maturity, q1, yield_dates1, yield_discounts1);
    const auto yield2 = curve(reference, maturity, q2, yield_dates2, yield_discounts2);
    const auto local1 = ext::make_shared<TensorLocalVol>(reference, surface1, ts1, xs1);
    const auto local2 = ext::make_shared<TensorLocalVol>(reference, surface2, ts2, xs2);
    // Independent passive Black volatilities fix both meshes. Only the actual
    // local-volatility fields and correlation enter the coefficient graph.
    const auto black1 = ext::make_shared<BlackConstantVol>(reference, NullCalendar(), mesh1, Actual365Fixed());
    const auto black2 = ext::make_shared<BlackConstantVol>(reference, NullCalendar(), mesh2, Actual365Fixed());
    const auto process1 = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(s1)), yield1, risk,
        Handle<BlackVolTermStructure>(black1), Handle<LocalVolTermStructure>(local1));
    const auto process2 = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(s2)), yield2, risk,
        Handle<BlackVolTermStructure>(black2), Handle<LocalVolTermStructure>(local2));

    ext::shared_ptr<Exercise> exercise;
    const auto ex_dates = dates(exercise_serials);
    if (exercise_type == 2) {
        TORCH_CHECK(!ex_dates.empty() && ex_dates.front() > reference && ex_dates.back() == maturity,
                    "Bermudan exercise dates must follow valuation and end at maturity");
        for (size_t i = 1; i < ex_dates.size(); ++i)
            TORCH_CHECK(ex_dates[i] > ex_dates[i-1], "exercise dates must be strictly increasing");
        exercise = ext::make_shared<BermudanExercise>(ex_dates);
    } else {
        TORCH_CHECK(ex_dates.empty(), "exercise_dates apply only to Bermudan exercise");
        exercise = exercise_type == 1
            ? ext::static_pointer_cast<Exercise>(ext::make_shared<AmericanExercise>(reference, maturity))
            : ext::static_pointer_cast<Exercise>(ext::make_shared<EuropeanExercise>(maturity));
    }
    const auto vanilla = ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(option_type), strike);
    const ext::shared_ptr<BasketPayoff> payoff = basket_type == 0
        ? ext::static_pointer_cast<BasketPayoff>(ext::make_shared<MinBasketPayoff>(vanilla))
        : ext::static_pointer_cast<BasketPayoff>(ext::make_shared<MaxBasketPayoff>(vanilla));
    const auto scheme = scheme_type == 0 ? FdmSchemeDesc::Douglas()
        : scheme_type == 1 ? FdmSchemeDesc::CraigSneyd()
        : scheme_type == 2 ? FdmSchemeDesc::ModifiedCraigSneyd() : FdmSchemeDesc::Hundsdorfer();
    auto engine = ext::make_shared<Fd2dBlackScholesVanillaEngine>(
        process1, process2, TorchReal(correlation), nx, ny, steps, 0, scheme, true, -Null<Real>());
    BasketOption option(payoff, exercise);
    option.setPricingEngine(engine);
    auto value = option.NPV().tensor();
    TORCH_CHECK(at::isfinite(value).all().item<bool>(), "non-finite basket finite-difference price");
    return value;
}
} // namespace

namespace quantlibtorch {
namespace { thread_local QuantLib::ext::shared_ptr<QuantLib::FdmMesher> fixed_pricing_mesh; }
QuantLib::ext::shared_ptr<QuantLib::FdmMesher> pricing_mesh_override() { return fixed_pricing_mesh; }
FixedMeshScope::FixedMeshScope(QuantLib::ext::shared_ptr<QuantLib::FdmMesher> mesh)
    : previous_(fixed_pricing_mesh) { fixed_pricing_mesh = std::move(mesh); }
FixedMeshScope::~FixedMeshScope() { fixed_pricing_mesh = previous_; }
namespace { thread_local bool npv_only = false; }
bool price_only() { return npv_only; }
PriceOnlyMode::PriceOnlyMode(bool enabled) : previous_(npv_only) { npv_only = enabled; }
PriceOnlyMode::~PriceOnlyMode() { npv_only = previous_; }

bool sample_local_variances(const QuantLib::LocalVolTermStructure* local_vol,
    QuantLib::Time time, const QuantLib::Array& spots, QuantLib::Array& variance) {
    if (!vector_sampling_enabled() || tensor_kernel_mode() < 2 || !time.is_passive()) return false;
    const auto* surface = dynamic_cast<const TensorLocalVol*>(local_vol);
    if (!surface || spots.empty()) return false;
    for (const auto& spot : spots) if (!spot.is_passive()) return false;
    unpack_array(surface->sampleGrid(time, spots), variance.begin());
    return true;
}
} // namespace quantlibtorch

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("date_adjust(int serial, int calendar, int convention) -> int");
    m.def("date_advance(int serial, int count, int unit, int calendar, int convention, bool end_of_month) -> int");
    m.def("year_fraction(int start, int end, int day_counter) -> float");
    m.def("shout_mesh(float[] parameters, float strike, Dict(str, int) config, int[] risk_dates, float[] risk_discounts, int[] yield_dates, float[] yield_discounts, int[] cash_dates, float[] cash_amounts, int[] exercise_dates) -> Tensor");
    m.def("fd_shout(Tensor parameters, float strike, Dict(str, int) config, int[] risk_dates, Tensor risk_discounts, int[] yield_dates, Tensor yield_discounts, int[] cash_dates, float[] cash_amounts, int[] exercise_dates, float[] log_spots) -> Tensor");
    m.def("heston_mesh(float[] parameters, float strike, Dict(str, int) config, int[] risk_dates, float[] risk_discounts, int[] yield_dates, float[] yield_discounts, int[] cash_dates, float[] cash_amounts, int[] exercise_dates) -> (Tensor, Tensor)");
    m.def("fd_heston(Tensor parameters, float strike, Dict(str, int) config, int[] risk_dates, Tensor risk_discounts, int[] yield_dates, Tensor yield_discounts, int[] cash_dates, float[] cash_amounts, int[] exercise_dates, float[] log_spots, float[] variances) -> Tensor");
    m.def("analytic_european(Tensor market, int reference, int maturity, int option_type, "
          "int[] risk_dates, Tensor risk_discounts, int[] yield_dates, Tensor yield_discounts, "
          "int[] cash_dates, Tensor cash_amounts) -> Tensor");
    m.def("_assembled_owner_sequence(Tensor first, Tensor second, Tensor state, bool use_owner=True, int transition=0, bool adjoint=False) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_guarded_step_probe(Tensor volatility, Tensor state, Tensor settings, int kind, bool adjoint, int mode=-1) -> Tensor");
    m.def("_evaluation_date() -> int");
    m.def("fd_vanilla(Tensor local_vol, float[] times, float[] log_spots, Dict(str, float) market, "
          "Dict(str, int) config, int[] risk_dates, float[] risk_discounts, int[] yield_dates, "
          "float[] yield_discounts, int[] cash_dates, float[] cash_amounts, int[] exercise_dates) -> Tensor");
    m.def("fd_basket(Tensor local_vol1, Tensor local_vol2, Tensor correlation, "
          "float[] times1, float[] log_spots1, float[] times2, float[] log_spots2, "
          "Dict(str, float) market, Dict(str, int) config, int[] risk_dates, float[] risk_discounts, "
          "int[] yield_dates1, float[] yield_discounts1, int[] yield_dates2, float[] yield_discounts2, "
          "int[] exercise_dates) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("date_adjust", TORCH_FN(date_adjust));
    m.impl("date_advance", TORCH_FN(date_advance));
    m.impl("year_fraction", TORCH_FN(year_fraction));
    m.impl("shout_mesh", TORCH_FN(shout_mesh));
    m.impl("fd_shout", TORCH_FN(fd_shout));
    m.impl("heston_mesh", TORCH_FN(heston_mesh));
    m.impl("fd_heston", TORCH_FN(fd_heston));
    m.impl("analytic_european", TORCH_FN(analytic_european));
    m.impl("_assembled_owner_sequence", TORCH_FN(assembled_owner_sequence));
    m.impl("_guarded_step_probe", TORCH_FN(guarded_step_probe));
    m.impl("_evaluation_date", TORCH_FN(evaluation_date));
    m.impl("fd_vanilla", TORCH_FN(fd_vanilla));
    m.impl("fd_basket", TORCH_FN(fd_basket));
}
