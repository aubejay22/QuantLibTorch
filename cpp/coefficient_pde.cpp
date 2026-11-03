// Tensor coefficient adapters. Spatial discretization, splitting solves, ADI
// stages, payoff averaging, exercise and interpolation belong to QuantLib 1.40.
#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/financial_curves.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <ql/exercise.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/predefined1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondordermixedderivativeop.hpp>
#include <ql/methods/finitedifferences/solvers/fdm2dimsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmndimsolver.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <cmath>
#include <limits>

namespace {
using namespace QuantLib;
using Doubles = c10::List<double>;
using Ints = c10::List<int64_t>;
using Axes = c10::List<Doubles>;

void tensor_check(const at::Tensor& t) {
    TORCH_CHECK(t.is_cpu() && t.scalar_type() == at::kDouble && t.layout() == at::kStrided,
                "coefficient PDE inputs must be dense CPU float64 tensors");
    TORCH_CHECK(at::isfinite(t).all().item<bool>(), "coefficient PDE inputs must be finite");
}

Array as_array(const at::Tensor& t) {
    auto flat = t.reshape({-1});
    Array a(flat.numel());
    quantlibtorch::unpack_array(flat, a.begin());
    return a;
}

ext::shared_ptr<FdmMesherComposite> make_mesh(const Axes& axes) {
    RECORD_FUNCTION("qlt::coefficient_mesh", std::vector<c10::IValue>());
    TORCH_CHECK(!axes.empty(), "axes must not be empty");
    std::vector<ext::shared_ptr<Fdm1dMesher>> meshers;
    Size nodes = 1;
    for (const auto& axis_ref : axes) {
        const Doubles axis = axis_ref;
        TORCH_CHECK(axis.size() >= 2, "each axis needs at least two nodes");
        TORCH_CHECK(nodes <= (std::numeric_limits<Size>::max)()/axis.size(), "mesh size overflow");
        nodes *= axis.size();
        std::vector<Real> x;
        for (size_t i=0; i<axis.size(); ++i) {
            TORCH_CHECK(std::isfinite(axis.get(i)) && (i == 0 || axis.get(i)>axis.get(i-1)),
                        "axes must be finite and strictly increasing");
            x.emplace_back(axis.get(i));
        }
        meshers.push_back(ext::make_shared<Predefined1dMesher>(x));
    }
    return ext::make_shared<FdmMesherComposite>(meshers);
}

void validate(const at::Tensor& drift, const at::Tensor& factor, const at::Tensor& killing,
              const Doubles& times, const ext::shared_ptr<FdmMesher>& mesh,
              double horizon, int64_t steps, bool equity) {
    tensor_check(drift); tensor_check(factor); tensor_check(killing);
    const int64_t d = mesh->layout()->dim().size(), n = mesh->layout()->size();
    TORCH_CHECK(std::isfinite(horizon) && horizon>0 && steps>0, "positive horizon and steps required");
    TORCH_CHECK(!times.empty(), "times must not be empty");
    for (size_t i=0; i<times.size(); ++i)
        TORCH_CHECK(std::isfinite(times.get(i)) && times.get(i)>=0
                    && (i == 0 || times.get(i)>times.get(i-1)), "times must increase and be nonnegative");
    TORCH_CHECK(times.size()==1 || (times.get(0)==0 && times.get(times.size()-1)>=horizon),
                "time samples must cover [0, horizon]");
    const int64_t nt = times.size();
    TORCH_CHECK(drift.dim()==3 && drift.size(0)==nt && drift.size(1)==n && drift.size(2)==d-equity,
                "drift shape must be [time, nodes, directions] (latent directions for equity)");
    TORCH_CHECK(factor.dim()==4 && factor.size(0)==nt && factor.size(1)==n && factor.size(2)==d
                && factor.size(3)>0, "diffusion shape must be [time, nodes, directions, noise factors]");
    TORCH_CHECK(killing.dim()==2 && killing.size(0)==nt && killing.size(1)==n,
                "killing shape must be [time, nodes]");
}

FdmSchemeDesc scheme_desc(int64_t scheme) {
    switch (scheme) {
      case 0: return FdmSchemeDesc::Douglas();
      case 1: return FdmSchemeDesc::Hundsdorfer();
      case 2: return FdmSchemeDesc::ModifiedCraigSneyd();
      case 3: return FdmSchemeDesc::CraigSneyd();
      default: TORCH_CHECK(false, "invalid coefficient PDE scheme");
    }
}

// Only this model adapter is new. There are d directional maps and d(d-1)/2
// original QuantLib mixed stencils, independent of the number of latent states.
class TensorDiffusionOp final : public FdmLinearOpComposite {
  public:
    TensorDiffusionOp(const ext::shared_ptr<FdmMesher>& mesh, at::Tensor drift,
                     const at::Tensor& factor, at::Tensor killing, const Doubles& times,
                     ext::shared_ptr<YieldTermStructure> risk = {},
                     ext::shared_ptr<YieldTermStructure> dividend = {})
    : drift_(std::move(drift)), covariance_(at::matmul(factor, factor.transpose(-1,-2))),
      killing_(std::move(killing)), times_(times.vec()), risk_(std::move(risk)),
      dividend_(std::move(dividend)) {
        RECORD_FUNCTION("qlt::coefficient_spatial_bases", std::vector<c10::IValue>());
        TORCH_CHECK(at::isfinite(covariance_).all().item<bool>(), "diffusion covariance overflow");
        const Size d = mesh->layout()->dim().size();
        for (Size i=0; i<d; ++i) {
            first_.emplace_back(i, mesh);
            second_.emplace_back(i, mesh);
            maps_.emplace_back(i, mesh);
            for (Size j=i+1; j<d; ++j) {
                pairs_.emplace_back(i,j);
                cross_basis_.emplace_back(i,j,mesh);
            }
        }
        if (risk_) {
            std::vector<double> mask(mesh->layout()->size(), 1.0);
            for (const auto& it : *mesh->layout())
                if (it.coordinates()[0]==0 || it.coordinates()[0]+1==mesh->layout()->dim()[0])
                    mask[it.index()] = 0.0;
            interior_ = at::tensor(mask, factor.options().requires_grad(false));
        }
    }
    Size size() const override { return maps_.size(); }
    void setTime(Time t1, Time t2) override {
        RECORD_FUNCTION("qlt::coefficient_assembly", std::vector<c10::IValue>());
        QL_REQUIRE(t1<=t2, "invalid coefficient time interval");
        const double time = (0.5*(t1+t2)).value();
        const auto b = sample(drift_, time), c = sample(covariance_, time);
        auto k = sample(killing_, time);
        Real rate=0.0, yield=0.0;
        if (risk_) {
            rate = risk_->forwardRate(t1,t2,Continuous).rate();
            yield = dividend_->forwardRate(t1,t2,Continuous).rate();
            k = rate.tensor().expand_as(k);
        }
        for (Size i=0; i<size(); ++i) {
            const auto variance = c.select(1,i).select(1,i);
            // Match FdmHestonEquityPart's linear-in-spot endpoint closure:
            // remove the Ito variance drift correction at the two stock edges.
            const auto drift = risk_ && i==0 ? (rate-yield).tensor()-0.5*variance*interior_
                : b.select(1, risk_ ? i-1 : i);
            maps_[i].axpyb(as_array(drift), first_[i],
                second_[i].mult(as_array(0.5*variance)), as_array(-k/static_cast<double>(size())));
        }
        cross_.clear();
        for (Size p=0; p<pairs_.size(); ++p) {
            const auto [i,j] = pairs_[p];
            // The symmetric 1/2 trace contributes C_ij once, not C_ij/2.
            cross_.push_back(cross_basis_[p].mult(as_array(c.select(1,i).select(1,j))));
        }
    }
    Array apply(const Array& a) const override {
        auto result = apply_mixed(a);
        for (const auto& map : maps_) result += map.apply(a);
        return result;
    }
    Array apply_mixed(const Array& a) const override {
        Array result(a.size(),0.0);
        for (const auto& map : cross_) result += map.apply(a);
        return result;
    }
    Array apply_direction(Size i, const Array& a) const override { return maps_.at(i).apply(a); }
    Array solve_splitting(Size i, const Array& a, Real s) const override {
        return maps_.at(i).solve_splitting(a,s,1.0);
    }
    Array preconditioner(const Array& a, Real s) const override {
        auto result=a;
        for (Size i=0; i<size(); ++i) result=solve_splitting(i,result,s);
        return result;
    }
  private:
    at::Tensor sample(const at::Tensor& field, double t) const {
        if (times_.size()==1 || t<=times_.front()) return field.select(0,0);
        if (t>=times_.back()) return field.select(0,times_.size()-1);
        const Size hi=std::upper_bound(times_.begin(),times_.end(),t)-times_.begin();
        const double w=(t-times_[hi-1])/(times_[hi]-times_[hi-1]);
        return (1-w)*field.select(0,hi-1)+w*field.select(0,hi);
    }
    at::Tensor drift_, covariance_, killing_, interior_;
    std::vector<double> times_;
    ext::shared_ptr<YieldTermStructure> risk_, dividend_;
    std::vector<FirstDerivativeOp> first_;
    std::vector<SecondDerivativeOp> second_;
    std::vector<TripleBandLinearOp> maps_;
    std::vector<SecondOrderMixedDerivativeOp> cross_basis_;
    std::vector<NinePointLinearOp> cross_;
    std::vector<std::pair<Size,Size>> pairs_;
};

at::Tensor coefficient_rollback(const at::Tensor& drift, const at::Tensor& factor,
    const at::Tensor& killing, const at::Tensor& terminal, const Axes& axes,
    const Doubles& times, double horizon, int64_t steps, int64_t scheme, bool adjoint,
    int64_t kernels) {
    RECORD_FUNCTION("qlt::coefficient_rollback", std::vector<c10::IValue>());
    TORCH_CHECK(kernels==0 || kernels==4, "kernels must be scalar or auto");
    const auto mesh=make_mesh(axes);
    validate(drift,factor,killing,times,mesh,horizon,steps,false);
    tensor_check(terminal);
    TORCH_CHECK(terminal.dim()==1 && terminal.numel()==mesh->layout()->size(), "terminal must have shape [nodes]");
    quantlibtorch::AdjointMode mode(adjoint);
    quantlibtorch::TensorKernelMode kernel_mode(static_cast<int>(kernels));
    auto op=ext::make_shared<TensorDiffusionOp>(mesh,drift,factor,killing,times);
    auto state=as_array(terminal);
    FdmBackwardSolver(op,{}, {},scheme_desc(scheme)).rollback(state,horizon,0.0,steps,0);
    auto result=quantlibtorch::pack_array(state.begin(),state.size());
    TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite coefficient PDE solution");
    return result;
}

ext::shared_ptr<YieldTermStructure> curve(const Date& ref, const Date& end, double rate,
    const Ints& dates, const Doubles& discounts) {
    TORCH_CHECK(std::isfinite(rate), "rate must be finite");
    if (dates.empty()) {
        TORCH_CHECK(discounts.empty(), "curve dates missing");
        return ext::make_shared<FlatForward>(ref,rate,Actual365Fixed());
    }
    TORCH_CHECK(dates.size()==discounts.size() && dates.size()>=2
                && dates.get(0)==ref.serialNumber() && dates.get(dates.size()-1)>=end.serialNumber()
                && discounts.get(0)==1.0, "invalid discount curve coverage");
    std::vector<Date> ds;
    std::vector<DiscountFactor> dfs;
    for (size_t i=0; i<dates.size(); ++i) {
        TORCH_CHECK(std::isfinite(discounts.get(i)) && discounts.get(i)>0
                    && (i==0 || dates.get(i)>dates.get(i-1)), "invalid discount curve");
        ds.emplace_back(dates.get(i)); dfs.emplace_back(discounts.get(i));
    }
    return ext::make_shared<DiscountCurve>(ds,dfs,Actual365Fixed());
}

at::Tensor latent_price(const at::Tensor& drift, const at::Tensor& factor,
    const at::Tensor& initial, const Axes& axes, const Doubles& times,
    double strike, int64_t reference, int64_t maturity, const at::Tensor& rate, const at::Tensor& yield,
    const Ints& risk_dates, const at::Tensor& risk_dfs, const Ints& yield_dates,
    const at::Tensor& yield_dfs, int64_t exercise_kind, const Ints& exercise_dates,
    int64_t option_type, int64_t steps, int64_t scheme, bool adjoint, int64_t kernels) {
    RECORD_FUNCTION("qlt::latent_price", std::vector<c10::IValue>());
    if (adjoint) for (const auto& input : {rate,yield,risk_dfs,yield_dfs})
        TORCH_CHECK(!input._fw_grad(0).defined(), "equation adjoint does not support forward tangents; use full AD");
    TORCH_CHECK(axes.size()==2 || axes.size()==3, "financial interpolation supports one or two latent states");
    TORCH_CHECK(kernels==0 || kernels==4, "kernels must be scalar or auto");
    const Date ref(reference), end(maturity);
    TORCH_CHECK(end>ref && std::isfinite(strike) && strike>0, "invalid maturity or strike");
    const double horizon=Actual365Fixed().yearFraction(ref,end).value();
    const auto mesh=make_mesh(axes);
    const auto killing=at::zeros({static_cast<int64_t>(times.size()),static_cast<int64_t>(mesh->layout()->size())},factor.options());
    validate(drift,factor,killing,times,mesh,horizon,steps,true);
    tensor_check(initial);
    TORCH_CHECK(initial.dim()==1 && initial.numel()==axes.size(), "initial state must have shape [directions]");
    for (size_t i=0; i<axes.size(); ++i) {
        const auto axis=axes.get(i);
        const double x=initial[i].item<double>();
        // MultiCubicSpline rejects its upper endpoint. Use a common strict
        // interior contract for both supported financial interpolation paths.
        TORCH_CHECK(axis.size()>=4 && x>axis.get(0) && x<axis.get(axis.size()-1),
                    "financial axes need four nodes and initial state strictly inside");
    }
    TORCH_CHECK(option_type==1 || option_type==-1, "invalid option type");
    ext::shared_ptr<Exercise> exercise;
    if (exercise_kind==0) exercise=ext::make_shared<EuropeanExercise>(end);
    else if (exercise_kind==1) exercise=ext::make_shared<AmericanExercise>(ref,end);
    else {
        TORCH_CHECK(exercise_kind==2 && !exercise_dates.empty()
            && exercise_dates.get(0)>reference && exercise_dates.get(exercise_dates.size()-1)==maturity,
            "invalid Bermudan exercise dates");
        std::vector<Date> ds;
        for (size_t i=0; i<exercise_dates.size(); ++i) {
            TORCH_CHECK(i==0 || exercise_dates.get(i)>exercise_dates.get(i-1), "exercise dates must increase");
            ds.emplace_back(exercise_dates.get(i));
        }
        exercise=ext::make_shared<BermudanExercise>(ds);
    }
    TORCH_CHECK(exercise_kind==2 || exercise_dates.empty(), "exercise dates require Bermudan exercise");
    quantlibtorch::AdjointMode mode(adjoint);
    quantlibtorch::TensorKernelMode kernel_mode(static_cast<int>(kernels));
    const auto risk=quantlibtorch::financial_curve(ref,end,rate,risk_dates,risk_dfs);
    const auto dividend=quantlibtorch::financial_curve(ref,end,yield,yield_dates,yield_dfs);
    const auto op=ext::make_shared<TensorDiffusionOp>(mesh,drift,factor,killing,times,risk,dividend);
    const auto payoff=ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(option_type),strike);
    const auto calculator=ext::make_shared<FdmLogInnerValue>(payoff,mesh,0);
    const auto condition=FdmStepConditionComposite::vanillaComposite({},exercise,mesh,calculator,ref,Actual365Fixed());
    const FdmSolverDesc desc={mesh,{},condition,calculator,horizon,static_cast<Size>(steps),0};
    std::vector<Real> x;
    for (int64_t i=0; i<initial.numel(); ++i) x.emplace_back(initial[i]);
    // Existing solver classes supply payoff averaging, theta snapshot, rollback
    // and interpolation. No new pricing engine or time-stepping implementation.
    const Real price=axes.size()==2 ? Fdm2DimSolver(desc,scheme_desc(scheme),op).interpolateAt(x[0],x[1])
        : FdmNdimSolver<3>(desc,scheme_desc(scheme),op).interpolateAt(x);
    TORCH_CHECK(std::isfinite(price.value()), "non-finite latent price");
    return price.tensor();
}

// Internal financial-generator validation: calls the same operator as pricing,
// without payoff/interpolation effects. No alternative discretization is used.
at::Tensor equity_generator_probe(const at::Tensor& drift, const at::Tensor& factor,
    const at::Tensor& state, const Axes& axes, const Doubles& times,
    double t1, double t2, double rate, double yield) {
    TORCH_CHECK(std::isfinite(t1) && std::isfinite(t2) && t1>=0 && t2>t1,
                "invalid probe time interval");
    const auto mesh=make_mesh(axes);
    const auto killing=at::zeros({static_cast<int64_t>(times.size()),static_cast<int64_t>(mesh->layout()->size())},factor.options());
    validate(drift,factor,killing,times,mesh,t2,1,true);
    tensor_check(state);
    TORCH_CHECK(state.dim()==1 && state.numel()==mesh->layout()->size(), "state must have shape [nodes]");
    quantlibtorch::AdjointMode mode(false);
    quantlibtorch::TensorKernelMode kernels(4);
    const Date ref(1,January,2025), end(1,January,2026);
    TensorDiffusionOp op(mesh,drift,factor,killing,times,
        curve(ref,end,rate,Ints(),Doubles()),curve(ref,end,yield,Ints(),Doubles()));
    op.setTime(t1,t2);
    auto result=op.apply(as_array(state));
    return quantlibtorch::pack_array(result.begin(),result.size());
}
} // namespace

TORCH_LIBRARY_FRAGMENT(quantlibtorch,m) {
    m.def("_coefficient_rollback(Tensor drift, Tensor diffusion, Tensor killing, Tensor terminal, float[][] axes, float[] times, float horizon, int steps, int scheme, bool adjoint, int kernels) -> Tensor");
    m.def("_latent_price(Tensor drift, Tensor diffusion, Tensor initial, float[][] axes, float[] times, float strike, int reference, int maturity, Tensor rate, Tensor yield, int[] risk_dates, Tensor risk_dfs, int[] yield_dates, Tensor yield_dfs, int exercise, int[] exercise_dates, int option_type, int steps, int scheme, bool adjoint, int kernels) -> Tensor");
    m.def("_equity_generator_probe(Tensor drift, Tensor diffusion, Tensor state, float[][] axes, float[] times, float t1, float t2, float rate, float yield) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch,CompositeImplicitAutograd,m) {
    m.impl("_coefficient_rollback",TORCH_FN(coefficient_rollback));
    m.impl("_latent_price",TORCH_FN(latent_price));
    m.impl("_equity_generator_probe",TORCH_FN(equity_generator_probe));
}
