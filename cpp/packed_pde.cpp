// Packed coefficient/RHS adapter around QuantLib v1.40 spatial operators,
// generated unchanged ADI stage bodies, and FiniteDifferenceModel scheduler.
// QuantLib payoff averaging and spline interpolation remain the financial
// boundaries. No stock-double QuantLib library is linked.
#include <quantlibtorch/tensor_kernels.hpp>
#include <quantlibtorch/financial_curves.hpp>
#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/thomas_arithmetic.hpp>
#include <quantlibtorch/parallel_solve.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/exercise.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmbermudanstepcondition.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdividendhandler.hpp>
#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/methods/finitedifferences/finitedifferencemodel.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/predefined1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondordermixedderivativeop.hpp>
#include <ql/methods/finitedifferences/solvers/fdmndimsolver.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <torch/custom_class.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace quantlibtorch { namespace packed {
using namespace QuantLib;
using Doubles = c10::List<double>;
using Ints = c10::List<int64_t>;
using Axes = c10::List<Doubles>;

at::Tensor indices(const Size* p, int64_t n) {
    std::vector<int64_t> values(n);
    for (int64_t i=0; i<n; ++i) values[i]=p ? p[i] : i;
    return at::tensor(values, at::TensorOptions().dtype(at::kLong));
}

struct BandBasis : TripleBandLinearOp {
    explicit BandBasis(const TripleBandLinearOp& op) : TripleBandLinearOp(op) {}
    at::Tensor bands() const {
        const auto n=mesher_->layout()->size();
        return at::stack({pack_array(lower_.get(),n),pack_array(diag_.get(),n),pack_array(upper_.get(),n)});
    }
    at::Tensor neighbors() const {
        const auto n=mesher_->layout()->size();
        return at::stack({indices(i0_.get(),n),indices(nullptr,n),indices(i2_.get(),n)});
    }
    at::Tensor permutation() const { return indices(reverseIndex_.get(),mesher_->layout()->size()); }
};
struct MixedBasis : NinePointLinearOp {
    explicit MixedBasis(const NinePointLinearOp& op) : NinePointLinearOp(op) {}
    at::Tensor bands() const {
        const auto n=mesher_->layout()->size();
        return at::stack({pack_array(a00_.get(),n),pack_array(a01_.get(),n),pack_array(a02_.get(),n),
            pack_array(a10_.get(),n),pack_array(a11_.get(),n),pack_array(a12_.get(),n),
            pack_array(a20_.get(),n),pack_array(a21_.get(),n),pack_array(a22_.get(),n)});
    }
    at::Tensor neighbors() const {
        const auto n=mesher_->layout()->size();
        return at::stack({indices(i00_.get(),n),indices(i01_.get(),n),indices(i02_.get(),n),
            indices(i10_.get(),n),indices(nullptr,n),indices(i12_.get(),n),
            indices(i20_.get(),n),indices(i21_.get(),n),indices(i22_.get(),n)});
    }
};

struct Geometry {
    ext::shared_ptr<FdmMesherComposite> mesh;
    std::vector<std::vector<Real>> axes;
    std::vector<int64_t> lengths;
    std::vector<at::Tensor> first, second, direction_indices, permutation, inverse, mixed, mixed_indices;
    std::vector<std::pair<int64_t,int64_t>> pairs;
    at::Tensor interior;
    int64_t n=1, d;
    at::Device device;
    Geometry(const Axes& input, at::Device dev) : d(input.size()), device(dev) {
        TORCH_CHECK(d>0, "axes must not be empty");
        std::vector<ext::shared_ptr<Fdm1dMesher>> meshers;
        for (const auto& entry : input) {
            const Doubles a=entry;
            TORCH_CHECK(a.size()>=2 && n<=(std::numeric_limits<int64_t>::max)()/a.size(), "invalid mesh size");
            std::vector<Real> x;
            for (size_t j=0; j<a.size(); ++j) {
                TORCH_CHECK(std::isfinite(a.get(j)) && (j==0 || a.get(j)>a.get(j-1)), "axes must increase and be finite");
                x.emplace_back(a.get(j));
            }
            axes.push_back(x); lengths.push_back(a.size()); n*=a.size();
            meshers.push_back(ext::make_shared<Predefined1dMesher>(x));
        }
        mesh=ext::make_shared<FdmMesherComposite>(meshers);
        for (int64_t i=0; i<d; ++i) {
            const BandBasis dx(FirstDerivativeOp(i,mesh)), dxx(SecondDerivativeOp(i,mesh));
            first.push_back(dx.bands().to(device)); second.push_back(dxx.bands().to(device));
            direction_indices.push_back(dx.neighbors().to(device));
            auto p=dx.permutation();
            permutation.push_back(p.to(device)); inverse.push_back(at::argsort(p).to(device));
            // Splitting into independent lines is legal only with these exact
            // upstream zero endpoint coefficients. Never mask an active edge.
            for (const auto& basis : {dx.bands(),dxx.bands()}) {
                auto ordered=basis.index_select(1,p).reshape({3,-1,lengths[i]});
                TORCH_CHECK((ordered[0].select(1,0)==0).all().item<bool>() &&
                    (ordered[2].select(1,lengths[i]-1)==0).all().item<bool>(), "coupled line endpoints cannot be batched");
            }
            for (int64_t j=i+1; j<d; ++j) {
                const MixedBasis cross(SecondOrderMixedDerivativeOp(i,j,mesh));
                pairs.emplace_back(i,j); mixed.push_back(cross.bands().to(device));
                mixed_indices.push_back(cross.neighbors().to(device));
            }
        }
        std::vector<double> mask(n,1.);
        for (const auto& it : *mesh->layout())
            if (it.coordinates()[0]==0 || it.coordinates()[0]+1==lengths[0]) mask[it.index()]=0.;
        interior=at::tensor(mask,at::TensorOptions().dtype(at::kDouble)).to(device);
    }
};

void check(const at::Tensor& t, at::Device device) {
    TORCH_CHECK(t.device()==device && t.scalar_type()==at::kDouble && t.layout()==at::kStrided,
        "packed inputs must be dense float64 on the plan device");
    TORCH_CHECK(at::isfinite(t).all().item<bool>(), "packed inputs must be finite");
}
void check_times(const Doubles& times, double horizon) {
    TORCH_CHECK(!times.empty() && std::isfinite(horizon) && horizon>0, "invalid horizon or empty times");
    for (size_t i=0; i<times.size(); ++i)
        TORCH_CHECK(std::isfinite(times.get(i)) && times.get(i)>=0 && (i==0 || times.get(i)>times.get(i-1)), "times must increase and be nonnegative");
    TORCH_CHECK(times.size()==1 || (times.get(0)==0 && times.get(times.size()-1)>=horizon), "times must cover horizon");
}
void check_fields(const at::Tensor& b, const at::Tensor& l, const at::Tensor& k,
    const Geometry& g, int64_t nt, bool equity) {
    check(b,g.device); check(l,g.device); check(k,g.device);
    TORCH_CHECK(b.dim()==3 && b.size(0)==nt && b.size(1)==g.n && b.size(2)==g.d-equity, "invalid drift shape");
    TORCH_CHECK(l.dim()==4 && l.size(0)==nt && l.size(1)==g.n && l.size(2)==g.d && l.size(3)>0, "invalid diffusion shape");
    TORCH_CHECK(k.dim()==2 && k.size(0)==nt && k.size(1)==g.n, "invalid killing shape");
}

// The CUDA prototype uses the existing tensor PCR kernel on independent
// directional lines. Its restricted numerical domain and backward residual
// are checked explicitly; no unsupported system is silently approximated.
at::Tensor solve_bands(const at::Tensor& lo,const at::Tensor& di,const at::Tensor& up,
    const at::Tensor& rhs,bool transpose=false) {
    if (di.is_cpu()) return thomas_solve(lo,di,up,rhs);
    auto result=tridiagonal_solve_pcr(lo,di,up,rhs,false);
    {
        at::AutoGradMode validation(false);
        const auto n=di.size(-1);
        const auto zero=at::zeros_like(result.slice(-2,0,1));
        const auto left=at::cat({zero,lo.unsqueeze(-1)*result.slice(-2,0,n-1)},-2);
        const auto right=at::cat({up.unsqueeze(-1)*result.slice(-2,1,n),zero},-2);
        const auto center=di.unsqueeze(-1)*result;
        const auto residual=(left+center+right-rhs).abs();
        const auto scale=1.+rhs.abs()+left.abs()+center.abs()+right.abs();
        auto valid=at::isfinite(result).all() & (residual<=1e-10*scale).all();
        if (!transpose) {
            const auto z=at::zeros_like(di.slice(-1,0,1));
            const auto off=at::cat({z,lo.abs()},-1)+at::cat({up.abs(),z},-1);
            valid=valid & (di.abs()>off).all();
        }
        TORCH_CHECK(valid.item<bool>(), "CUDA packed solve requires strictly row-diagonally-dominant forward systems and a finite scaled residual <= 1e-10; use CPU for other systems");
    }
    return result;
}

// The custom numerical boundary retains original shared bands and all RHS.
// Its backward solves A^T lambda=g and sums -lambda*x over the contract axis.
class SolveAdjoint : public torch::autograd::Function<SolveAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor lo,
        at::Tensor di, at::Tensor up, at::Tensor rhs) {
        auto x=solve_bands(lo,di,up,rhs);
        ctx->save_for_backward({lo,di,up,x});
        return x;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list gradients) {
        TORCH_CHECK(!at::GradMode::is_enabled(), "packed equation adjoint supports first derivatives only; use full AD");
        TORCH_CHECK(!gradients[0]._fw_grad(0).defined(), "packed equation adjoint does not support forward tangents");
        const auto s=ctx->get_saved_variables();
        const auto lambda=solve_bands(s[2],s[1],s[0],gradients[0],true);
        const auto& x=s[3]; const auto n=x.size(-2);
        return {-(lambda.slice(-2,1,n)*x.slice(-2,0,n-1)).sum(-1),
            -(lambda*x).sum(-1),-(lambda.slice(-2,0,n-1)*x.slice(-2,1,n)).sum(-1),lambda};
    }
};

at::Tensor apply_stencil(const at::Tensor& c, const at::Tensor& idx, const at::Tensor& u) {
    auto result=c[0].unsqueeze(-1)*u.index_select(0,idx[0]);
    for (int64_t j=1; j<c.size(0); ++j) result=result+c[j].unsqueeze(-1)*u.index_select(0,idx[j]);
    return result;
}

class PackedOperator {
  public:
    PackedOperator(const Geometry& geometry, at::Tensor b, const at::Tensor& l, at::Tensor k,
        const Doubles& times, bool adjoint, ext::shared_ptr<YieldTermStructure> risk={},
        ext::shared_ptr<YieldTermStructure> dividend={})
    : g_(geometry), b_(std::move(b)), c_(at::matmul(l,l.transpose(-1,-2))), k_(std::move(k)),
      times_(times.vec()), adjoint_(adjoint), risk_(std::move(risk)), dividend_(std::move(dividend)) {
        if (adjoint_) for (const auto& t : {b_,l,k_}) TORCH_CHECK(!t._fw_grad(0).defined(),
            "packed equation adjoint does not support forward tangents; use full AD");
        TORCH_CHECK(at::isfinite(c_).all().item<bool>(), "diffusion covariance overflow");
    }
    Size size() const { return g_.d; }
    void setTime(double t1,double t2) {
        RECORD_FUNCTION("qlt::packed_coefficient_assembly",std::vector<c10::IValue>());
        auto b=sample(b_,.5*(t1+t2)), c=sample(c_,.5*(t1+t2)), k=sample(k_,.5*(t1+t2));
        auto rate=at::zeros({},b.options()), yield=at::zeros({},b.options());
        if (risk_) {
            rate=risk_->forwardRate(t1,t2,Continuous).rate().tensor().to(g_.device);
            yield=dividend_->forwardRate(t1,t2,Continuous).rate().tensor().to(g_.device);
            k=rate.expand_as(k);
        }
        directions_.clear(); mixed_.clear();
        for (int64_t i=0; i<g_.d; ++i) {
            auto v=c.select(1,i).select(1,i);
            auto drift=risk_ && i==0 ? (rate-yield)-.5*v*g_.interior : b.select(1,risk_ ? i-1 : i);
            auto base=g_.second[i]*(.5*v).unsqueeze(0)+drift.unsqueeze(0)*g_.first[i];
            const auto kill=-k/static_cast<double>(g_.d);
            directions_.push_back(at::stack({base[0],base[1]+kill,base[2]}));
        }
        for (size_t p=0; p<g_.pairs.size(); ++p) {
            const auto [i,j]=g_.pairs[p];
            mixed_.push_back(g_.mixed[p]*c.select(1,i).select(1,j).unsqueeze(0));
        }
    }
    at::Tensor apply(const at::Tensor& u) const {
        auto result=apply_mixed(u);
        for (size_t i=0; i<directions_.size(); ++i) result=result+apply_direction(i,u);
        return result;
    }
    at::Tensor apply_mixed(const at::Tensor& u) const {
        auto result=at::zeros_like(u);
        for (size_t i=0; i<mixed_.size(); ++i) result=result+apply_stencil(mixed_[i],g_.mixed_indices[i],u);
        return result;
    }
    at::Tensor apply_direction(Size i,const at::Tensor& u) const {
        return apply_stencil(directions_.at(i),g_.direction_indices.at(i),u);
    }
    at::Tensor solve_splitting(Size i,const at::Tensor& u,double scale) const {
        RECORD_FUNCTION("qlt::packed_directional_solve",std::vector<c10::IValue>());
        const auto n=g_.lengths.at(i);
        const auto bands=scale*directions_[i].index_select(1,g_.permutation[i]).reshape({3,-1,n});
        const auto lo=bands[0].slice(-1,1,n), di=1.+bands[1], up=bands[2].slice(-1,0,n-1);
        const auto rhs=u.index_select(0,g_.permutation[i]).reshape({-1,n,u.size(1)});
        const auto result=adjoint_ ? SolveAdjoint::apply(lo,di,up,rhs) : solve_bands(lo,di,up,rhs);
        return result.reshape(u.sizes()).index_select(0,g_.inverse[i]);
    }
  private:
    at::Tensor sample(const at::Tensor& x,double t) const {
        if (times_.size()==1 || t<=times_.front()) return x[0];
        if (t>=times_.back()) return x[times_.size()-1];
        const auto hi=std::upper_bound(times_.begin(),times_.end(),t)-times_.begin();
        const double w=(t-times_[hi-1])/(times_[hi]-times_[hi-1]);
        return (1-w)*x[hi-1]+w*x[hi];
    }
    const Geometry& g_;
    at::Tensor b_,c_,k_;
    std::vector<double> times_;
    bool adjoint_;
    ext::shared_ptr<YieldTermStructure> risk_,dividend_;
    std::vector<at::Tensor> directions_,mixed_;
};

// Empty boundary set only. The public plan rejects custom boundaries/events.
// Retaining the hooks in the generated stages keeps upstream execution order.
struct EmptyBoundaries {
    void setTime(double) const {}
    void applyBeforeApplying(PackedOperator&) const {}
    void applyAfterApplying(at::Tensor&) const {}
    void applyAfterSolving(at::Tensor&) const {}
};
class PackedEvolver {
  public:
    struct traits {
        using operator_type=PackedOperator;
        using array_type=at::Tensor;
        using bc_set=EmptyBoundaries;
        using condition_type=StepCondition<at::Tensor>;
    };
    PackedEvolver(PackedOperator& op,int64_t scheme) : map_(&op),scheme_(scheme) {
        TORCH_CHECK(scheme>=0 && scheme<=3,"invalid packed scheme");
        const auto desc=scheme==1 ? FdmSchemeDesc::Hundsdorfer() : scheme==2 ?
            FdmSchemeDesc::ModifiedCraigSneyd() : scheme==3 ? FdmSchemeDesc::CraigSneyd() : FdmSchemeDesc::Douglas();
        theta_=desc.theta.value(); mu_=desc.mu.value();
    }
    void setStep(Time dt) { dt_=dt.value(); }
    void step(at::Tensor& a,Time t) {
        if (scheme_==0) step_douglas(a,t.value());
        else if (scheme_==1) step_hundsdorfer(a,t.value());
        else if (scheme_==2) step_modified_craig_sneyd(a,t.value());
        else step_craig_sneyd(a,t.value());
    }
  private:
    void step_douglas(at::Tensor&,double);
    void step_hundsdorfer(at::Tensor&,double);
    void step_modified_craig_sneyd(at::Tensor&,double);
    void step_craig_sneyd(at::Tensor&,double);
    PackedOperator* map_;
    EmptyBoundaries bcSet_;
    int64_t scheme_;
    double dt_=0.,theta_=0.,mu_=0.;
};
#include <packed_stages.inc>
#include "packed_events.inc"

at::Tensor rollback(PackedOperator& op,at::Tensor state,double horizon,int64_t steps,int64_t scheme,bool snapshot,
                    const PackedEvents* events=nullptr) {
    TORCH_CHECK(steps>0,"positive steps required");
    std::vector<Time> stops;
    // Fdm2DimSolver/FdmNdimSolver add this stopping time even for price-only
    // requests. Omitting it would change the final two ADI time intervals.
    if (events) stops=events->stops;
    else if (snapshot) stops.emplace_back(.99*std::min(1./365.,horizon));
    auto model=FiniteDifferenceModel<PackedEvolver>(PackedEvolver(op,scheme),stops);
    if (events) model.rollback(state,horizon,0.,steps,*events);
    else model.rollback(state,horizon,0.,steps);
    TORCH_CHECK(at::isfinite(state).all().item<bool>(),"non-finite packed solution");
    return state;
}

ext::shared_ptr<YieldTermStructure> curve(Date ref,Date end,double rate,const Ints& dates,const Doubles& dfs) {
    TORCH_CHECK(std::isfinite(rate),"rate must be finite");
    if (dates.empty()) {
        TORCH_CHECK(dfs.empty(),"curve dates missing");
        return ext::make_shared<FlatForward>(ref,rate,Actual365Fixed());
    }
    TORCH_CHECK(dates.size()==dfs.size() && dates.size()>=2 && dates.get(0)==ref.serialNumber() &&
        dates.get(dates.size()-1)>=end.serialNumber() && dfs.get(0)==1.,"invalid curve coverage");
    std::vector<Date> ds; std::vector<Real> discounts;
    for (size_t i=0; i<dates.size(); ++i) {
        TORCH_CHECK(std::isfinite(dfs.get(i)) && dfs.get(i)>0 && (i==0 || dates.get(i)>dates.get(i-1)),"invalid curve");
        ds.emplace_back(dates.get(i)); discounts.emplace_back(dfs.get(i));
    }
    return ext::make_shared<DiscountCurve>(ds,discounts,Actual365Fixed());
}

class LatentBatch : public torch::CustomClassHolder {
    friend class MultiAssetBatch;
  public:
    LatentBatch(Axes axes,Doubles times,Doubles strikes,Ints types,int64_t reference,int64_t maturity,
        double rate,double yield,Ints risk_dates,Doubles risk_dfs,Ints yield_dates,Doubles yield_dfs,
        int64_t steps,int64_t scheme,std::string device,int64_t exercise,Ints exercise_dates,
        Ints cash_dates,Doubles cash_amounts)
    : g_(axes,at::Device(device)), times_(std::move(times)), steps_(steps),scheme_(scheme),
      reference_(reference),maturity_(maturity) {
        TORCH_CHECK(g_.d==2 || g_.d==3,"financial batching supports one or two latent states");
        TORCH_CHECK(g_.device.is_cpu() || g_.device.is_cuda(),"packed plans require CPU or CUDA");
        TORCH_CHECK(steps>0 && scheme>=0 && scheme<=3,"invalid time steps or scheme");
        for (const auto n : g_.lengths) TORCH_CHECK(n>=4,"financial axes need at least four nodes");
        const Date ref(reference),end(maturity);
        TORCH_CHECK(end>ref,"maturity must follow valuation");
        horizon_=Actual365Fixed().yearFraction(ref,end).value();
        check_times(times_,horizon_);
        risk_=curve(ref,end,rate,risk_dates,risk_dfs); dividend_=curve(ref,end,yield,yield_dates,yield_dfs);
        TORCH_CHECK(!strikes.empty() && strikes.size()==types.size(),"strikes/types must be nonempty and equal length");
        std::vector<at::Tensor> payoffs;
        for (size_t j=0; j<strikes.size(); ++j) {
            TORCH_CHECK(std::isfinite(strikes.get(j)) && strikes.get(j)>0 && (types.get(j)==1 || types.get(j)==-1),"invalid strike or option type");
            auto payoff=ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(types.get(j)),strikes.get(j));
            FdmLogInnerValue calculator(payoff,g_.mesh,0);
            Array terminal(g_.n);
            for (const auto& it : *g_.mesh->layout()) terminal[it.index()]=calculator.avgInnerValue(it,horizon_);
            payoffs.push_back(pack_array(terminal.begin(),terminal.size()));
        }
        terminal_=at::stack(payoffs,1).to(g_.device);
        events_=std::make_shared<PackedEvents>(g_,strikes,types,ref,end,exercise,exercise_dates,cash_dates,cash_amounts);
    }
    at::Tensor price(const at::Tensor& b,const at::Tensor& l,const at::Tensor& initial,bool adjoint) const {
        return price_curves(b,l,initial,adjoint,risk_,dividend_);
    }
    at::Tensor price_market(const at::Tensor& b,const at::Tensor& l,const at::Tensor& initial,bool adjoint,
        const at::Tensor& rate,const at::Tensor& yield,const Ints& risk_dates,const at::Tensor& risk_dfs,
        const Ints& yield_dates,const at::Tensor& yield_dfs) const {
        if (adjoint) for (const auto& input : {rate,yield,risk_dfs,yield_dfs})
            TORCH_CHECK(!input._fw_grad(0).defined(), "packed equation adjoint does not support forward tangents; use full AD");
        const auto risk=financial_curve(Date(reference_),Date(maturity_),rate,risk_dates,risk_dfs);
        const auto dividend=financial_curve(Date(reference_),Date(maturity_),yield,yield_dates,yield_dfs);
        return price_curves(b,l,initial,adjoint,risk,dividend);
    }
    at::Tensor forward_rates(const Doubles& starts,const Doubles& ends,
        const at::Tensor& rate,const at::Tensor& yield,const Ints& risk_dates,const at::Tensor& risk_dfs,
        const Ints& yield_dates,const at::Tensor& yield_dfs) const {
        TORCH_CHECK(starts.size()==ends.size() && !starts.empty(), "invalid forward intervals");
        const auto risk=financial_curve(Date(reference_),Date(maturity_),rate,risk_dates,risk_dfs);
        const auto dividend=financial_curve(Date(reference_),Date(maturity_),yield,yield_dates,yield_dfs);
        std::vector<at::Tensor> rows;
        for (size_t i=0;i<starts.size();++i) {
            TORCH_CHECK(starts.get(i)>=0 && ends.get(i)>starts.get(i) && ends.get(i)<=horizon_,
                        "invalid forward interval");
            rows.push_back(at::stack({risk->forwardRate(starts.get(i),ends.get(i),Continuous).rate().tensor(),
                dividend->forwardRate(starts.get(i),ends.get(i),Continuous).rate().tensor()}));
        }
        return at::stack(rows);
    }
    at::Tensor price_curves(const at::Tensor& b,const at::Tensor& l,const at::Tensor& initial,bool adjoint,
        const ext::shared_ptr<YieldTermStructure>& risk,const ext::shared_ptr<YieldTermStructure>& dividend) const {
        RECORD_FUNCTION("qlt::packed_latent_price",std::vector<c10::IValue>());
        auto k=at::zeros({static_cast<int64_t>(times_.size()),g_.n},l.options());
        check_fields(b,l,k,g_,times_.size(),true); check(initial,g_.device);
        TORCH_CHECK(initial.dim()==1 && initial.numel()==g_.d,"initial must have shape [directions]");
        if (adjoint) TORCH_CHECK(!initial._fw_grad(0).defined(),"packed equation adjoint does not support forward tangents; use full AD");
        auto z=initial.to(at::kCPU);
        for (int64_t i=0; i<g_.d; ++i) TORCH_CHECK(z[i].item<double>()>g_.axes[i].front().value() &&
            z[i].item<double>()<g_.axes[i].back().value(),"initial must be strictly inside axes");
        PackedOperator op(g_,b,l,k,times_,adjoint,risk,dividend);
        auto state=rollback(op,terminal_,horizon_,steps_,scheme_,true,events_.get()).to(at::kCPU);
        AdjointMode adjoint_guard(adjoint); TensorKernelMode kernel_guard(4);
        std::vector<at::Tensor> prices;
        for (int64_t j=0; j<state.size(1); ++j) {
            const auto column=state.select(1,j);
            Real price;
            if (g_.d==2) {
                Matrix values(g_.lengths[1],g_.lengths[0]);
                unpack_array(column,values.begin());
                price=BicubicSpline(g_.axes[0].begin(),g_.axes[0].end(),g_.axes[1].begin(),g_.axes[1].end(),values)(Real(z[0]),Real(z[1]));
            } else {
                MultiCubicSpline<3>::data_table values(g_.axes);
                Array a(g_.n); unpack_array(column,a.begin());
                for (const auto& it : *g_.mesh->layout()) FdmNdimSolver<3>::setValue(values,it.coordinates(),a[it.index()]);
                price=MultiCubicSpline<3>(g_.axes,values,std::vector<bool>(3,false))(std::vector<Real>{Real(z[0]),Real(z[1]),Real(z[2])});
            }
            prices.push_back(price.tensor());
        }
        auto result=at::stack(prices).to(g_.device);
        TORCH_CHECK(at::isfinite(result).all().item<bool>(),"non-finite packed price");
        return result;
    }
  private:
    Geometry g_;
    Doubles times_;
    int64_t steps_,scheme_,reference_,maturity_;
    double horizon_;
    ext::shared_ptr<YieldTermStructure> risk_,dividend_;
    at::Tensor terminal_;
    std::shared_ptr<PackedEvents> events_;
};

at::Tensor diffusion(const at::Tensor& b,const at::Tensor& l,const at::Tensor& k,const at::Tensor& terminal,
    const Axes& axes,const Doubles& times,double horizon,int64_t steps,int64_t scheme,bool adjoint) {
    Geometry g(axes,l.device()); check_times(times,horizon); check_fields(b,l,k,g,times.size(),false); check(terminal,g.device);
    TORCH_CHECK(terminal.dim()==2 && terminal.size(0)==g.n && terminal.size(1)>0,"terminal must have shape [nodes, contracts]");
    TORCH_CHECK(g.device.is_cpu() || g.device.is_cuda(),"packed rollback requires CPU or CUDA");
    if (adjoint) TORCH_CHECK(!terminal._fw_grad(0).defined(),"packed equation adjoint does not support forward tangents; use full AD");
    PackedOperator op(g,b,l,k,times,adjoint);
    return rollback(op,terminal,horizon,steps,scheme,false);
}
#include "multi_asset.inc"
}} // namespace quantlibtorch::packed

TORCH_LIBRARY_FRAGMENT(quantlibtorch,m) {
    using namespace quantlibtorch::packed;
    m.class_<MultiChecks>("MultiChecks");
    m.class_<LatentBatch>("LatentBatch")
        .def(torch::init<Axes,Doubles,Doubles,Ints,int64_t,int64_t,double,double,Ints,Doubles,Ints,Doubles,int64_t,int64_t,std::string,int64_t,Ints,Ints,Doubles>())
        .def("price",&LatentBatch::price)
        .def("price_market",&LatentBatch::price_market)
        .def("forward_rates",&LatentBatch::forward_rates);
    m.class_<MultiAssetBatch>("MultiAssetBatch")
        .def(torch::init<c10::List<c10::intrusive_ptr<LatentBatch>>>())
        .def("price",&MultiAssetBatch::price)
        .def("price_market",&MultiAssetBatch::price_market)
        .def("interval_times",&MultiAssetBatch::interval_times)
        .def("readout",&MultiAssetBatch::readout);
    m.def("_packed_diffusion(Tensor drift, Tensor diffusion, Tensor killing, Tensor terminal, float[][] axes, float[] times, float horizon, int steps, int scheme, bool adjoint) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch,CompositeImplicitAutograd,m) {
    m.impl("_packed_diffusion",TORCH_FN(quantlibtorch::packed::diffusion));
}
