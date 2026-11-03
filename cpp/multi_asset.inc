// Included inside quantlibtorch::packed. This adapter keeps the Phase 23
// single-asset implementation independent. Spatial and spline bases come from
// pinned QuantLib objects; multi_stages.inc contains the original ADI bodies.

// Validation is a reduction over device flags, inspected once per forward or
// complete adjoint. No per-stage .item(), device assertion, or silent fallback.
// Scopes are per call/thread, exception safe, and never retained by a plan.
struct MultiChecks : public torch::CustomClassHolder {
    std::vector<at::Tensor> flags;
    std::mutex mutex;
    void add(at::Tensor value) {
        std::lock_guard<std::mutex> lock(mutex);
        flags.push_back(std::move(value));
    }
};
thread_local c10::intrusive_ptr<MultiChecks> multi_checks;
struct MultiValidation {
    c10::intrusive_ptr<MultiChecks> state=c10::make_intrusive<MultiChecks>(),previous;
    MultiValidation() : previous(multi_checks) { multi_checks=state; }
    ~MultiValidation() { multi_checks=previous; }
    void finish() const {
        if (!state->flags.empty()) TORCH_CHECK(at::stack(state->flags).all().item<bool>(),
            "multi-asset validation failed: finite inputs/results, initial state strictly inside axes, "
            "and CUDA strictly row-diagonally-dominant forward systems with scaled solve residual <= 1e-10 required");
    }
};
void multi_finite(const at::Tensor& t) {
    at::NoGradGuard guard;
    TORCH_INTERNAL_ASSERT(multi_checks);
    multi_checks->add(at::isfinite(t).all());
}
void multi_tensor(const at::Tensor& t,at::Device device) {
    TORCH_CHECK(t.device()==device && t.scalar_type()==at::kDouble && t.layout()==at::kStrided,
        "multi-asset inputs must be dense float64 on the plan device");
    multi_finite(t);
}
at::Tensor multi_solve(const at::Tensor& lo,const at::Tensor& di,const at::Tensor& up,
    const at::Tensor& rhs,bool transpose=false,c10::intrusive_ptr<MultiChecks> validation={}) {
    if (di.is_cpu()) return thomas_solve(lo,di,up,rhs);
    auto x=tridiagonal_solve_pcr(lo,di,up,rhs,false);
    at::NoGradGuard guard;
    const auto n=di.size(-1);
    const auto z=at::zeros_like(x.slice(-2,0,1));
    const auto left=at::cat({z,lo.unsqueeze(-1)*x.slice(-2,0,n-1)},-2);
    const auto right=at::cat({up.unsqueeze(-1)*x.slice(-2,1,n),z},-2);
    const auto center=di.unsqueeze(-1)*x;
    auto ok=at::isfinite(x).all() & ((left+center+right-rhs).abs()<=
        1e-10*(1.+rhs.abs()+left.abs()+center.abs()+right.abs())).all();
    if (!transpose) {
        const auto zero=at::zeros_like(di.slice(-1,0,1));
        ok=ok & (di.abs()>at::cat({zero,lo.abs()},-1)+at::cat({up.abs(),zero},-1)).all();
    }
    if (!validation) validation=multi_checks;
    TORCH_INTERNAL_ASSERT(validation);
    validation->add(ok);
    return x;
}
class MultiSolveAdjoint : public torch::autograd::Function<MultiSolveAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,at::Tensor lo,
        at::Tensor di,at::Tensor up,at::Tensor rhs) {
        auto x=multi_solve(lo,di,up,rhs);
        // The autograd engine may run reverse nodes on a different thread.
        // Carry the collector explicitly; do not depend on backward TLS.
        ctx->saved_data["validation"]=multi_checks;
        ctx->save_for_backward({lo,di,up,x});
        return x;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list incoming) {
        TORCH_CHECK(!at::GradMode::is_enabled(),"multi-asset adjoint is first-order only; use full AD");
        const auto s=ctx->get_saved_variables();
        const auto v=multi_solve(s[2],s[1],s[0],incoming[0],true,
            ctx->saved_data["validation"].toCustomClass<MultiChecks>());
        const auto& x=s[3]; const auto n=x.size(-2);
        return {-(v.slice(-2,1,n)*x.slice(-2,0,n-1)).sum(-1),-(v*x).sum(-1),
            -(v.slice(-2,0,n-1)*x.slice(-2,1,n)).sum(-1),v};
    }
};
at::Tensor multi_stencil(const at::Tensor& c,const at::Tensor& idx,const at::Tensor& u) {
    auto out=c.select(1,0).unsqueeze(-1)*u.index_select(1,idx[0]);
    for (int64_t j=1;j<c.size(1);++j)
        out=out+c.select(1,j).unsqueeze(-1)*u.index_select(1,idx[j]);
    return out;
}
class MultiStencilAdjoint : public torch::autograd::Function<MultiStencilAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,at::Tensor c,
        at::Tensor idx,at::Tensor u) {
        ctx->save_for_backward({c,idx,u});
        return multi_stencil(c,idx,u);
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list incoming) {
        TORCH_CHECK(!at::GradMode::is_enabled(),"multi-asset adjoint is first-order only; use full AD");
        const auto s=ctx->get_saved_variables();
        const auto& c=s[0]; const auto& idx=s[1]; const auto& u=s[2]; const auto& v=incoming[0];
        std::vector<at::Tensor> dc;
        auto du=at::zeros_like(u);
        for (int64_t j=0;j<c.size(1);++j) {
            dc.push_back((v*u.index_select(1,idx[j])).sum(-1));
            du=du.index_add(1,idx[j],v*c.select(1,j).unsqueeze(-1));
        }
        return {at::stack(dc,1),at::Tensor(),du};
    }
};

struct MultiInterval { double t,dt; int64_t lo,hi; double weight; double event_time=0.; };
// Ask the unchanged QuantLib scheduler for its exact steps, including the
// price solver's theta snapshot. Store its passive schedule for checkpointing.
struct MultiSchedule {
    struct traits {
        using operator_type=int; using array_type=at::Tensor; using bc_set=int;
        using condition_type=StepCondition<at::Tensor>;
    };
    std::vector<MultiInterval>* intervals;
    double dt=0.;
    void setStep(Time x) { dt=x.value(); }
    void step(at::Tensor&,Time t) { intervals->push_back({t.value(),dt,0,0,0.}); }
};
struct MultiScheduleEvents : StepCondition<at::Tensor> {
    explicit MultiScheduleEvents(std::vector<MultiInterval>& steps) : intervals(steps) {}
    std::vector<MultiInterval>& intervals;
    mutable bool initial=false;
    void applyTo(at::Tensor&,Time t) const override {
        if (intervals.empty()) initial=true;
        else intervals.back().event_time=t.value();
    }
};

// Natural cubic interpolation is linear in grid values. Materialize each
// cardinal polynomial from QuantLib CubicInterpolation itself, once on CPU.
// Evaluation and all derivatives with respect to grid values/query points then
// stay on the requested device. MultiCubicSpline also uses natural end rows.
at::Tensor multi_spline_basis(const std::vector<Real>& x) {
    const int64_t n=x.size();
    std::vector<double> out((n-1)*4*n,0.);
    for (int64_t j=0;j<n;++j) {
        std::vector<Real> y(n,0.); y[j]=1.;
        CubicInterpolation spline(x.begin(),x.end(),y.begin(),CubicInterpolation::Spline,false,
            CubicInterpolation::SecondDerivative,0.,CubicInterpolation::SecondDerivative,0.);
        for (int64_t i=0;i<n-1;++i) {
            out[(i*4)*n+j]=i==j ? 1. : 0.;
            out[(i*4+1)*n+j]=spline.aCoefficients()[i].value();
            out[(i*4+2)*n+j]=spline.bCoefficients()[i].value();
            out[(i*4+3)*n+j]=spline.cCoefficients()[i].value();
        }
    }
    return at::tensor(out,at::TensorOptions().dtype(at::kDouble)).reshape({n-1,4,n});
}

class MultiAssetBatch : public torch::CustomClassHolder {
  public:
    using Plans=c10::List<c10::intrusive_ptr<LatentBatch>>;
    explicit MultiAssetBatch(Plans plans) : plans_(std::move(plans)) {
        TORCH_CHECK(!plans_.empty(),"multi-asset plan cannot be empty");
        const auto p=plans_.get(0);
        assets=plans_.size(); contracts=p->terminal_.size(1); nodes=p->g_.n; dimensions=p->g_.d;
        lengths=p->g_.lengths; times=p->times_.vec(); scheme=p->scheme_; device=p->g_.device;
        for (const auto& item : plans_) {
            const c10::intrusive_ptr<LatentBatch> q=item;
            TORCH_CHECK(q->g_.lengths==lengths && q->times_.vec()==times && q->horizon_==p->horizon_ &&
                q->steps_==p->steps_ && q->scheme_==scheme && q->g_.device==device && q->terminal_.size(1)==contracts,
                "incompatible multi-asset topology, coefficient times, horizon, steps, scheme, device or contract count");
            TORCH_CHECK(q->events_->exercise==p->events_->exercise &&
                q->events_->exercise_times==p->events_->exercise_times &&
                q->events_->dividend_times==p->events_->dividend_times,
                "incompatible multi-asset financial event schedules");
        }
        at::Tensor dummy;
        MultiScheduleEvents callbacks(intervals);
        FiniteDifferenceModel<MultiSchedule>(MultiSchedule{&intervals},p->events_->stops).rollback(
            dummy,p->horizon_,0.,p->steps_,callbacks);
        std::vector<at::Tensor> rate_rows,yield_rows,terminals,intrinsics;
        for (auto& interval : intervals) {
            const double t1=std::max(0.,interval.t-interval.dt),t2=interval.t,t=.5*(t1+t2);
            const auto hi=std::upper_bound(times.begin(),times.end(),t)-times.begin();
            interval.lo=hi==0 ? 0 : hi-1; interval.hi=std::min<int64_t>(hi,times.size()-1);
            interval.weight=interval.lo==interval.hi ? 0. : (t-times[interval.lo])/(times[interval.hi]-times[interval.lo]);
            std::vector<double> r,q;
            for (const auto& item : plans_) {
                const c10::intrusive_ptr<LatentBatch> plan=item;
                r.push_back(plan->risk_->forwardRate(t1,t2,Continuous).rate().value());
                q.push_back(plan->dividend_->forwardRate(t1,t2,Continuous).rate().value());
            }
            rate_rows.push_back(at::tensor(r,at::kDouble)); yield_rows.push_back(at::tensor(q,at::kDouble));
        }
        rates=at::stack(rate_rows).unsqueeze(-1).to(device);
        yields=at::stack(yield_rows).unsqueeze(-1).to(device);
        for (int64_t i=0;i<dimensions;++i) {
            std::vector<at::Tensor> f,s,x,w;
            for (const auto& item : plans_) {
                const c10::intrusive_ptr<LatentBatch> plan=item;
                f.push_back(plan->g_.first[i]); s.push_back(plan->g_.second[i]);
                x.push_back(pack_array(plan->g_.axes[i].data(),lengths[i]));
                w.push_back(multi_spline_basis(plan->g_.axes[i]));
            }
            first.push_back(at::stack(f)); second.push_back(at::stack(s));
            coordinates.push_back(at::stack(x).to(device)); splines.push_back(at::stack(w).to(device));
        }
        for (size_t i=0;i<p->g_.pairs.size();++i) {
            std::vector<at::Tensor> bases;
            for (const auto& item : plans_) {
                const c10::intrusive_ptr<LatentBatch> plan=item;
                bases.push_back(plan->g_.mixed[i]);
            }
            mixed.push_back(at::stack(bases));
        }
        for (const auto& item : plans_) {
            const c10::intrusive_ptr<LatentBatch> plan=item;
            auto value=plan->terminal_;
            if (callbacks.initial) plan->events_->applyTo(value,p->horizon_);
            terminals.push_back(value);
            intrinsics.push_back(plan->events_->intrinsic);
        }
        terminal=at::stack(terminals);
        intrinsic=at::stack(intrinsics);
        for (size_t i=0;i<p->events_->dividend_maps.size();++i) {
            std::vector<at::Tensor> maps;
            for (const auto& item : plans_) {
                const c10::intrusive_ptr<LatentBatch> plan=item;
                maps.push_back(plan->events_->dividend_maps[i]);
            }
            dividend_maps.push_back(at::stack(maps).unsqueeze(1));
        }
        direction_indices=p->g_.direction_indices; mixed_indices=p->g_.mixed_indices;
        permutation=p->g_.permutation; inverse=p->g_.inverse; pairs=p->g_.pairs; interior=p->g_.interior;
    }
    at::Tensor segment(const at::Tensor& b,const at::Tensor& c,const at::Tensor& rates,const at::Tensor& yields,
        at::Tensor state,int64_t start,int64_t end,bool adjoint) const;
    at::Tensor price(const at::Tensor& b,const at::Tensor& l,const at::Tensor& z,bool adjoint,int64_t checkpoint);
    at::Tensor price_market(const at::Tensor& b,const at::Tensor& l,const at::Tensor& z,bool adjoint,
        int64_t checkpoint,const at::Tensor& rates,const at::Tensor& yields);
    Axes interval_times() const {
        Doubles starts,ends;
        for (const auto& step : intervals) {
            starts.push_back(std::max(0.,step.t-step.dt)); ends.push_back(step.t);
        }
        Axes out; out.push_back(starts); out.push_back(ends); return out;
    }
    at::Tensor apply_events(at::Tensor u,double t) const {
        const auto events=plans_.get(0)->events_;
        const auto index=events->dividend_index(t);
        if (index>=0) u=at::matmul(dividend_maps[index],u.reshape({assets,-1,lengths[0],contracts})).reshape(u.sizes());
        if (events->exercises(t)) u=at::where(intrinsic>u,intrinsic,u);
        return u;
    }
    at::Tensor interpolate(at::Tensor u,const at::Tensor& z) const {
        RECORD_FUNCTION("qlt::multi_asset_device_spline",std::vector<c10::IValue>());
        for (int64_t i=0;i<dimensions;++i) {
            const auto n=lengths[i];
            const auto x=coordinates[i];
            // Match upper_bound interval selection, including exact knots.
            const auto index=((z.select(1,i).unsqueeze(-1)>=x).sum(-1)-1).clamp(0,n-2);
            const auto base=x.gather(1,index.unsqueeze(-1)).squeeze(1);
            const auto h=(z.select(1,i)-base).unsqueeze(-1);
            const auto coeff=splines[i].gather(1,index.reshape({assets,1,1,1}).expand({assets,1,4,n})).squeeze(1);
            const auto w=coeff.select(1,0)+h*(coeff.select(1,1)+h*(coeff.select(1,2)+h*coeff.select(1,3)));
            u=(u.reshape({assets,-1,n,contracts})*w.reshape({assets,1,n,1})).sum(2);
        }
        return u.reshape({assets,contracts});
    }
    void check_initial(const at::Tensor& z) const {
        multi_tensor(z,device);
        TORCH_CHECK(z.dim()==2 && z.size(0)==assets && z.size(1)==dimensions,"initial must have shape [assets, directions]");
        at::NoGradGuard guard;
        for (int64_t i=0;i<dimensions;++i)
            multi_checks->add(((z.select(1,i)>coordinates[i].select(1,0)) &
                (z.select(1,i)<coordinates[i].select(1,lengths[i]-1))).all());
    }
    at::Tensor readout(const at::Tensor& u,const at::Tensor& z) const {
        MultiValidation checks;
        multi_tensor(u,device); check_initial(z);
        TORCH_CHECK(u.dim()==3 && u.size(0)==assets && u.size(1)==nodes && u.size(2)==contracts,
            "state must have shape [assets, nodes, contracts]");
        auto out=interpolate(u,z); multi_finite(out); checks.finish(); return out;
    }
    int64_t assets,contracts,nodes,dimensions,scheme;
    at::Device device{at::kCPU};
    std::vector<int64_t> lengths;
    std::vector<double> times;
    std::vector<MultiInterval> intervals;
    at::Tensor rates,yields,terminal,interior,intrinsic;
    std::vector<at::Tensor> dividend_maps;
    std::vector<at::Tensor> first,second,mixed,direction_indices,mixed_indices,permutation,inverse,coordinates,splines;
    std::vector<std::pair<int64_t,int64_t>> pairs;
  private:
    Plans plans_;
};

class MultiOperator {
  public:
    MultiOperator(const MultiAssetBatch& data,const at::Tensor& b,const at::Tensor& c,
        const at::Tensor& rates,const at::Tensor& yields,bool adjoint)
        : g(data),b_(b),c_(c),rates_(rates),yields_(yields),adjoint_(adjoint) {}
    int64_t interval=0;
    Size size() const { return g.dimensions; }
    void setTime(double,double) {
        RECORD_FUNCTION("qlt::multi_asset_assembly",std::vector<c10::IValue>());
        const auto& step=g.intervals[interval];
        const auto sample=[&step](const at::Tensor& x) {
            return step.lo==step.hi ? x.select(1,step.lo) :
                (1.-step.weight)*x.select(1,step.lo)+step.weight*x.select(1,step.hi);
        };
        const auto b=sample(b_),c=sample(c_),rate=rates_[interval],yield=yields_[interval];
        directions.clear(); cross.clear();
        for (int64_t i=0;i<g.dimensions;++i) {
            const auto v=c.select(-1,i).select(-1,i);
            const auto drift=i==0 ? (rate-yield)-.5*v*g.interior : b.select(-1,i-1);
            const auto base=g.second[i]*(.5*v).unsqueeze(1)+g.first[i]*drift.unsqueeze(1);
            directions.push_back(at::stack({base.select(1,0),base.select(1,1)-rate/g.dimensions,base.select(1,2)},1));
        }
        for (size_t p=0;p<g.pairs.size();++p) {
            const auto [i,j]=g.pairs[p];
            cross.push_back(g.mixed[p]*c.select(-2,i).select(-1,j).unsqueeze(1));
        }
    }
    at::Tensor stencil(const at::Tensor& c,const at::Tensor& idx,const at::Tensor& u) const {
        return adjoint_ ? MultiStencilAdjoint::apply(c,idx,u) : multi_stencil(c,idx,u);
    }
    at::Tensor apply_mixed(const at::Tensor& u) const {
        auto out=at::zeros_like(u);
        for (size_t i=0;i<cross.size();++i) out=out+stencil(cross[i],g.mixed_indices[i],u);
        return out;
    }
    at::Tensor apply_direction(Size i,const at::Tensor& u) const {
        return stencil(directions[i],g.direction_indices[i],u);
    }
    at::Tensor apply(const at::Tensor& u) const {
        auto out=apply_mixed(u);
        for (int64_t i=0;i<g.dimensions;++i) out=out+apply_direction(i,u);
        return out;
    }
    at::Tensor solve_splitting(Size i,const at::Tensor& u,double scale) const {
        RECORD_FUNCTION("qlt::multi_asset_solve",std::vector<c10::IValue>());
        const auto n=g.lengths[i];
        const auto bands=(scale*directions[i].index_select(2,g.permutation[i])).reshape({g.assets,3,-1,n});
        const auto lo=bands.select(1,0).slice(-1,1,n),di=1.+bands.select(1,1),up=bands.select(1,2).slice(-1,0,n-1);
        const auto rhs=u.index_select(1,g.permutation[i]).reshape({g.assets,-1,n,g.contracts});
        const auto out=adjoint_ ? MultiSolveAdjoint::apply(lo,di,up,rhs) : multi_solve(lo,di,up,rhs);
        return out.reshape(u.sizes()).index_select(1,g.inverse[i]);
    }
  private:
    const MultiAssetBatch& g;
    at::Tensor b_,c_,rates_,yields_;
    bool adjoint_;
    std::vector<at::Tensor> directions,cross;
};
struct MultiBoundaries {
    void setTime(double) const {}
    void applyBeforeApplying(MultiOperator&) const {}
    void applyAfterApplying(at::Tensor&) const {}
    void applyAfterSolving(at::Tensor&) const {}
};
class MultiEvolver {
  public:
    MultiEvolver(MultiOperator& op,int64_t scheme) : map_(&op),scheme_(scheme) {
        const auto desc=scheme==1 ? FdmSchemeDesc::Hundsdorfer() : scheme==2 ?
            FdmSchemeDesc::ModifiedCraigSneyd() : scheme==3 ? FdmSchemeDesc::CraigSneyd() : FdmSchemeDesc::Douglas();
        theta_=desc.theta.value(); mu_=desc.mu.value();
    }
    void setStep(double dt) { dt_=dt; }
    void step(at::Tensor& u,double t) {
        if (scheme_==0) step_douglas(u,t);
        else if (scheme_==1) step_hundsdorfer(u,t);
        else if (scheme_==2) step_modified_craig_sneyd(u,t);
        else step_craig_sneyd(u,t);
    }
  private:
    void step_douglas(at::Tensor&,double);
    void step_hundsdorfer(at::Tensor&,double);
    void step_modified_craig_sneyd(at::Tensor&,double);
    void step_craig_sneyd(at::Tensor&,double);
    MultiOperator* map_;
    MultiBoundaries bcSet_;
    int64_t scheme_;
    double dt_=0.,theta_=0.,mu_=0.;
};
#include <multi_stages.inc>

at::Tensor MultiAssetBatch::segment(const at::Tensor& b,const at::Tensor& c,
    const at::Tensor& rates,const at::Tensor& yields,at::Tensor state,
    int64_t start,int64_t end,bool adjoint) const {
    MultiOperator op(*this,b,c,rates,yields,adjoint);
    MultiEvolver evolver(op,scheme);
    for (int64_t j=start;j<end;++j) {
        op.interval=j;
        evolver.setStep(intervals[j].dt); evolver.step(state,intervals[j].t);
        state=apply_events(state,intervals[j].event_time);
    }
    return state;
}

class MultiRollbackAdjoint : public torch::autograd::Function<MultiRollbackAdjoint> {
  public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,at::Tensor b,at::Tensor c,
        at::Tensor rates,at::Tensor yields,at::Tensor state,c10::intrusive_ptr<MultiAssetBatch> plan,int64_t checkpoint) {
        ctx->saved_data["plan"]=plan; ctx->saved_data["checkpoint"]=checkpoint;
        std::vector<at::Tensor> saved{b,c,rates,yields};
        const int64_t count=plan->intervals.size();
        for (int64_t start=0;start<count;start+=checkpoint) {
            saved.push_back(state);
            state=plan->segment(b,c,rates,yields,state,start,std::min(count,start+checkpoint),false);
        }
        ctx->save_for_backward(saved);
        return state;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list incoming) {
        RECORD_FUNCTION("qlt::multi_asset_checkpoint_adjoint",std::vector<c10::IValue>());
        TORCH_CHECK(!at::GradMode::is_enabled(),"multi-asset adjoint is first-order only; use full AD");
        TORCH_CHECK(!incoming[0]._fw_grad(0).defined(),"multi-asset adjoint does not support forward tangents");
        MultiValidation checks;
        auto plan=ctx->saved_data["plan"].toCustomClass<MultiAssetBatch>();
        const auto saved=ctx->get_saved_variables();
        const int64_t stride=ctx->saved_data["checkpoint"].toInt(),count=plan->intervals.size();
        auto db=at::zeros_like(saved[0]),dc=at::zeros_like(saved[1]),v=incoming[0];
        const bool need_r=ctx->needs_input_grad(2),need_q=ctx->needs_input_grad(3);
        auto dr=need_r ? at::zeros_like(saved[2]) : at::Tensor();
        auto dq=need_q ? at::zeros_like(saved[3]) : at::Tensor();
        for (int64_t j=static_cast<int64_t>(saved.size())-5;j>=0;--j) {
            // Explicit first-order checkpoint rematerialization. These local
            // leaves receive exact segment VJPs, which are returned to the
            // ORIGINAL b/c inputs below (and hence to neural parameters).
            auto b=saved[0].detach().set_requires_grad(true);
            auto c=saved[1].detach().set_requires_grad(true);
            // Passive plan curves need no additional market-gradient graph.
            auto r=saved[2].detach().set_requires_grad(need_r);
            auto q=saved[3].detach().set_requires_grad(need_q);
            auto u=saved[j+4].detach().set_requires_grad(true);
            torch::autograd::variable_list inputs{b,c,u};
            if (need_r) inputs.push_back(r);
            if (need_q) inputs.push_back(q);
            torch::autograd::variable_list gradients;
            {
                at::AutoGradMode graph(true);
                auto out=plan->segment(b,c,r,q,u,j*stride,std::min(count,(j+1)*stride),true);
                gradients=torch::autograd::grad({out},inputs,{v},false,false);
            }
            db=db+gradients[0]; dc=dc+gradients[1];
            v=gradients[2];
            if (need_r) dr=dr+gradients[3];
            if (need_q) dq=dq+gradients[3+static_cast<int>(need_r)];
        }
        multi_finite(db); multi_finite(dc); multi_finite(v);
        if (need_r) multi_finite(dr);
        if (need_q) multi_finite(dq);
        checks.finish();
        return {db,dc,dr,dq,v,at::Tensor(),at::Tensor()};
    }
};
at::Tensor MultiAssetBatch::price(const at::Tensor& b,const at::Tensor& l,const at::Tensor& z,
    bool adjoint,int64_t checkpoint) {
    return price_market(b,l,z,adjoint,checkpoint,rates,yields);
}
at::Tensor MultiAssetBatch::price_market(const at::Tensor& b,const at::Tensor& l,const at::Tensor& z,
    bool adjoint,int64_t checkpoint,const at::Tensor& rates,const at::Tensor& yields) {
    RECORD_FUNCTION("qlt::multi_asset_price",std::vector<c10::IValue>());
    MultiValidation checks;
    TORCH_CHECK(checkpoint>0,"checkpoint_steps must be positive");
    multi_tensor(b,device); multi_tensor(l,device); check_initial(z);
    multi_tensor(rates,device); multi_tensor(yields,device);
    TORCH_CHECK(rates.sizes()==this->rates.sizes() && yields.sizes()==this->yields.sizes(),
        "interval rates and yields must have shape [intervals, assets, 1]");
    TORCH_CHECK(b.dim()==4 && b.size(0)==assets && b.size(1)==times.size() && b.size(2)==nodes &&
        b.size(3)==dimensions-1,"drift must have shape [assets, times, nodes, latent dimensions]");
    TORCH_CHECK(l.dim()==5 && l.size(0)==assets && l.size(1)==times.size() && l.size(2)==nodes &&
        l.size(3)==dimensions && l.size(4)>0,"diffusion must have shape [assets, times, nodes, dimensions, noise factors]");
    if (adjoint) for (const auto& t : {b,l,z,rates,yields}) TORCH_CHECK(!t._fw_grad(0).defined(),
        "multi-asset adjoint does not support forward tangents; use full AD");
    const auto covariance=at::matmul(l,l.transpose(-1,-2)); multi_finite(covariance);
    auto state=adjoint ? MultiRollbackAdjoint::apply(b,covariance,rates,yields,terminal,
        c10::intrusive_ptr<MultiAssetBatch>::unsafe_reclaim_from_nonowning(this),checkpoint) :
        segment(b,covariance,rates,yields,terminal,0,intervals.size(),false);
    multi_finite(state);
    auto prices=interpolate(state,z); multi_finite(prices); checks.finish();
    return prices;
}
