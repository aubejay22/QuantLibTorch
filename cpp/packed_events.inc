// Financial tensor event adapter. QuantLib 1.40 supplies exercise calendars,
// stopping times, intrinsic values, and the cash-dividend interpolation maps.
// Keep this adapter separate from upstream numerical-stage compatibility.
struct PackedEvents : StepCondition<at::Tensor> {
    int64_t exercise, nx;
    std::vector<Time> stops;
    std::vector<double> exercise_times, dividend_times;
    at::Tensor intrinsic;
    std::vector<at::Tensor> dividend_maps;

    PackedEvents(const Geometry& g,const Doubles& strikes,const Ints& types,
                 Date ref,Date end,int64_t kind,const Ints& ex_dates,
                 const Ints& cash_dates,const Doubles& cash_amounts)
        : exercise(kind),nx(g.lengths[0]) {
        TORCH_CHECK(kind>=0 && kind<=2,"invalid exercise kind");
        ext::shared_ptr<Exercise> ex;
        if (kind==0) ex=ext::make_shared<EuropeanExercise>(end);
        else if (kind==1) ex=ext::make_shared<AmericanExercise>(ref,end);
        else {
            TORCH_CHECK(!ex_dates.empty() && ex_dates.get(0)>ref.serialNumber() &&
                ex_dates.get(ex_dates.size()-1)==end.serialNumber(),"invalid Bermudan dates");
            std::vector<Date> ds;
            for (size_t i=0;i<ex_dates.size();++i) {
                TORCH_CHECK(i==0 || ex_dates.get(i)>ex_dates.get(i-1),"exercise dates must increase");
                ds.emplace_back(ex_dates.get(i));
            }
            ex=ext::make_shared<BermudanExercise>(ds);
        }
        TORCH_CHECK(kind==2 || ex_dates.empty(),"exercise dates require Bermudan exercise");
        TORCH_CHECK(cash_dates.size()==cash_amounts.size() && (cash_dates.empty() || g.d==2),
            "cash dividends require one latent factor and matching dates/amounts");
        DividendSchedule cash;
        for (size_t i=0;i<cash_dates.size();++i) {
            TORCH_CHECK(cash_dates.get(i)>ref.serialNumber() && cash_dates.get(i)<=end.serialNumber() &&
                (i==0 || cash_dates.get(i)>cash_dates.get(i-1)),"invalid cash dividend dates");
            TORCH_CHECK(std::isfinite(cash_amounts.get(i)) && cash_amounts.get(i)>=0.,"invalid cash dividends");
            cash.push_back(ext::make_shared<FixedDividend>(cash_amounts.get(i),Date(cash_dates.get(i))));
        }
        const Actual365Fixed dc;
        std::vector<at::Tensor> values;
        for (size_t j=0;j<strikes.size();++j) {
            auto payoff=ext::make_shared<PlainVanillaPayoff>(static_cast<Option::Type>(types.get(j)),strikes.get(j));
            auto calculator=ext::make_shared<FdmLogInnerValue>(payoff,g.mesh,0);
            Array a(g.n);
            for (const auto& it : *g.mesh->layout()) a[it.index()]=calculator->innerValue(it,0.);
            values.push_back(pack_array(a.begin(),a.size()));
            if (j==0) {
                stops=FdmStepConditionComposite::vanillaComposite(cash,ex,g.mesh,calculator,ref,dc)->stoppingTimes();
                if (kind==2) {
                    FdmBermudanStepCondition bermudan(ex->dates(),ref,dc,g.mesh,calculator);
                    for (const auto& t : bermudan.exerciseTimes()) exercise_times.push_back(t.value());
                }
            }
        }
        intrinsic=at::stack(values,1).to(g.device);
        // Fdm2DimSolver/FdmNdimSolver put their snapshot before the first event.
        const double first=stops.empty() ? dc.yearFraction(ref,end).value() : stops.front().value();
        stops.emplace_back(.99*std::min(1./365.,first));
        std::sort(stops.begin(),stops.end());
        if (!cash.empty()) {
            // FdmDividendHandler is linear in the continuation values. Probe
            // its actual one-dimensional implementation on cardinal vectors;
            // this preserves its clamping, interpolation and extrapolation.
            // Axis zero is contiguous, so the same map applies to each latent
            // row, exactly as the original handler's two-dimensional branch.
            auto mesh=ext::make_shared<FdmMesherComposite>(ext::make_shared<Predefined1dMesher>(g.axes[0]));
            FdmDividendHandler handler(cash,mesh,ref,dc,0);
            for (const auto& t : handler.dividendTimes()) {
                dividend_times.push_back(t.value());
                std::vector<at::Tensor> columns;
                for (int64_t j=0;j<nx;++j) {
                    Array basis(nx,0.); basis[j]=1.; handler.applyTo(basis,t);
                    columns.push_back(pack_array(basis.begin(),basis.size()));
                }
                dividend_maps.push_back(at::stack(columns,1).to(g.device));
            }
        }
    }
    bool exercises(double t) const {
        return exercise==1 || (exercise==2 && std::find(exercise_times.begin(),exercise_times.end(),t)!=exercise_times.end());
    }
    int64_t dividend_index(double t) const {
        const auto it=std::find(dividend_times.begin(),dividend_times.end(),t);
        return it==dividend_times.end() ? -1 : it-dividend_times.begin();
    }
    void applyTo(at::Tensor& u,Time time) const override {
        const double t=time.value();
        const auto index=dividend_index(t);
        if (index>=0) u=at::matmul(dividend_maps[index],u.reshape({-1,nx,u.size(-1)})).reshape(u.sizes());
        // QuantLib's strict comparison retains continuation at equality.
        // maximum() would split a derivative at ties and violate this rule.
        if (exercises(t)) u=at::where(intrinsic>u,intrinsic,u);
    }
};
