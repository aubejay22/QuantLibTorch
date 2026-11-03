"""Pinned stock-double Heston oracle. Run in a separate process from TorchReal."""
import json
import math
import sys


def price(request, shifts=None):
    import QuantLib as ql
    if ql.__version__ != "1.40":
        raise RuntimeError("stock oracle requires QuantLib 1.40")
    shifts = shifts or [0.]*6
    kappa, theta, sigma, rho, logspot, v0 = [x+y for x, y in zip(
        request.get("parameters", [1.5, .04, .3, -.6, math.log(100.), .045]), shifts)]
    def date(s):
        y, m, d = map(int, s.split("-"))
        return ql.Date(d, m, y)
    ref = date(request.get("valuation_date", "2025-01-01"))
    end = date(request.get("maturity_date", "2026-01-01"))
    dc = ql.Actual365Fixed()
    def curve(name, flat):
        points = request.get(name)
        return ql.YieldTermStructureHandle(ql.FlatForward(ref, flat, dc) if points is None else
            ql.DiscountCurve([date(d) for d, _ in points], [v for _, v in points], dc))
    risk, dividend = curve("discount_curve", request.get("rate", .03)), curve("dividend_curve", request.get("dividend_yield", .01))
    process = ql.HestonProcess(risk, dividend, ql.QuoteHandle(ql.SimpleQuote(math.exp(logspot))),
                               v0, kappa, theta, sigma, rho)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put if request.get("option_type") == "put" else ql.Option.Call,
                                  request.get("strike", 100.))
    kind = request.get("exercise", "european")
    exercise = (ql.AmericanExercise(ref, end) if kind == "american" else
                ql.BermudanExercise([date(d) for d in request["exercise_dates"]]) if kind == "bermudan" else
                ql.EuropeanExercise(end))
    if request.get("analytic"):
        saved = ql.SavedSettings()
        ql.Settings.instance().evaluationDate = ref
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process)))
        value = option.NPV()
        del saved
        return value
    mesh = ql.FdmMesherComposite([ql.Predefined1dMesher(a) for a in request["axes"]])
    calc = ql.FdmLogInnerValue(payoff, mesh, 0)
    cash = [ql.FixedDividend(amount, date(d)) for d, amount in request.get("cash_dividends", [])]
    conditions = ql.FdmStepConditionComposite.vanillaComposite(cash, exercise, mesh, calc, ref, dc)
    desc = ql.FdmSolverDesc(mesh, [], conditions, calc, dc.yearFraction(ref, end), request.get("time_steps", 5), 0)
    scheme = {"douglas": ql.FdmSchemeDesc.Douglas, "hundsdorfer": ql.FdmSchemeDesc.Hundsdorfer,
              "craig_sneyd": ql.FdmSchemeDesc.CraigSneyd, "modified_craig_sneyd": ql.FdmSchemeDesc.ModifiedCraigSneyd}[request.get("scheme", "hundsdorfer")]()
    solver = ql.Fdm2DimSolver(desc, scheme, ql.FdmHestonOp(mesh, process))
    return solver.interpolateAt(logspot, v0)


def evaluate(request):
    result = {"price": price(request), "quantlib_version": "1.40"}
    if request.get("bump"):
        h = request["bump"]
        derivatives = []
        for i in range(6):
            plus, minus = [0.]*6, [0.]*6
            plus[i], minus[i] = h, -h
            derivatives.append((price(request, plus)-price(request, minus))/(2*h))
        result["gradients"] = derivatives
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
