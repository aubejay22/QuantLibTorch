"""Isolated pinned QuantLib market fixtures, meshing, and analytic price oracles."""
from datetime import date
import json
import math
import sys


def evaluate(request):
    import QuantLib as ql
    if ql.__version__ != "1.40":
        raise RuntimeError("QuantLib 1.40 required")
    def qdate(s):
        d = date.fromisoformat(s)
        return ql.Date(d.day, d.month, d.year)
    ref = qdate(request.get("valuation_date", "2025-01-01"))
    saved = ql.SavedSettings()
    ql.Settings.instance().evaluationDate = ref
    dc = ql.Actual365Fixed()
    risk = ql.YieldTermStructureHandle(ql.FlatForward(ref, request.get("rate", .03), dc))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(ref, request.get("dividend_yield", .01), dc))
    spot = request.get("spot", 100.)
    params = request.get("heston", [1.5, .04, .3, -.6, .045])
    kappa, theta, sigma, rho, v0 = params
    heston = ql.HestonProcess(risk, dividend, ql.QuoteHandle(ql.SimpleQuote(spot)), v0, kappa, theta, sigma, rho)
    analytic = ql.AnalyticHestonEngine(ql.HestonModel(heston), 144)
    black = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(spot)), dividend, risk,
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref, ql.NullCalendar(), request.get("volatility", .2), dc)))
    quotes = []
    for q in request.get("contracts", []):
        option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if q.get("option_type", "call")=="call" else ql.Option.Put, q["strike"]),
                                  ql.EuropeanExercise(qdate(q["maturity"])))
        option.setPricingEngine(ql.AnalyticEuropeanEngine(black) if request.get("black_scholes") else analytic)
        quotes.append(dict(q, price=option.NPV()))
    meshes = {}
    for name, mesh in request.get("meshes", {}).items():
        x = ql.Concentrating1dMesher(math.log(mesh.get("smin", 20.)), math.log(mesh.get("smax", 400.)),
            mesh["nx"], (math.log(100.), .12), True)
        v = ql.Concentrating1dMesher(0., mesh.get("vmax", .5), mesh["nv"], (.04, .12), True)
        meshes[name] = dict(mesh, x=list(x.locations()), v=list(v.locations()))
    del saved
    return dict(quantlib_version="1.40", quotes=quotes, meshes=meshes)


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
