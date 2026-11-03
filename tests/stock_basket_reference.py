"""Isolated stock QuantLib 1.40 oracle for the real two-asset basket engine.

Never import this module's QuantLib into a TorchReal pricing process. The stock
process uses independent fixed Black volatilities for the mesh and explicit
local-volatility handles for the bumped coefficients.
"""
from __future__ import annotations

import json
import sys


def price(request, shifts=(0.0, 0.0, 0.0)):
    import QuantLib as ql

    if ql.__version__ != "1.40":
        raise RuntimeError(f"reference requires QuantLib 1.40, got {ql.__version__}")

    def date(value):
        year, month, day = map(int, value.split("-"))
        return ql.Date(day, month, year)

    reference = date(request.get("valuation_date", "2025-01-01"))
    maturity = date(request.get("maturity_date", "2026-01-01"))
    saved = ql.SavedSettings()
    ql.Settings.instance().evaluationDate = reference
    dc = ql.Actual365Fixed()

    def curve(name, flat):
        points = request.get(name)
        result = (ql.FlatForward(reference, flat, dc) if points is None else
                  ql.DiscountCurve([date(d) for d, _ in points], [df for _, df in points], dc))
        return ql.YieldTermStructureHandle(result)

    risk = curve("discount_curve", request.get("rate", 0.03))

    def process(asset, shift):
        dividend = curve(f"dividend_curve{asset}",
                         request.get(f"dividend_yield{asset}", 0.01 if asset == 1 else 0.015))
        black = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            reference, ql.NullCalendar(),
            request.get(f"mesh_volatility{asset}", 0.2 if asset == 1 else 0.25), dc))
        if f"time_vols{asset}" in request:
            # Spatially constant fields make the stock S-linear interpolation
            # agree exactly with the adapter's log(S)-linear convention.
            values = [v + shift for v in request[f"time_vols{asset}"]]
            local = ql.FixedLocalVolSurface(reference, request[f"times{asset}"],
                                            [1.0, 1000.0], ql.Matrix([values, values]), dc)
        else:
            local = ql.LocalConstantVol(reference, request.get(f"volatility{asset}", 0.2) + shift, dc)
        return ql.GeneralizedBlackScholesProcess(
            ql.QuoteHandle(ql.SimpleQuote(request.get(f"spot{asset}", 100. if asset == 1 else 95.))),
            dividend, risk, black, ql.LocalVolTermStructureHandle(local))

    kind = request.get("exercise", "european")
    if kind == "american":
        exercise = ql.AmericanExercise(reference, maturity)
    elif kind == "bermudan":
        exercise = ql.BermudanExercise([date(d) for d in request["exercise_dates"]])
    else:
        exercise = ql.EuropeanExercise(maturity)
    vanilla = ql.PlainVanillaPayoff(
        ql.Option.Put if request.get("option_type", "call") == "put" else ql.Option.Call,
        request.get("strike", 100.))
    payoff = (ql.MinBasketPayoff(vanilla) if request.get("basket", "min") == "min"
              else ql.MaxBasketPayoff(vanilla))
    schemes = {"douglas": ql.FdmSchemeDesc.Douglas,
               "craig_sneyd": ql.FdmSchemeDesc.CraigSneyd,
               "modified_craig_sneyd": ql.FdmSchemeDesc.ModifiedCraigSneyd,
               "hundsdorfer": ql.FdmSchemeDesc.Hundsdorfer}
    nx, ny = request.get("grid_points", (31, 25))
    engine = ql.Fd2dBlackScholesVanillaEngine(
        process(1, shifts[0]), process(2, shifts[1]),
        request.get("correlation", 0.35) + shifts[2], nx, ny,
        request.get("time_steps", 30), 0, schemes[request.get("scheme", "hundsdorfer")](),
        True, -1.0)
    option = ql.BasketOption(payoff, exercise)
    option.setPricingEngine(engine)
    result = option.NPV()
    del saved
    return result


def evaluate(request):
    result = {"price": price(request), "quantlib_version": "1.40"}
    if request.get("bump"):
        h = request["bump"]
        derivatives = []
        for i in range(3):
            plus, minus = [0.] * 3, [0.] * 3
            plus[i], minus[i] = h, -h
            derivatives.append((price(request, plus) - price(request, minus)) / (2 * h))
        result["coefficient_derivatives"] = derivatives
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
