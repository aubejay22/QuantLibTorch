"""Stock QuantLib 1.40 oracle, run only in a separate Python process.

Read one JSON request on stdin and write one JSON result on stdout. This module
does not import QuantLib at import time. The pricing process never imports
quantlibtorch, so a double QuantLib library cannot enter its TorchReal process.
"""
from __future__ import annotations

import json
import statistics
import sys
import time


def price(request, shift=0.0):
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
        if points is None:
            result = ql.FlatForward(reference, flat, dc)
        else:
            result = ql.DiscountCurve(
                [date(d) for d, _ in points], [df for _, df in points], dc
            )
        return ql.YieldTermStructureHandle(result)

    risk = curve("discount_curve", request.get("rate", 0.03))
    dividend = curve("dividend_curve", request.get("dividend_yield", 0.01))
    black = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(reference, ql.NullCalendar(), request.get("mesh_volatility", 0.2), dc)
    )
    if "time_vols" in request:
        # Equal spatial rows make FixedLocalVolSurface's S-linear interpolation
        # agree with TensorLocalVol's log(S)-linear interpolation exactly.
        vols = [v + shift for v in request["time_vols"]]
        matrix = ql.Matrix([vols, vols])
        local = ql.FixedLocalVolSurface(reference, request["times"], [1.0, 1000.0], matrix, dc)
    else:
        local = ql.LocalConstantVol(reference, request.get("volatility", 0.2) + shift, dc)
    process = ql.GeneralizedBlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(request.get("spot", 100.0))),
        dividend, risk, black, ql.LocalVolTermStructureHandle(local),
    )
    kind = request.get("exercise", "european")
    if kind == "american":
        exercise = ql.AmericanExercise(reference, maturity)
    elif kind == "bermudan":
        exercise = ql.BermudanExercise([date(d) for d in request["exercise_dates"]])
    else:
        exercise = ql.EuropeanExercise(maturity)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Put if request.get("option_type", "call") == "put" else ql.Option.Call,
        request.get("strike", 100.0),
    )
    dividends = [ql.FixedDividend(amount, date(d)) for d, amount in request.get("cash_dividends", [])]
    schemes = {
        "douglas": ql.FdmSchemeDesc.Douglas,
        "crank_nicolson": ql.FdmSchemeDesc.CrankNicolson,
        "implicit_euler": ql.FdmSchemeDesc.ImplicitEuler,
        "craig_sneyd": ql.FdmSchemeDesc.CraigSneyd,
        "modified_craig_sneyd": ql.FdmSchemeDesc.ModifiedCraigSneyd,
        "hundsdorfer": ql.FdmSchemeDesc.Hundsdorfer,
    }
    engine = ql.FdBlackScholesVanillaEngine(
        process, dividends,
        request.get("time_steps", 50), request.get("grid_points", 100),
        request.get("damping_steps", 2), schemes[request.get("scheme", "douglas")](),
        True, -1.0, (ql.FdBlackScholesVanillaEngine.Escrowed
                    if request.get("cash_dividend_model") == "escrowed"
                    else ql.FdBlackScholesVanillaEngine.Spot),
    )
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    result = option.NPV()
    # Keep the guard alive until all computations have completed.
    del saved
    return result


def evaluate(request):
    result = {"price": price(request), "quantlib_version": "1.40"}
    if request.get("bump"):
        h = request["bump"]
        result["parallel_vol_shift_derivative"] = (price(request, h) - price(request, -h)) / (2 * h)
    if request.get("repeats"):
        samples = []
        for _ in range(request["repeats"]):
            start = time.perf_counter()
            price(request)
            samples.append(time.perf_counter() - start)
        result.update(forward_seconds_samples=samples,
                      forward_seconds_median=statistics.median(samples))
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
