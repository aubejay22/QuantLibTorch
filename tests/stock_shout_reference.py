"""Isolated original stock QuantLib 1.40 Black-Scholes shout engine oracle.

Stock bumps rebuild automatic geometry; they are labelled accordingly and
must not be interpreted as exact fixed-mesh equation sensitivities.
"""
import json
import sys


def price(request, shifts=None):
    import QuantLib as ql
    if ql.__version__ != "1.40":
        raise RuntimeError("stock oracle requires QuantLib 1.40")
    shifts = shifts or {}

    def date(value):
        year, month, day = map(int, value.split("-"))
        return ql.Date(day, month, year)

    reference = date(request.get("valuation_date", "2025-01-01"))
    maturity = date(request.get("maturity_date", "2026-01-01"))
    saved = ql.SavedSettings()
    ql.Settings.instance().evaluationDate = reference
    dc = {"actual365fixed": ql.Actual365Fixed, "actual360": ql.Actual360}[
        request.get("day_counter", "actual365fixed")]()

    def scalar(name, default):
        return request.get(name, default) + shifts.get(name, 0.)

    def curve(name, flat):
        points = request.get(name)
        return ql.YieldTermStructureHandle(ql.FlatForward(reference, flat, dc) if points is None else
            ql.DiscountCurve([date(d) for d, _ in points],
                             [v + shifts.get(f"{name}:{i}", 0.) for i, (_, v) in enumerate(points)], dc))

    process = ql.GeneralizedBlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(scalar("spot", 100.))),
        curve("dividend_curve", scalar("dividend_yield", .01)),
        curve("discount_curve", scalar("rate", .03)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            reference, ql.NullCalendar(), scalar("volatility", .2), dc)))
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Put if request.get("option_type", "call") == "put" else ql.Option.Call,
        request.get("strike", 100.))
    kind = request.get("exercise", "american")
    exercise = (ql.AmericanExercise(reference, maturity) if kind == "american" else
                ql.BermudanExercise([date(d) for d in request["exercise_dates"]])
                if kind == "bermudan" else ql.EuropeanExercise(maturity))
    cash = [ql.FixedDividend(amount, date(d)) for d, amount in request.get("cash_dividends", [])]
    schemes = {"douglas": ql.FdmSchemeDesc.Douglas,
               "crank_nicolson": ql.FdmSchemeDesc.CrankNicolson,
               "implicit_euler": ql.FdmSchemeDesc.ImplicitEuler,
               "craig_sneyd": ql.FdmSchemeDesc.CraigSneyd,
               "modified_craig_sneyd": ql.FdmSchemeDesc.ModifiedCraigSneyd,
               "hundsdorfer": ql.FdmSchemeDesc.Hundsdorfer}
    engine = ql.FdBlackScholesShoutEngine(process, cash,
        request.get("time_steps", 12), request.get("x_grid", 31),
        request.get("damping_steps", 0), schemes[request.get("scheme", "douglas")]())
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    value = option.NPV()
    del saved
    return value


def evaluate(request):
    result = {"price": price(request), "quantlib_version": "1.40"}
    if request.get("bump"):
        h = request["bump"]
        result["moving_mesh_gradients"] = {
            name: (price(request, {name: h}) - price(request, {name: -h}))/(2*h)
            for name in request.get("gradient_names", ("volatility", "spot", "rate", "dividend_yield"))}
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
