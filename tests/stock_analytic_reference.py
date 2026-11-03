"""Independent stock QuantLib oracle; run only as a separate process."""
import json
import sys


def price(request):
    import QuantLib as ql
    assert ql.__version__ == "1.40"
    def date(s):
        y, m, d = map(int, s.split("-"))
        return ql.Date(d, m, y)
    ref, expiry = date("2025-01-01"), date("2026-01-01")
    ql.Settings.instance().evaluationDate = ref
    dc = ql.Actual365Fixed()
    def curve(name, rate):
        points = request.get(name)
        return ql.YieldTermStructureHandle(
            ql.DiscountCurve([date(d) for d, _ in points], [v for _, v in points], dc)
            if points else ql.FlatForward(ref, rate, dc))
    risk = curve("discount_curve", request.get("rate", .03))
    div = curve("dividend_curve", request.get("dividend_yield", .01))
    vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref, ql.NullCalendar(), request.get("volatility", .2), dc))
    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(request.get("spot", 100.))), div, risk, vol)
    cash = [ql.FixedDividend(v, date(d)) for d, v in request.get("cash_dividends", [])]
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if request.get("option_type", "call") == "call"
                                                 else ql.Option.Put, request.get("strike", 100.)), ql.EuropeanExercise(expiry))
    option.setPricingEngine(ql.AnalyticDividendEuropeanEngine(process, cash) if cash else ql.AnalyticEuropeanEngine(process))
    return option.NPV()


if __name__ == "__main__":
    print(json.dumps([price(r) for r in json.load(sys.stdin)]))
