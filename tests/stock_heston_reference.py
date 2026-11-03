"""Isolated stock QuantLib 1.40 Heston engine and fixed-mesh sensitivity oracle.

The full engine supplies the independent NPV. For equation sensitivities,
FdmHestonSolver uses its original operator, events and interpolation on a mesh
constructed with the original engine's passive reference parameters. Never
import stock QuantLib into a process that loaded the TorchReal native library.
"""
from __future__ import annotations

import copy
import json
import sys


PARAMETERS = ("v0", "kappa", "theta", "sigma", "rho", "spot", "rate", "dividend_yield")
DEFAULTS = (.045, 1.5, .04, .3, -.6, 100., .03, .01)


def price(request, *, fixed=False, shifts=None, return_axes=False):
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
    dc = {"actual365fixed": ql.Actual365Fixed, "actual360": ql.Actual360}[
        request.get("day_counter", "actual365fixed")]()

    def process(data, shifts=None):
        values = {name: data.get(name, default) for name, default in zip(PARAMETERS, DEFAULTS)}
        for name, shift in (shifts or {}).items():
            if name in values:
                values[name] += shift

        def curve(name, flat):
            points = data.get(name)
            if points is None:
                return ql.YieldTermStructureHandle(ql.FlatForward(reference, flat, dc))
            discounts = [value + (shifts or {}).get(f"{name}:{i}", 0.)
                         for i, (_, value) in enumerate(points)]
            return ql.YieldTermStructureHandle(ql.DiscountCurve(
                [date(d) for d, _ in points], discounts, dc))

        return ql.HestonProcess(
            curve("discount_curve", values["rate"]),
            curve("dividend_curve", values["dividend_yield"]),
            ql.QuoteHandle(ql.SimpleQuote(values["spot"])),
            *(values[name] for name in PARAMETERS[:5]))

    actual_process = process(request, shifts)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Put if request.get("option_type", "call") == "put" else ql.Option.Call,
        request.get("strike", 100.))
    kind = request.get("exercise", "european")
    exercise = (ql.AmericanExercise(reference, maturity) if kind == "american" else
                ql.BermudanExercise([date(d) for d in request["exercise_dates"]])
                if kind == "bermudan" else ql.EuropeanExercise(maturity))
    cash = [ql.FixedDividend(amount, date(d)) for d, amount in request.get("cash_dividends", [])]
    schemes = {"douglas": ql.FdmSchemeDesc.Douglas,
               "craig_sneyd": ql.FdmSchemeDesc.CraigSneyd,
               "modified_craig_sneyd": ql.FdmSchemeDesc.ModifiedCraigSneyd,
               "hundsdorfer": ql.FdmSchemeDesc.Hundsdorfer}
    scheme = schemes[request.get("scheme", "hundsdorfer")]()
    steps = request.get("time_steps", 8)
    damping = request.get("damping_steps", 0)
    nx, nv = request.get("grid_points", [request.get("x_grid", 15), request.get("v_grid", 9)])

    if fixed or return_axes:
        axes = request.get("axes")
        if axes is None:
            mesh_data = dict(request)
            mesh_data.update(request.get("mesh_parameters", {}))
            mesh_process = process(mesh_data)
            horizon = dc.yearFraction(reference, maturity)
            variance = ql.FdmHestonVarianceMesher(nv, mesh_process, horizon, max(5, steps // 50), .0001)
            # Preserve the original engine's helper argument order verbatim.
            equity = ql.FdmBlackScholesMesher(
                nx, ql.FdmBlackScholesMesher.processHelper(
                    mesh_process.s0(), mesh_process.dividendYield(),
                    mesh_process.riskFreeRate(), variance.volaEstimate()),
                horizon, request.get("strike", 100.), ql.nullDouble(), ql.nullDouble(),
                .0001, 2., (request.get("strike", 100.), .1), cash)
            axes = [list(equity.locations()), list(variance.locations())]
        if return_axes:
            del saved
            return axes
        mesher = ql.FdmMesherComposite([ql.Predefined1dMesher(axis) for axis in axes])
        calculator = ql.FdmLogInnerValue(payoff, mesher, 0)
        conditions = ql.FdmStepConditionComposite.vanillaComposite(
            cash, exercise, mesher, calculator, reference, dc)
        desc = ql.FdmSolverDesc(mesher, [], conditions, calculator,
                              dc.yearFraction(reference, maturity), steps, damping)
        solver = ql.FdmHestonSolver(actual_process, desc, scheme)
        value = solver.valueAt(actual_process.s0().value(),
                               request.get("v0", DEFAULTS[0]) + (shifts or {}).get("v0", 0.))
    else:
        engine = ql.FdHestonVanillaEngine(ql.HestonModel(actual_process), cash,
                                         steps, nx, nv, damping, scheme)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        value = option.NPV()
    del saved
    return value


def evaluate(request):
    axes = price(request, return_axes=True)
    fixed_request = copy.deepcopy(request)
    fixed_request["axes"] = axes
    result = {"price": price(request), "fixed_price": price(fixed_request, fixed=True),
              "axes": axes, "quantlib_version": "1.40"}
    if request.get("bump"):
        h = request["bump"]
        result["gradients"] = {
            name: (price(fixed_request, fixed=True, shifts={name: h})
                   - price(fixed_request, fixed=True, shifts={name: -h})) / (2*h)
            for name in request.get("gradient_names", PARAMETERS)}
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
