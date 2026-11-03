"""Shared-operator contract plans using packed QuantLib coefficient/stage kernels."""
from __future__ import annotations

from datetime import date, timedelta
import math

import torch

from .fd import _curve, _integer, _serial
from .latent import _axes, _config, _fields, _tensor
from .pricing import _finite_number
from .financial import _market


class LatentBatch:
    """A reusable fixed plan for common-expiry calls and puts.

    All contracts share exactly one mesh, coefficient-time grid, maturity,
    curves, scheme, step count, initial state and model fields. Only strikes
    and option types vary. No mesh merging, time rescaling, event alignment or
    boundary substitution is performed. European, American and Bermudan
    exercise are supported. Fixed cash dividends use QuantLib's spot jump
    convention with one latent factor. Damping and custom boundaries raise.

    The plan owns passive QuantLib derivative bases, indices, averaged payoffs
    and financial configuration only. Every price call creates a fresh model
    graph. Arrays passed to the constructor are copied. Full AD retains higher
    derivatives; equation adjoints are first order. Both use the original ADI
    formulas and scheduler. Float64; no torch.func/compile contract.

    device='cuda' is an experimental device-resident coefficient/rollback
    path using tensor PCR, with strict row diagonal dominance and scaled solve
    residual checks. CPU uses checked Thomas arithmetic. CUDA financial spline
    readout runs on CPU through connected transfers; device synchronization and
    these transfers can outweigh GPU gains. No GPU speedup is implied.
    """
    def __init__(self, strikes, *, axes, times=(0.,), option_types=None,
                 valuation_date=date(2025, 1, 1), maturity_date=date(2026, 1, 1),
                 rate=.03, dividend_yield=.01, discount_curve=None, dividend_curve=None,
                 time_steps=30, scheme="hundsdorfer", exercise="european",
                 exercise_dates=(), cash_dividends=(), boundary_conditions=(),
                 damping_steps=0, device="cpu"):
        axes = _axes(axes, minimum=4)
        if len(axes) not in (2, 3):
            raise ValueError("LatentBatch supports one or two latent states")
        if boundary_conditions:
            raise ValueError("LatentBatch does not support custom boundary conditions")
        if _integer("damping_steps", damping_steps, 0) != 0:
            raise ValueError("LatentBatch requires zero damping_steps")
        reference, maturity = _serial("valuation_date", valuation_date), _serial("maturity_date", maturity_date)
        if maturity <= reference:
            raise ValueError("maturity_date must follow valuation_date")
        if exercise not in ("european", "american", "bermudan"):
            raise ValueError("exercise must be european, american or bermudan")
        ex_dates = [_serial("exercise_dates", d) for d in exercise_dates]
        if exercise == "bermudan":
            if (not ex_dates or ex_dates[0] <= reference or ex_dates[-1] != maturity
                    or any(b <= a for a, b in zip(ex_dates, ex_dates[1:]))):
                raise ValueError("Bermudan dates must increase, follow valuation and end at maturity")
        elif ex_dates:
            raise ValueError("exercise_dates require Bermudan exercise")
        cash_pairs = list(cash_dividends)
        cash_dates = [_serial("cash_dividends", d) for d, _ in cash_pairs]
        cash_amounts = [_finite_number("cash_dividends", a) for _, a in cash_pairs]
        if cash_pairs and len(axes) != 2:
            raise ValueError("cash dividends currently require one latent factor")
        if (any(d <= reference or d > maturity for d in cash_dates)
                or any(b <= a for a, b in zip(cash_dates, cash_dates[1:]))):
            raise ValueError("cash dividend dates must increase, follow valuation and not exceed maturity")
        if any(a < 0 for a in cash_amounts):
            raise ValueError("cash dividends must be nonnegative")
        if isinstance(strikes, torch.Tensor) or isinstance(times, torch.Tensor):
            raise TypeError("strikes and times must be fixed sequences, not tensors")
        strikes = tuple(_finite_number("strike", k) for k in strikes)
        if not strikes or min(strikes) <= 0:
            raise ValueError("strikes must be nonempty and positive")
        option_types = tuple(option_types) if option_types is not None else ("call",)*len(strikes)
        if len(option_types) != len(strikes) or any(t not in ("call", "put") for t in option_types):
            raise ValueError("option_types must contain one call/put for every strike")
        times = tuple(_finite_number("times", t) for t in times)
        horizon = (maturity-reference)/365.
        if (not times or times[0] < 0 or any(b <= a for a, b in zip(times, times[1:]))
                or (len(times) > 1 and (times[0] != 0 or times[-1] < horizon))):
            raise ValueError("times must increase and cover [0, horizon]")
        steps, scheme_id, _, _ = _config(time_steps, scheme, "full", "auto")
        risk_dates, risk_dfs = _curve("discount_curve", discount_curve, reference, maturity)
        yield_dates, yield_dfs = _curve("dividend_curve", dividend_curve, reference, maturity)
        rate = _finite_number("rate", rate)
        dividend_yield = _finite_number("dividend_yield", dividend_yield)
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("LatentBatch requires a CPU or CUDA device")
        device = torch.device('cpu') if device.type == 'cpu' else torch.device(
            'cuda', torch.cuda.current_device() if device.index is None else device.index)
        self._axes = tuple(tuple(a) for a in axes)
        self._times, self._horizon, self._device = times, horizon, device
        self._count = len(strikes)
        self._reference, self._maturity = reference, maturity
        self._market_defaults = dict(rate=rate, dividend_yield=dividend_yield,
            discount_curve=None if not risk_dates else tuple(zip(risk_dates, risk_dfs)),
            dividend_curve=None if not yield_dates else tuple(zip(yield_dates, yield_dfs)))
        # Keep dates as date objects for the same public market validator.
        for key in ("discount_curve", "dividend_curve"):
            if self._market_defaults[key] is not None:
                self._market_defaults[key] = tuple((valuation_date+timedelta(days=d-reference), v)
                    for d, v in self._market_defaults[key])
        self._group_key = (tuple(map(len, axes)), times, horizon, steps, scheme_id,
                           str(device), len(strikes), exercise,
                           tuple(d-reference for d in ex_dates), tuple(d-reference for d in cash_dates))
        self._native = torch.classes.quantlibtorch.LatentBatch(axes, list(times), list(strikes),
            [1 if t == "call" else -1 for t in option_types], reference, maturity, rate, dividend_yield,
            risk_dates, risk_dfs, yield_dates, yield_dfs, steps, scheme_id, str(device),
            {"european": 0, "american": 1, "bermudan": 2}[exercise], ex_dates, cash_dates, cash_amounts)

    @property
    def axes(self):
        return self._axes

    @property
    def times(self):
        return self._times

    @property
    def contracts(self):
        return self._count

    @classmethod
    def from_contracts(cls, contracts, **shared):
        """Build from fd_latent-style fixed keyword dictionaries.

        Missing entries inherit shared values/defaults. Every explicit per-
        contract override except strike/option_type must agree exactly after
        sequence normalization. Incompatible problems raise instead of being
        silently regrouped or modified. Use separate plans for those problems.
        """
        def freeze(value):
            if isinstance(value, torch.Tensor):
                raise TypeError("contract configuration must be fixed, not tensors")
            if isinstance(value, dict):
                return tuple(sorted((k, freeze(v)) for k, v in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(v) for v in value)
            return value
        import inspect
        defaults = {k: v.default for k, v in inspect.signature(cls).parameters.items()
                    if v.default is not inspect.Parameter.empty and k not in ("strikes", "option_types")}
        rows = []
        for contract in contracts:
            row = {**defaults, **shared, **dict(contract)}
            if "strike" not in row:
                raise ValueError("each contract requires strike")
            rows.append(row)
        if not rows:
            raise ValueError("contracts must not be empty")
        config = {k: v for k, v in rows[0].items() if k not in ("strike", "option_type")}
        key = freeze(config)
        for row in rows[1:]:
            candidate = {k: v for k, v in row.items() if k not in ("strike", "option_type")}
            if freeze(candidate) != key:
                changed = sorted(k for k in config.keys() | candidate.keys()
                                 if freeze(config.get(k)) != freeze(candidate.get(k)))
                raise ValueError(f"incompatible contract configuration: {', '.join(changed)}")
        return cls([r["strike"] for r in rows], option_types=[r.get("option_type", "call") for r in rows], **config)

    def _market_inputs(self, overrides):
        unknown = set(overrides)-self._market_defaults.keys()
        if unknown:
            raise TypeError(f"unsupported market inputs: {', '.join(sorted(unknown))}")
        return _market(**{**self._market_defaults, **overrides},
                       reference=self._reference, maturity=self._maturity)

    def price(self, latent_drift, diffusion, initial_state, *, mode="adjoint", **market):
        """Return connected prices [contracts].

        Optional rate, dividend_yield, discount_curve and dividend_curve overrides
        build fresh QuantLib curves each call. Scalars/discount factors may be
        CPU float64 tensors, including for CUDA plans; interval rates transfer
        with their graph. Pass curve=None to replace a configured curve with a
        flat rate. Geometry, financial events and dated conventions stay fixed.
        """
        if mode not in ("full", "adjoint"):
            raise ValueError("mode must be full or adjoint")
        # Native validation owns shape/device/finiteness checks so a large field
        # is not scanned twice at this execution boundary.
        for name, value in (("latent_drift", latent_drift), ("diffusion", diffusion), ("initial_state", initial_state)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if market:
            return self._native.price_market(latent_drift, diffusion, initial_state, mode == "adjoint",
                                             *self._market_inputs(market))
        return self._native.price(latent_drift, diffusion, initial_state, mode == "adjoint")


def diffusion_rollback_batch(drift, diffusion, terminal, *, axes, times=None,
                             killing=None, horizon=1., time_steps=30,
                             scheme="hundsdorfer", mode="full"):
    """QuantLib diffusion rollback with shared coefficients and terminal [N,B].

    Each column is an independent RHS. All columns use the exact same fixed
    numerical problem, including boundaries. Spatial/coefficient conventions
    match diffusion_rollback; the original single-RHS implementation remains
    independently available. One to arbitrary spatial dimensions, CPU float64.
    """
    axes = _axes(axes)
    horizon = _finite_number("horizon", horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    times = _fields(drift, diffusion, times, axes, horizon, False)
    n = math.prod(map(len, axes))
    _tensor("terminal", terminal)
    if terminal.ndim != 2 or terminal.shape[0] != n or terminal.shape[1] < 1:
        raise ValueError("terminal must have shape [nodes, contracts]")
    if killing is None:
        killing = drift.new_zeros(drift.shape[:2])
    _tensor("killing", killing)
    steps, scheme_id, adjoint, _ = _config(time_steps, scheme, mode, "auto")
    return torch.ops.quantlibtorch._packed_diffusion(drift, diffusion, killing, terminal,
        axes, times, horizon, steps, scheme_id, adjoint)
