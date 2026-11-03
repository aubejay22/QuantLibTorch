"""Multi-contract objectives over the existing QuantLib latent pricing adapter.

Serial execution remains the default reference. Optional packed execution
groups common-expiry European contracts without changing their numerical grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import torch
from torch.profiler import record_function

from .fd import _integer, _serial
from .latent import _axes, _tensor, fd_latent, state_grid
from .batch import LatentBatch
from .pricing import _finite_number


@dataclass(frozen=True)
class OptionQuote:
    """A European quote with a fixed positive residual scale in price units."""
    strike: float
    maturity: date
    price: float
    scale: float = 1.0
    option_type: str = "call"

    def __post_init__(self):
        for name in ("strike", "price", "scale"):
            value = _finite_number(name, getattr(self, name))
            if value < 0 or (name != "price" and value == 0):
                raise ValueError(f"{name} must be {'nonnegative' if name == 'price' else 'positive'}")
        _serial("maturity", self.maturity)
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be call or put")


class LatentCalibration:
    """Fixed-contract calibration objective with shared absolute model time.

    model(times, states) returns drift [T,N,m] and loadings [T,N,1+m,w].
    times is a fixed float64 vector, states is state_grid(axes). All calls
    use times from valuation_date, including shorter maturities; time is never
    rescaled separately for each contract. latent_state() returns the current
    differentiable [m] initial state (e.g. a bounded transform of a Parameter).

    loss() retains a single graph and supports higher derivatives in full mode.
    backward() recomputes fields and the initial state for each contract, adds
    first-order gradients of the identical separable mean squared objective,
    and frees each PDE graph. It accumulates into the caller's parameter .grad
    buffers, just like loss().backward(); callers zero these before an update.
    Coefficients are never detached to create surrogate leaves. The returned
    prices and loss from backward() are detached logging values only.

    execution='batched' creates fixed common-expiry LatentBatch plans and
    accumulates one graph per compatible group. The default serial path remains
    independently available. Plan configuration changes require a new instance.

    Models/state callbacks must be deterministic and free of mutable training
    state (no dropout or updating batch-normalization statistics). Stochastic
    objectives and nonseparable penalties are not handled by this
    streaming method. Add a model penalty once outside backward().
    """
    def __init__(self, quotes, *, axes, times, valuation_date=date(2025, 1, 1),
                 spot=100., rate=.03, dividend_yield=.01, discount_curve=None,
                 dividend_curve=None, steps_per_year=30, minimum_steps=4,
                 scheme="hundsdorfer", execution="serial", device="cpu"):
        self.quotes = tuple(quotes)
        if not self.quotes or not all(isinstance(q, OptionQuote) for q in self.quotes):
            raise ValueError("quotes must be a nonempty sequence of OptionQuote")
        self.axes = _axes(axes, minimum=4)
        if len(self.axes) not in (2, 3):
            raise ValueError("calibration supports one or two latent dimensions")
        self.valuation_date = valuation_date
        reference = _serial("valuation_date", valuation_date)
        if any(_serial("maturity", q.maturity) <= reference for q in self.quotes):
            raise ValueError("every maturity must follow valuation_date")
        self.spot = _finite_number("spot", spot)
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if isinstance(times, torch.Tensor):
            raise TypeError("times must be fixed numbers")
        self.times = tuple(_finite_number("times", t) for t in times)
        horizon = max((q.maturity-valuation_date).days/365 for q in self.quotes)
        if (not self.times or self.times[0] < 0 or any(b <= a for a, b in zip(self.times, self.times[1:]))
                or (len(self.times) > 1 and (self.times[0] != 0 or self.times[-1] < horizon))):
            raise ValueError("times must increase and cover all maturities")
        self.steps_per_year = _integer("steps_per_year", steps_per_year, 1)
        self.minimum_steps = _integer("minimum_steps", minimum_steps, 1)
        self.market = dict(rate=rate, dividend_yield=dividend_yield,
                           discount_curve=discount_curve, dividend_curve=dividend_curve, scheme=scheme)
        self.states = state_grid(self.axes)
        self.time_tensor = torch.tensor(self.times, dtype=torch.float64)
        self.targets = torch.tensor([q.price for q in self.quotes], dtype=torch.float64)
        self.scales = torch.tensor([q.scale for q in self.quotes], dtype=torch.float64)
        if execution not in ("serial", "batched"):
            raise ValueError("execution must be serial or batched")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type != "cpu" and execution != "batched"):
            raise ValueError("CUDA calibration requires batched execution")
        self.device = torch.device('cpu') if self.device.type == 'cpu' else torch.device(
            'cuda', torch.cuda.current_device() if self.device.index is None else self.device.index)
        self.states, self.time_tensor = self.states.to(self.device), self.time_tensor.to(self.device)
        self.targets, self.scales = self.targets.to(self.device), self.scales.to(self.device)
        self.execution = execution
        self.groups = ()
        if execution == "batched":
            groups = {}
            for i, quote in enumerate(self.quotes):
                groups.setdefault(quote.maturity, []).append(i)
            self.groups = tuple((tuple(indices), LatentBatch(
                [self.quotes[i].strike for i in indices],
                option_types=[self.quotes[i].option_type for i in indices],
                axes=self.axes, times=self.times, valuation_date=self.valuation_date,
                maturity_date=maturity, time_steps=max(self.minimum_steps,
                    math.ceil(self.steps_per_year*(maturity-self.valuation_date).days/365)),
                device=self.device, **self.market)) for maturity, indices in groups.items())
            self._batch_configuration = self._configuration()

    def _configuration(self):
        def freeze(value):
            if isinstance(value, dict):
                return tuple(sorted((k, freeze(v)) for k, v in value.items()))
            if isinstance(value, (tuple, list)):
                return tuple(freeze(v) for v in value)
            return value
        return freeze((self.quotes, self.axes, self.times, self.valuation_date,
                       self.steps_per_year, self.minimum_steps, self.market, self.device))

    def _check_configuration(self):
        if self.execution == "batched" and self._configuration() != self._batch_configuration:
            raise ValueError("batched calibration configuration changed; construct a new fixed plan")

    def _fields(self, model):
        with record_function("calibration::network_fields"):
            return model(self.time_tensor, self.states)

    def _initial(self, latent_state):
        latent = latent_state()
        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent state must be a torch.Tensor")
        if latent.device != self.device or latent.dtype != torch.float64 or latent.layout != torch.strided:
            raise ValueError("latent state must be dense float64 on the calibration device")
        if self.device.type == 'cpu':
            _tensor("latent state", latent)
        if latent.shape != (len(self.axes)-1,):
            raise ValueError("latent state must have shape [latent dimensions]")
        return torch.cat((latent.new_tensor([math.log(self.spot)]), latent))

    def _price(self, fields, initial, quote, mode):
        steps = max(self.minimum_steps, math.ceil(self.steps_per_year*(quote.maturity-self.valuation_date).days/365))
        with record_function("calibration::contract_price"):
            return fd_latent(*fields, initial, axes=self.axes, times=self.times,
                valuation_date=self.valuation_date, maturity_date=quote.maturity,
                strike=quote.strike, option_type=quote.option_type, time_steps=steps, mode=mode, **self.market)

    def prices(self, model, latent_state, *, mode="adjoint"):
        """Return a connected price vector; fields are evaluated once."""
        self._check_configuration()
        fields, initial = self._fields(model), self._initial(latent_state)
        if self.execution == "batched":
            prices = [None]*len(self.quotes)
            for indices, plan in self.groups:
                for index, value in zip(indices, plan.price(*fields, initial, mode=mode).unbind()):
                    prices[index] = value
            return torch.stack(prices)
        return torch.stack([self._price(fields, initial, q, mode) for q in self.quotes])

    def loss(self, model, latent_state, *, mode="adjoint"):
        prices = self.prices(model, latent_state, mode=mode)
        return ((prices-self.targets)/self.scales).square().mean(), prices

    def backward(self, model, latent_state, *, mode="adjoint", contract_context=None):
        """Accumulate exact first-order objective gradients, one graph at a time.

        contract_context(i) optionally supplies a context manager for bounded
        profiling/diagnostics around a whole contract, including its backward.
        In batched execution it receives a tuple of original quote indices for
        the whole common-expiry group, including the group's backward.
        """
        from contextlib import nullcontext
        self._check_configuration()
        if self.execution == "batched":
            prices, terms = [None]*len(self.quotes), []
            for indices, plan in self.groups:
                # For packed execution the diagnostic context receives all
                # original quote indices in this compatible group.
                with contract_context(indices) if contract_context else nullcontext():
                    fields, initial = self._fields(model), self._initial(latent_state)
                    values = plan.price(*fields, initial, mode=mode)
                    term = ((values-self.targets[list(indices)])/self.scales[list(indices)]).square().sum()/len(self.quotes)
                    with record_function("calibration::batch_backward"):
                        term.backward()
                    for index, value in zip(indices, values.detach().unbind()):
                        prices[index] = value
                    terms.append(term.detach())
                    del fields, initial, values, term
            return torch.stack(terms).sum(), torch.stack(prices)
        prices, terms = [], []
        for i, quote in enumerate(self.quotes):
            with contract_context(i) if contract_context else nullcontext():
                fields, initial = self._fields(model), self._initial(latent_state)
                price = self._price(fields, initial, quote, mode)
                term = ((price-self.targets[i])/self.scales[i]).square()/len(self.quotes)
                with record_function("calibration::contract_backward"):
                    term.backward()
                prices.append(price.detach())
                terms.append(term.detach())
                del fields, initial, price, term
        return torch.stack(terms).sum(), torch.stack(prices)
