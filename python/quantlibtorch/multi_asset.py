"""Independent asset operators, shared contract RHS, and device financial readout."""
from __future__ import annotations

import torch

from .batch import LatentBatch
from .fd import _integer


class MultiAssetBatch:
    """Price a rectangular collection of options as [assets, contracts].

    ``assets`` contains LatentBatch plans or dictionaries accepted by LatentBatch
    (including ``strikes``). Shared keyword arguments supply dictionary defaults.
    Every asset has its own fields, curves, mesh coordinates, strikes, and initial
    state. Within an asset, the existing LatentBatch compatibility contract holds.

    Assets with equal mesh *shapes*, coefficient times, horizon, scheme, step
    count, event schedules, device and contract count execute together. Different coordinates and
    curves use different QuantLib bases/operators in the same tensor batch.
    Incompatible schedules/shapes run in separate groups, preserving asset order;
    no grid, date, boundary or event is changed. All assets must have equal
    contract counts and use one device. American/Bermudan exercise and fixed
    spot cash dividends (one latent factor) retain QuantLib event ordering.

    Fields are float64 tensors with a leading asset axis:
    drift [A,T,N,m], diffusion [A,T,N,m+1,W], initial_state [A,m+1].
    Alternatively pass one tensor per asset, to support differently shaped
    groups. N uses QuantLib's axis-zero-fastest order. The internal state is
    [A,N,C]; it represents [assets, contracts, spatial dimensions].

    Full AD retains higher derivatives. First-order equation adjoints checkpoint
    every ``checkpoint_steps`` actual QuantLib intervals (including the theta
    snapshot) and recompute segments during backward. This returns exact discrete
    VJPs to the original fields; no solution Jacobian is formed. Memory still
    scales with A*C*N and saved segment boundaries. CUDA uses checked tensor PCR
    and requires strictly row-diagonally-dominant forward systems. Device flags
    are checked once per group forward/backward, without per-solve host waits.

    Natural cubic cardinal polynomials are obtained from pinned QuantLib at plan
    construction. Financial readout and initial-state gradients stay on device.
    Grids/dates/strikes remain fixed. Per-call ``markets`` override rates and
    dated discount factors with graph-connected CPU float64 scalars; the
    original QuantLib forwards transfer to the plan device with their graph.
    No torch.func/compile contract.
    """

    def __init__(self, assets, *, checkpoint_steps=8, **shared):
        self._checkpoint_steps = _integer("checkpoint_steps", checkpoint_steps, 1)
        plans = []
        for asset in assets:
            if isinstance(asset, LatentBatch):
                if shared:
                    raise ValueError("shared configuration cannot override an existing LatentBatch")
                plans.append(asset)
            else:
                plans.append(LatentBatch(**{**shared, **dict(asset)}))
        if not plans:
            raise ValueError("assets must not be empty")
        if len({p.contracts for p in plans}) != 1:
            raise ValueError("all assets must have equal contract counts; use separate plans for ragged portfolios")
        if len({p._device for p in plans}) != 1:
            raise ValueError("all assets must use the same device")
        self._plans = tuple(plans)
        grouped = {}
        for index, plan in enumerate(plans):
            grouped.setdefault(plan._group_key, []).append(index)
        self._groups = tuple(tuple(indices) for indices in grouped.values())
        self._native = tuple(torch.classes.quantlibtorch.MultiAssetBatch(
            [plans[i]._native for i in indices]) for indices in self._groups)
        self._indices = tuple(torch.tensor(indices, device=plans[0]._device) for indices in self._groups)
        self._interval_times = tuple(native.interval_times() for native in self._native)

    @classmethod
    def from_contracts(cls, assets, *, checkpoint_steps=8, **shared):
        """Each item is a list of fd_latent-style contract dictionaries.

        Within each asset all fixed numerical/model configuration must match;
        only strike and call/put may vary. Across assets compatible execution
        groups are formed without requiring equal coefficients or markets.
        """
        return cls([LatentBatch.from_contracts(contracts, **shared) for contracts in assets],
                   checkpoint_steps=checkpoint_steps)

    @property
    def assets(self):
        return len(self._plans)

    @property
    def contracts(self):
        return self._plans[0].contracts

    @property
    def groups(self):
        """Asset indices in each independently executed compatibility group."""
        return self._groups

    @property
    def plans(self):
        """Fixed per-asset plans, useful for serial references and state grids."""
        return self._plans

    def price(self, latent_drift, diffusion, initial_state, *, mode="adjoint", markets=None):
        """Price all assets, optionally with one market-override dictionary each.

        Each dictionary accepts rate, dividend_yield, discount_curve and
        dividend_curve. Missing entries inherit that asset's fixed plan values.
        Pass curve=None explicitly to replace a configured curve by a flat rate.
        Market graphs are fresh on every call, including checkpointed adjoints.
        """
        if mode not in ("full", "adjoint"):
            raise ValueError("mode must be full or adjoint")
        if markets is not None:
            markets = list(markets)
            if len(markets) != self.assets or any(not isinstance(m, dict) for m in markets):
                raise ValueError("markets must contain one override dictionary per asset")
        fields = (latent_drift, diffusion, initial_state)
        for name, value in zip(("latent_drift", "diffusion", "initial_state"), fields):
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != self.assets:
                    raise ValueError(f"{name} must have a leading dimension of {self.assets} assets")
            elif not isinstance(value, (list, tuple)) or len(value) != self.assets or any(
                    not isinstance(v, torch.Tensor) for v in value):
                raise TypeError(f"{name} must be a tensor or one tensor per asset")

        def select(value, indices, device_indices):
            if isinstance(value, torch.Tensor):
                if len(indices) == self.assets:
                    return value
                # A wrong device is diagnosed explicitly, not silently moved.
                if value.device != device_indices.device:
                    raise ValueError("fields must be on the plan device")
                return value.index_select(0, device_indices)
            return torch.stack([value[i] for i in indices])

        outputs = [None]*self.assets
        for indices, device_indices, native, intervals in zip(
                self._groups, self._indices, self._native, self._interval_times):
            inputs = [select(v, indices, device_indices) for v in fields]
            if markets is None:
                result = native.price(*inputs, mode == "adjoint", self._checkpoint_steps)
            else:
                forwards = torch.stack([self._plans[i]._native.forward_rates(
                    *intervals, *self._plans[i]._market_inputs(markets[i])) for i in indices], 1)
                forwards = forwards.to(self._plans[indices[0]]._device)
                result = native.price_market(*inputs, mode == "adjoint", self._checkpoint_steps,
                    forwards[..., :1], forwards[..., 1:])
            for index, row in zip(indices, result.unbind(0)):
                outputs[index] = row
        return torch.stack(outputs)
