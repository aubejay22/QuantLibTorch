"""Price and differentiate with reusable financial objects and a neural model.

Run ``python examples/pricing_layer.py --engine analytic`` for spot/rate Greeks,
or ``--engine fd --mode adjoint`` for a neural Bermudan put with a cash dividend.
"""
from __future__ import annotations

import argparse
from datetime import date

import torch

from quantlibtorch import (
    AnalyticEuropeanEngine, EquityMarket, Exercise, FdVanillaEngine,
    PricingLayer, VanillaOption,
)


class VolatilityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(),
                                           torch.nn.Linear(4, 1)).double()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return .05 + .35 * torch.sigmoid(self.network(feature).squeeze())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("analytic", "fd"), default="analytic")
    parser.add_argument("--mode", choices=("full", "adjoint"), default="full")
    args = parser.parse_args()
    if args.engine == "analytic" and args.mode != "full":
        parser.error("the analytic engine supports full AD only")
    torch.manual_seed(7)
    expiry, mid = date(2026, 1, 1), date(2025, 7, 1)
    if args.engine == "analytic":
        option = VanillaOption(100., expiry)
        market = EquityMarket(spot=100., rate=.03, dividend_yield=.01)
        engine = AnalyticEuropeanEngine()
    else:
        option = VanillaOption(100., expiry, "put", Exercise.bermudan([mid, expiry]))
        market = EquityMarket(spot=97., cash_dividends=[(mid, 1.5)])
        engine = FdVanillaEngine(time_steps=30, grid_points=61, mode=args.mode)
    layer = PricingLayer(option, market, engine, model=VolatilityModel())
    feature = torch.tensor([0.], dtype=torch.float64)
    overrides = {}
    if args.engine == "analytic":
        overrides = {"spot": torch.tensor(market.spot, dtype=torch.float64, requires_grad=True),
                     "rate": torch.tensor(market.rate, dtype=torch.float64, requires_grad=True)}
    price = layer(feature, **overrides)
    price.backward()
    norm = torch.stack([p.grad.square().sum() for p in layer.parameters()]).sum().sqrt()
    print(f"{args.engine} / {args.mode}: price={price.item():.8f}, neural gradient norm={norm.item():.8f}")
    if overrides:
        print(f"delta={overrides['spot'].grad.item():.8f}, rho={overrides['rate'].grad.item():.8f}")


if __name__ == "__main__":
    main()
