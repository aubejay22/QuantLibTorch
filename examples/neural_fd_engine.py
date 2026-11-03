"""Neural local volatility through QuantLib's actual Bermudan pricing engine."""
import argparse
from datetime import date
import math

import torch

from quantlibtorch import fd_vanilla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "adjoint"), default="adjoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    network = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)).double()
    times = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)
    log_moneyness = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    t, x = torch.meshgrid(times, log_moneyness, indexing="ij")
    sigma = 0.10 + 0.30 * torch.sigmoid(network(torch.stack((t, x), dim=-1)).squeeze(-1))
    price = fd_vanilla(
        sigma, times=times.tolist(), log_spots=(log_moneyness + math.log(100.0)).tolist(),
        spot=100.0, strike=100.0, option_type="put", exercise="bermudan",
        exercise_dates=(date(2025, 7, 1), date(2026, 1, 1)),
        cash_dividends=((date(2025, 7, 1), 1.5),),
        time_steps=20, grid_points=41, mode=args.mode,
    )
    loss = (price - 8.5).square()
    loss.backward()
    norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in network.parameters()))
    print(f"mode={args.mode} engine=FdBlackScholesVanillaEngine exercise=bermudan")
    print(f"price={price.item():.10f} loss={loss.item():.10f} network_gradient_norm={norm.item():.10f}")


if __name__ == "__main__":
    main()
