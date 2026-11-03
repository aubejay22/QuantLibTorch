"""Run a neural local-volatility pricing loss and its parameter VJP on CPU.

Build/install the package first, then run:
    python examples/neural_local_vol.py --mode full
    python examples/neural_local_vol.py --mode adjoint

This is a gradient demonstration, not an identified calibration from one price.
"""

import argparse

import torch

from quantlibtorch import european_call, log_moneyness_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "adjoint"), default="adjoint")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    steps, nodes, maturity = 12, 39, 1.0
    space = log_moneyness_grid(nodes, log_moneyness_min=-1.0, log_moneyness_max=1.0)
    time = torch.arange(steps, dtype=torch.float64) * maturity / steps
    t, x = torch.meshgrid(time, space, indexing="ij")
    features = torch.stack((t, x), dim=-1)
    network = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).double()
    # Bounded positive volatility, with fixed risk-neutral drift r-q in S-space.
    sigma = 0.1 + 0.3 * torch.sigmoid(network(features).squeeze(-1))
    price = european_call(
        sigma,
        maturity=maturity,
        log_moneyness_min=-1.0,
        log_moneyness_max=1.0,
        mode=args.mode,
    )
    target = 9.0
    loss = (price - target).square()
    loss.backward()
    gradient_norm = torch.sqrt(sum(p.grad.square().sum() for p in network.parameters()))
    print(f"mode={args.mode}, grid={steps}x{nodes}, device=cpu, dtype=float64")
    print(f"price={price.item():.8f}, target={target:.8f}, loss={loss.item():.8f}")
    print(f"network gradient norm={gradient_norm.item():.8f}")


if __name__ == "__main__":
    main()
