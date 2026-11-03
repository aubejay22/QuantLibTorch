"""Train bounded joint-state drift/diffusion fields through QuantLib pricing.

Small synthetic identification demonstration, not a market calibration or an
identifiability claim. Both student and target use the same fixed PDE mesh.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from quantlibtorch import fd_latent, state_grid


class NeuralDiffusion(torch.nn.Module):
    def __init__(self, latent_dimensions):
        super().__init__()
        self.m = latent_dimensions
        self.net = torch.nn.Sequential(torch.nn.Linear(self.m+2, 6), torch.nn.Tanh(),
                                       torch.nn.Linear(6, self.m+(self.m+1)**2)).double()

    def forward(self, features):
        raw = self.net(features)
        drift = .08*torch.tanh(raw[..., :self.m])
        # Bounded full loading matrix. LL^T is PSD; each stock row is bounded.
        loading = (.18*torch.eye(self.m+1, dtype=features.dtype)
                   + .07*torch.tanh(raw[..., self.m:].reshape(*raw.shape[:-1], self.m+1, self.m+1)))
        return drift, loading


def run(mode="adjoint", latent_dimensions=1, iterations=5):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    m = latent_dimensions
    axes = [[math.log(s) for s in [45., 65., 85., 100., 120., 150., 210.]]]+[[-1., -.3, .3, 1.]]*m
    grid = state_grid(axes)
    times = [0., .5, 1.]
    features = torch.stack([torch.cat((torch.full((len(grid), 1), t, dtype=torch.float64),
        grid[:, :1]-math.log(100.), grid[:, 1:]), -1) for t in times])
    network = NeuralDiffusion(m)
    initial = torch.nn.Parameter(torch.tensor([math.log(101.)]+[.08]*m, dtype=torch.float64))
    strikes = [95., 105.]

    def prices(drift, factor, z):
        return torch.stack([fd_latent(drift, factor, z, axes=axes, times=times, strike=k,
                            time_steps=4, mode=mode) for k in strikes])

    with torch.no_grad():
        drift, factor = network(features)
        target = prices(drift+.02, factor*1.15, initial)
    optimizer = torch.optim.Adam(network.parameters(), lr=.015)
    history = []
    first_gradients = None
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        initial.grad = None
        drift, factor = network(features)
        value = prices(drift, factor, initial)
        loss = (value-target).square().mean()
        loss.backward()
        if first_gradients is None:
            first_gradients = {
                "network_norm": float(torch.cat([p.grad.flatten() for p in network.parameters()]).norm()),
                "latent_drift_output_norm": float(network.net[-1].bias.grad[:m].norm()),
                "diffusion_output_norm": float(network.net[-1].bias.grad[m:].norm()),
                "initial_state": initial.grad.tolist(),
            }
        history.append(float(loss.detach()))
        optimizer.step()
    with torch.no_grad():
        fitted = prices(*network(features), initial)
        final_loss = float((fitted-target).square().mean())
    return {"mode": mode, "latent_dimensions": m, "nodes": len(grid), "time_steps": 4,
            "iterations": iterations, "loss_history": history, "final_loss": final_loss,
            "target_prices": target.tolist(), "fitted_prices": fitted.tolist(),
            "first_gradients": first_gradients,
            "note": "Network parameters trained; initial-state gradients measured on fixed mesh. Synthetic demonstration only."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "adjoint"], default="adjoint")
    parser.add_argument("--latent-dimensions", type=int, choices=[1, 2], default=1)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    print(json.dumps(run(args.mode, args.latent_dimensions, args.iterations), indent=2))
