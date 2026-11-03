"""Calibrate a neural stochastic-volatility model and its initial variance.

The benchmark uses independent analytic Heston quotes, with market data and
mesh coordinates supplied in a JSON file. See scripts/benchmark_calibration.py.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import time

import torch

from quantlibtorch import LatentCalibration, OptionQuote


class NeuralVarianceModel(torch.nn.Module):
    """CIR-based latent dynamics with smooth bounded neural corrections.

    Latent variables are variances. At zero, their diffusion vanishes and drift
    is inward. At large variance, drift remains mean reverting. A common stock
    Brownian driver and independent latent drivers guarantee PSD also for m>1.
    This restricted model class is deliberate; it is not a general boundary
    prescription for all neural SDEs. Fixed base scales anchor latent units.
    """
    def __init__(self, latent_dimensions=1, seed=29):
        super().__init__()
        self.m = latent_dimensions
        # Do not alter the caller's RNG state when constructing a benchmark.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = torch.nn.Sequential(torch.nn.Linear(self.m+2, 4), torch.nn.Tanh(),
                                           torch.nn.Linear(4, 1+3*self.m)).double()
        with torch.no_grad():
            self.net[-1].weight.mul_(.15)
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = .12
        self.raw_initial = torch.nn.Parameter(torch.full((self.m,), math.log(.03/(.12-.03)), dtype=torch.float64))

    def initial_latent(self):
        return .12*torch.sigmoid(self.raw_initial)

    def forward(self, times, states):
        nt, n = len(times), len(states)
        features = torch.cat((times[:, None, None].expand(nt, n, 1),
            (states[:, :1]-math.log(100.))[None].expand(nt, n, 1),
            ((states[:, 1:]-.04)/.08)[None].expand(nt, n, self.m)), -1)
        raw = self.net(features)
        variance = states[None, :, 1:].expand(nt, n, self.m)
        root = variance.sqrt()
        leverage = 1.+.3*torch.tanh(raw[..., 0])
        drift = 1.5*(.04-variance)+.15*variance*torch.tanh(raw[..., 1:1+self.m])
        volvol = .3*(1.+.2*torch.tanh(raw[..., 1+self.m:1+2*self.m]))
        correlation = -.6+.2*torch.tanh(raw[..., 1+2*self.m:])
        stock = torch.cat(((variance.mean(-1).sqrt()*leverage)[..., None],
                           torch.zeros(nt, n, self.m, dtype=states.dtype, device=states.device)), -1)
        latent_rows = []
        eye = torch.eye(self.m, dtype=states.dtype, device=states.device)
        for i in range(self.m):
            amplitude = volvol[..., i]*root[..., i]
            latent_rows.append(torch.cat(((amplitude*correlation[..., i])[..., None],
                (amplitude*torch.sqrt(1-correlation[..., i].square()))[..., None]*eye[i]), -1))
        return drift, torch.stack((stock, *latent_rows), -2)

    def penalty(self):
        # Penalize learned corrections, not the independent initial state.
        return 1e-4*sum(p.square().sum() for p in self.net.parameters())


def make_problem(data, level="calibration", count=None, latent_dimensions=1, *, heldout=False,
                 execution="serial", device="cpu"):
    mesh = data["meshes"][level]
    quotes = data["heldout" if heldout else "quotes"]
    if count is not None:
        quotes = quotes[:count]
    return LatentCalibration([OptionQuote(q["strike"], date.fromisoformat(q["maturity"]), q["price"],
                                          q.get("scale", 1.), q.get("option_type", "call")) for q in quotes],
        axes=[mesh["x"], *([mesh["v"]]*latent_dimensions)], times=data["times"],
        valuation_date=date.fromisoformat(data["valuation_date"]), spot=data["spot"],
        rate=data["rate"], dividend_yield=data["dividend_yield"],
        steps_per_year=mesh["steps_per_year"], minimum_steps=mesh.get("minimum_steps", 4),
        execution=execution, device=device)


def parameter_values(model):
    return {k: v.detach().tolist() for k, v in model.state_dict().items()}


def restore(model, values):
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in values.items()})


def train(data, *, mode="adjoint", iterations=15, level="calibration", output=None, resume=None,
          execution="serial", device="cpu"):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = NeuralVarianceModel()
    if resume:
        restore(model, resume)
    model.to(device)
    problem = make_problem(data, level, execution=execution, device=device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.5, max_iter=iterations,
        max_eval=2*iterations, history_size=10, tolerance_grad=1e-6,
        tolerance_change=1e-10, line_search_fn="strong_wolfe")
    history = []
    start = time.perf_counter()
    initial_parameters = parameter_values(model)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        before = time.perf_counter()
        loss, prices = problem.backward(model, model.initial_latent, mode=mode)
        penalty = model.penalty()
        penalty.backward()
        gradient_norm = float(torch.cat([p.grad.flatten() for p in model.parameters()]).norm())
        row = dict(evaluation=len(history), price_rmse=float((prices-problem.targets).square().mean().sqrt()),
                   scaled_loss=float(loss), penalty=float(penalty.detach()), gradient_norm=gradient_norm,
                   latent_state=model.initial_latent().detach().tolist(), prices=prices.tolist(),
                   seconds=time.perf_counter()-before)
        history.append(row)
        if output:
            Path(output).write_text(json.dumps(dict(mode=mode, mesh=level, initial_parameters=initial_parameters,
                history=history, parameters=parameter_values(model), elapsed_seconds=time.perf_counter()-start), indent=2)+"\n",
                encoding="utf-8", newline="\n")
        print(json.dumps(row), flush=True)
        return loss+penalty.detach()
    optimizer.step(closure)
    with torch.no_grad():
        prices = problem.prices(model, model.initial_latent, mode=mode)
    result = dict(mode=mode, mesh=level, execution=execution, device=str(device), initial_parameters=initial_parameters, history=history,
        parameters=parameter_values(model), final_prices=prices.tolist(),
        final_rmse=float((prices-problem.targets).square().mean().sqrt()),
        final_latent_state=model.initial_latent().detach().tolist(), elapsed_seconds=time.perf_counter()-start,
        optimizer="L-BFGS, strong Wolfe line search", history_kind="closure evaluations including line-search trials",
        accepted_iterations=optimizer.state[next(iter(model.parameters()))].get("n_iter",0))
    if output:
        Path(output).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--mode", choices=["full", "adjoint"], default="adjoint")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--mesh", default="calibration")
    parser.add_argument("--execution", choices=["serial", "batched"], default="serial")
    parser.add_argument("--output", type=Path, default=Path("build/calibration.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train(json.loads(args.data.read_text()), mode=args.mode, iterations=args.iterations, level=args.mesh,
          output=args.output, execution=args.execution)
