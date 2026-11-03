"""Run from outside the checkout with an installed wheel (no test dependencies)."""
import argparse
from datetime import date
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import torch
import quantlibtorch as qlt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not Path(qlt.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Package did not import from the clean environment")
    assert qlt.__version__ == importlib.metadata.version("quantlibtorch")
    dtype = torch.float64
    spot = torch.tensor(100., dtype=dtype, requires_grad=True)
    sigma = torch.tensor(.2, dtype=dtype, requires_grad=True)
    price = qlt.analytic_european(sigma, spot=spot, rate=.03, dividend_yield=.01)
    delta, vega = torch.autograd.grad(price, (spot, sigma))
    # Independent Black-Scholes vega (one year, continuous dividends).
    d1 = (.03-.01+.5*.2**2)/.2
    expected_vega = 100*math.exp(-.01)*math.exp(-.5*d1*d1)/math.sqrt(2*math.pi)
    assert abs(vega.item()-expected_vega) < 1e-10
    assert .4 < delta < .7 and 5 < price < 15
    scalar = {}
    for exercise in ("european", "american", "bermudan"):
        vol = torch.full((3, 41), .2, dtype=dtype, requires_grad=True)
        kw = dict(exercise=exercise, option_type="put", time_steps=12,
                  log_spots=torch.linspace(math.log(30), math.log(250), 41).tolist(),
                  times=[0., .5, 1.], mode="adjoint")
        if exercise == "bermudan":
            kw["exercise_dates"] = [date(2025, 7, 1), date(2026, 1, 1)]
        value = qlt.fd_vanilla(vol, **kw)
        grad, = torch.autograd.grad(value, vol)
        assert torch.isfinite(value) and value > 0 and torch.isfinite(grad).all() and grad.norm() > 0
        full = qlt.fd_vanilla(vol, **(kw | {"mode": "full"}))
        full_grad, = torch.autograd.grad(full, vol)
        torch.testing.assert_close(full, value, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(full_grad, grad, rtol=1e-10, atol=1e-10)
        scalar[exercise] = {"price": value.item(), "vega": grad.sum().item()}
    heston = qlt.HestonModel()
    heston_price = qlt.fd_heston(*heston(), mesh=qlt.heston_mesh(x_grid=11, v_grid=7, time_steps=3),
                                 time_steps=3, mode="adjoint")
    heston_grads = torch.autograd.grad(heston_price, tuple(heston()))
    assert heston_price > 0 and all(torch.isfinite(g) for g in heston_grads)
    shout_vol = torch.tensor(.2, dtype=dtype, requires_grad=True)
    shout = qlt.fd_shout(shout_vol, mesh=qlt.shout_mesh(.2, x_grid=21, time_steps=6),
                         time_steps=6, mode="adjoint")
    shout_grad, = torch.autograd.grad(shout, shout_vol)
    assert shout > 0 and torch.isfinite(shout_grad) and shout_grad > 0
    def packed(device):
        axes = [[math.log(100)+.15*(i-4) for i in range(9)], [-.4, -.2, 0., .2, .4]]
        config = dict(strikes=[95., 105.], axes=axes, times=[0.], time_steps=4)
        plan = qlt.MultiAssetBatch([config, config], device=device, checkpoint_steps=2)
        vol = torch.tensor(.2, dtype=dtype, device=device, requires_grad=True)
        drift = torch.zeros((2, 1, 45, 1), dtype=dtype, device=device)
        # Loading layout: [asset, time, flattened nodes, state=(logS,Y), Brownian].
        row0 = torch.stack((vol, vol*0))
        row1 = torch.stack((vol*0, vol*.3))
        loading = torch.stack((row0, row1)).expand(2, 1, 45, 2, 2)
        initial = torch.tensor([[math.log(100), 0.], [math.log(100), 0.]], dtype=dtype, device=device)
        values = plan.price(drift, loading, initial, mode="adjoint")
        grad, = torch.autograd.grad(values.sum(), vol)
        assert torch.isfinite(values).all() and torch.isfinite(grad) and grad.abs() > 0
        return values.detach().cpu(), grad.detach().cpu()
    cpu, cpu_grad = packed("cpu")
    cuda = None
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation required but no CUDA device is available")
    if torch.cuda.is_available():
        gpu, gpu_grad = packed("cuda")
        torch.testing.assert_close(gpu, cpu, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(gpu_grad, cpu_grad, rtol=1e-9, atol=1e-9)
        cuda = {"device": torch.cuda.get_device_name(), "prices": gpu.tolist(), "vega": gpu_grad.item()}
    result = dict(version=importlib.metadata.version("quantlibtorch"),
                  package_path=qlt.__file__, executable=sys.executable, python=sys.version,
                  torch=torch.__version__, native=json.loads(torch.ops.quantlibtorch.build_info()),
                  analytic={"price": price.item(), "delta": delta.item(), "vega": vega.item()},
                  heston={"price": heston_price.item(), "gradients": [g.item() for g in heston_grads]},
                  shout={"price": shout.item(), "vega": shout_grad.item()},
                  exercise=scalar, packed_cpu={"prices": cpu.tolist(), "vega": cpu_grad.item()}, cuda=cuda)
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
