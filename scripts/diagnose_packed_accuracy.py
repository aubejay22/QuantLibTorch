"""Frozen-model price AND sensitivity refinement against independent oracles."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/benchmarks/packed_phase23_accuracy.json')
    parser.add_argument('--market', type=Path, default=ROOT/'docs/benchmarks/packed_phase23_market.json')
    args = parser.parse_args()
    from runtime_controls import worker_qos
    runtime = worker_qos('high', 8)
    import torch
    from quantlibtorch import LatentBatch, state_grid
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = json.loads(args.market.read_text())
    strikes = [85., 100., 115.]
    contracts = [dict(strike=k, maturity='2026-01-01') for k in strikes]
    def oracle(parameters, black=False):
        request = dict(contracts=contracts, heston=parameters, black_scholes=black, volatility=.2)
        completed = subprocess.run([sys.executable, str(ROOT/'scripts/calibration_stock_reference.py')],
            input=json.dumps(request), text=True, capture_output=True, check=True)
        return torch.tensor([q['price'] for q in json.loads(completed.stdout)['quotes']], dtype=torch.float64)
    params = [1.5, .04, .3, -.6, .045]
    targets = oracle(params)
    reference_gradients = []
    for j, h in [(2, 1e-4), (3, 1e-4), (4, 1e-5)]:
        plus, minus = params.copy(), params.copy()
        plus[j] += h; minus[j] -= h
        reference_gradients.append(((oracle(plus)-oracle(minus))/(2*h)).tolist())
    reference_gradients = torch.tensor(reference_gradients, dtype=torch.float64).T
    bs_targets = oracle(params, black=True)
    d1 = torch.tensor([(math.log(100/k)+(.03-.01+.5*.2**2))/.2 for k in strikes], dtype=torch.float64)
    vega = 100*math.exp(-.01)*torch.exp(-.5*d1.square())/math.sqrt(2*math.pi)
    binary = ROOT/'python/quantlibtorch/_native/quantlibtorch_native.dll'
    if not binary.exists():
        binary = next(p for p in binary.parent.glob('*native.*') if p.suffix in ('.so', '.dylib'))
    paths = [Path(__file__), ROOT/'cpp/packed_pde.cpp', ROOT/'python/quantlibtorch/batch.py', args.market]
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        heston_reference_prices=targets.tolist(), heston_sensitivity_columns=['sigma', 'rho', 'v0'],
        heston_reference_sensitivities=reference_gradients.tolist(),
        black_reference_prices=bs_targets.tolist(), black_reference_vega=vega.tolist(), records=[])
    for level in ['small', 'medium', 'training', 'time', 'variance', 'stock', 'refined', 'fine']:
        mesh = data['meshes'][level]
        axes = [mesh['x'], mesh['v']]
        x = state_grid(axes)
        plan = LatentBatch(strikes, axes=axes, time_steps=mesh['steps_per_year'])
        p = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        v = x[:, 1]; root = v.sqrt(); zero = torch.zeros_like(root)
        def evaluate(parameters):
            kappa, theta, sigma, rho, v0 = parameters
            b = (kappa*(theta-v))[None, :, None]
            l = torch.stack((torch.stack((root, zero), -1),
                torch.stack((sigma*rho*root, sigma*(1-rho.square()).sqrt()*root), -1)), -2)[None]
            z = torch.stack((v0*0+math.log(100.), v0))
            return plan.price(b, l, z)
        start = time.perf_counter()
        prices = evaluate(p)
        grad = torch.stack([torch.autograd.grad(prices[j], p, retain_graph=j<2)[0][[2, 3, 4]] for j in range(3)])
        sigma = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        b = (.04-v)[None, :, None]
        l = torch.stack((torch.stack((sigma.expand_as(v), zero), -1),
                         torch.stack((zero, .1*root), -1)), -2)[None]
        bs_prices = plan.price(b, l, torch.tensor([math.log(100.), .045], dtype=torch.float64))
        bs_vega = torch.stack([torch.autograd.grad(bs_prices[j], sigma, retain_graph=j<2)[0] for j in range(3)])
        with torch.no_grad():
            put_plan = LatentBatch(strikes, axes=axes, time_steps=mesh['steps_per_year'], option_types=['put']*3)
            puts = put_plan.price(b, l, torch.tensor([math.log(100.), .045], dtype=torch.float64))
            parity = bs_prices.detach()-puts-(100*math.exp(-.01)-torch.tensor(strikes, dtype=torch.float64)*math.exp(-.03))
        row = dict(mesh=level, nodes=[len(a) for a in axes], steps=mesh['steps_per_year'],
            heston_prices=prices.detach().tolist(), heston_sensitivities=grad.tolist(),
            heston_max_price_error=float((prices.detach()-targets).abs().max()),
            heston_max_sensitivity_errors=(grad-reference_gradients).abs().max(0).values.tolist(),
            black_prices=bs_prices.detach().tolist(), black_vega=bs_vega.tolist(),
            black_max_price_error=float((bs_prices.detach()-bs_targets).abs().max()),
            black_max_vega_error=float((bs_vega-vega).abs().max()),
            black_max_put_call_parity_error=float(parity.abs().max()), seconds=time.perf_counter()-start)
        result['records'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
