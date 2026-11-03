"""Independent analytic-Heston and frozen neural price/sensitivity refinement."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from benchmark_multi_asset import ROOT, fingerprints
sys.path.insert(0, str(ROOT/'examples'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from runtime_controls import worker_qos
    runtime = worker_qos('high', 8)
    import torch
    from quantlibtorch import MultiAssetBatch, LatentBatch, state_grid
    from multi_asset_neural_sde import NeuralAssets, configuration
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dtype, device = torch.float64, args.device
    params = [[1.5, .04, .3, -.6, .045], [1.3, .05, .26, -.4, .055]]
    strikes = [85., 100., 115.]
    def oracle(p, i):
        request = dict(contracts=[dict(strike=k, maturity='2026-01-01') for k in strikes],
                       heston=p, rate=.03+.005*i, dividend_yield=.01)
        process = subprocess.run([sys.executable, str(ROOT/'scripts/calibration_stock_reference.py')],
            input=json.dumps(request), capture_output=True, text=True, check=True)
        return torch.tensor([q['price'] for q in json.loads(process.stdout)['quotes']], dtype=dtype)
    references, sensitivities = [], []
    for i, p in enumerate(params):
        references.append(oracle(p, i)); bumps=[]
        for k, h in [(2, 1e-4), (3, 1e-4), (4, 1e-5)]:
            plus, minus = p.copy(), p.copy(); plus[k]+=h; minus[k]-=h
            bumps.append((oracle(plus, i)-oracle(minus, i))/(2*h))
        sensitivities.append(torch.stack(bumps, -1))
    references, sensitivities = torch.stack(references), torch.stack(sensitivities)
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), device=device, runtime=runtime,
                  **fingerprints(), heston_reference=references.tolist(),
                  heston_sensitivity_columns=['sigma', 'rho', 'v0'], heston_reference_sensitivities=sensitivities.tolist(),
                  heston=[], neural=[])
    result['diagnostic_source_sha256'] = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in
        ['scripts/diagnose_multi_asset.py', 'scripts/calibration_stock_reference.py', 'docs/benchmarks/packed_phase23_market.json']}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    market = json.loads((ROOT/'docs/benchmarks/packed_phase23_market.json').read_text())
    for level in ['small', 'training', 'time', 'variance', 'stock', 'refined', 'fine']:
        mesh = market['meshes'][level]; axes = [mesh['x'], mesh['v']]
        configs = [dict(strikes=strikes, axes=axes, time_steps=mesh['steps_per_year'], rate=.03+.005*i) for i in range(2)]
        plan = MultiAssetBatch(configs, device=device, checkpoint_steps=8)
        p = torch.tensor(params, dtype=dtype, device=device, requires_grad=True)
        v = state_grid(axes)[:, 1].to(device); root = v.sqrt(); zero = torch.zeros_like(v)
        b = (p[:, 0, None]*(p[:, 1, None]-v))[:, None, :, None]
        l = torch.stack([torch.stack((torch.stack((root, zero), -1), torch.stack((row[2]*row[3]*root,
            row[2]*(1-row[3].square()).sqrt()*root), -1)), -2) for row in p])[:, None]
        z = torch.stack((p[:, 4]*0+math.log(100.), p[:, 4]), -1)
        prices = plan.price(b, l, z)
        grads = torch.stack([torch.autograd.grad(prices[:, j].sum(), p, retain_graph=j<2)[0][:, [2, 3, 4]] for j in range(3)], 1)
        expected = torch.stack([LatentBatch(**config).price(b[i].detach().cpu(), l[i].detach().cpu(),
                               z[i].detach().cpu()) for i, config in enumerate(configs)])
        result['heston'].append(dict(mesh=level, prices=prices.detach().cpu().tolist(), sensitivities=grads.cpu().tolist(),
            max_price_error=float((prices.detach().cpu()-references).abs().max()),
            max_sensitivity_error=(grads.cpu()-sensitivities).abs().amax((0, 1)).tolist(),
            max_existing_cpu_price_difference=float((prices.detach().cpu()-expected).abs().max())))
        save(); print('heston', level, flush=True)
        del plan, prices, grads, b, l, z, p; gc.collect()
    for nx, ny, steps, label in [(21,9,12,'coarse'), (41,9,12,'stock'), (21,17,12,'latent'),
                                 (21,9,24,'time'), (41,17,24,'combined'), (81,33,48,'fine'), (161,65,96,'finest')]:
        configs, times, spots, _ = configuration(2, 3, nx, ny, steps)
        model = NeuralAssets([c['axes'] for c in configs], times, spots, device)
        if 'neural_parameters' not in result:
            result['neural_parameters'] = {k: v.detach().cpu().tolist() for k, v in model.named_parameters()}
        plan = MultiAssetBatch(configs, device=device, checkpoint_steps=8)
        prices = plan.price(*model.fields())
        parameters = tuple(model.parameters())
        direction = [torch.cos(torch.arange(v.numel(), dtype=dtype, device=device)).reshape_as(v)*.05 for v in parameters]
        projections, initial = [], []
        names = [name for name, _ in model.named_parameters()]
        for i in range(2):
            projected_row, initial_row = [], []
            for j in range(3):
                grads = torch.autograd.grad(prices[i, j], parameters, retain_graph=(i, j)!=(1, 2))
                projected_row.append(float(sum((g*d).sum() for name, g, d in zip(names, grads, direction)
                                               if name!='raw_initial')))
                initial_row.append(grads[names.index('raw_initial')][i].cpu().tolist())
            projections.append(projected_row); initial.append(initial_row)
        result['neural'].append(dict(mesh=label, nodes=[nx,ny], steps=steps, prices=prices.detach().cpu().tolist(),
                                      neural_directional=projections, raw_initial_sensitivities=initial))
        save(); print('neural', label, flush=True)
        del plan, prices, model, grads, parameters; gc.collect()
    save()


if __name__ == '__main__':
    main()
