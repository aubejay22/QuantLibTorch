"""Isolated Phase 23 neural calibration timing, memory, accuracy and GPU controls.

Use one worker at a time. Timing excludes plan construction but includes model
evaluation, residuals, backward, logging materialization, gradient zeroing and
graph cleanup. Cold setup is reported separately; memory and profiler workers
are separate from timing. Every report fingerprints source, market and binary.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))


def prepare(path):
    contracts = [dict(strike=k, maturity=t, scale=.05) for t in
                 ['2025-04-01', '2025-07-01', '2026-01-01'] for k in range(80, 121, 5)]
    meshes = {name: dict(nx=nx, nv=nv, steps_per_year=steps) for name, nx, nv, steps in [
        ('small', 21, 9, 12), ('medium', 41, 17, 24), ('training', 81, 17, 24),
        ('refined', 81, 33, 48), ('fine', 161, 65, 96),
        ('time', 81, 17, 48), ('variance', 81, 33, 24), ('stock', 161, 17, 24),
        ('diagnostic', 9, 5, 4), ('three_dimensional', 9, 5, 4)]}
    request = dict(contracts=contracts, meshes=meshes)
    completed = subprocess.run([sys.executable, str(ROOT/'scripts/calibration_stock_reference.py')],
                               input=json.dumps(request), text=True, capture_output=True, check=True)
    data = json.loads(completed.stdout)
    data.update(valuation_date='2025-01-01', spot=100., rate=.03, dividend_yield=.01,
                times=[i/8 for i in range(9)], heldout=[], target_model='Independent analytic Heston, QuantLib 1.40')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['prepare', 'timing', 'memory', 'profile', 'convergence', 'fit', 'gradients'])
    p.add_argument('--market', type=Path, default=ROOT/'docs/benchmarks/packed_phase23_market.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mesh', default='medium')
    p.add_argument('--contracts', type=int, choices=[1, 3, 9, 27], default=9)
    p.add_argument('--maturities', choices=['common', 'multiple'], default='common')
    p.add_argument('--execution', choices=['serial', 'batched'], default='batched')
    p.add_argument('--mode', choices=['full', 'adjoint'], default='adjoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--latent-dimensions', type=int, choices=[1, 2], default=1)
    p.add_argument('--parameters', type=Path)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--iterations', type=int, default=15)
    p.add_argument('--windows-qos', choices=['system', 'high'], default='high')
    p.add_argument('--affinity', type=int, default=8)
    args = p.parse_args()
    if args.task == 'prepare':
        prepare(args.output)
        return
    from runtime_controls import worker_qos
    runtime = worker_qos(args.windows_qos, args.affinity)
    import torch
    from calibrate_neural_sde import NeuralVarianceModel, make_problem, restore, parameter_values, train
    from investigate_fd import process_memory
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cuda = torch.device(args.device).type == 'cuda'
    def sync():
        if cuda:
            torch.cuda.synchronize()
    data = json.loads(args.market.read_text())
    if args.maturities == 'common':
        rows = [q for q in data['quotes'] if q['maturity'] == '2026-01-01']
        if args.contracts == 1:
            rows = [min(rows, key=lambda q: abs(q['strike']-100))]
        elif args.contracts == 3:
            rows = [rows[0], rows[len(rows)//2], rows[-1]]
        elif args.contracts == 27:
            raise ValueError('27 distinct contracts require --maturities multiple')
        data['quotes'] = rows
    else:
        if args.contracts == 9:
            data['quotes'] = [q for q in data['quotes'] if q['strike'] in (85, 100, 115)]
        elif args.contracts != 27:
            raise ValueError('multiple maturities requires 9 or 27 contracts')
    before_setup = process_memory()
    start = time.perf_counter()
    model = NeuralVarianceModel(args.latent_dimensions).to(args.device)
    if args.parameters:
        restore(model, json.loads(args.parameters.read_text())['parameters'])
    problem = make_problem(data, args.mesh, latent_dimensions=args.latent_dimensions,
                           execution=args.execution, device=args.device)
    sync()
    setup_seconds = time.perf_counter()-start
    native = next(f for f in (ROOT/'python/quantlibtorch/_native').glob('*native.*') if f.suffix in ('.dll', '.so', '.dylib'))
    paths = [ROOT/f for f in ['cpp/packed_pde.cpp', 'cpp/thomas_arithmetic.cpp',
        'scripts/prepare_packed.py', 'python/quantlibtorch/batch.py', 'python/quantlibtorch/calibration.py',
        'examples/calibrate_neural_sde.py', 'scripts/benchmark_packed_calibration.py']]+[args.market]
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), config=vars(args), runtime=runtime,
        torch=torch.__version__, python=sys.version, source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        binary_sha256=hashlib.sha256(native.read_bytes()).hexdigest(), baseline_git='12ec6ee',
        setup_seconds=setup_seconds, memory_before_setup=before_setup, memory_after_setup=process_memory(),
        nodes=len(problem.states), groups=[list(indices) for indices, _ in problem.groups],
        parameters=parameter_values(model))
    if cuda:
        result['gpu'] = dict(name=torch.cuda.get_device_name(), allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str)+'\n', encoding='utf-8', newline='\n')

    def run():
        sync()
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        forward = backward = 0.
        prices = [None]*len(problem.quotes)
        terms = []
        groups = problem.groups if args.execution == 'batched' else [((i,), None) for i in range(len(problem.quotes))]
        for indices, plan in groups:
            sync(); begin = time.perf_counter()
            fields, initial = problem._fields(model), problem._initial(model.initial_latent)
            values = plan.price(*fields, initial, mode=args.mode) if plan else torch.stack([
                problem._price(fields, initial, problem.quotes[indices[0]], args.mode)])
            term = ((values-problem.targets[list(indices)])/problem.scales[list(indices)]).square().sum()/len(prices)
            sync(); middle = time.perf_counter()
            term.backward()
            sync(); end = time.perf_counter()
            forward += middle-begin; backward += end-middle
            for i, value in zip(indices, values.detach().cpu().tolist()):
                prices[i] = value
            terms.append(float(term.detach()))
            del fields, initial, values, term
        begin = time.perf_counter()
        penalty = model.penalty()
        penalty.backward()
        sync()
        backward += time.perf_counter()-begin
        gradients = {k: v.grad.detach().cpu().tolist() for k, v in model.named_parameters()}
        penalty_value = float(penalty.detach())
        del penalty
        sync()
        elapsed = time.perf_counter()-start
        clean = time.perf_counter()
        model.zero_grad(set_to_none=True)
        gc.collect()
        sync()
        cleanup = time.perf_counter()-clean
        return dict(seconds=elapsed+cleanup, forward_seconds=forward, backward_seconds=backward,
                    cleanup_seconds=cleanup, loss=sum(terms), penalty=penalty_value, prices=prices, gradients=gradients)

    if args.task == 'timing':
        result['warmups'] = [run() for _ in range(args.warmup)]
        result['samples'] = []
        for _ in range(args.repeats):
            result['samples'].append(run()); save()
        result['median'] = {key: statistics.median(row[key] for row in result['samples']) for key in
                            ['seconds', 'forward_seconds', 'backward_seconds', 'cleanup_seconds']}
    elif args.task == 'memory':
        gc.collect()
        result['baseline_memory'] = process_memory()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        result['cycles'] = []
        for _ in range(max(3, args.repeats)):
            row = run()
            row['after_cleanup'] = process_memory()
            if cuda:
                row['cuda'] = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                                   peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            result['cycles'].append(row); save()
        result['peak_growth_bytes'] = max(r['after_cleanup']['peak_rss_bytes'] for r in result['cycles'])-result['baseline_memory']['peak_rss_bytes']
    elif args.task == 'profile':
        result['warmup'] = run()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            result['evaluation'] = run()
        result['events'] = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                                total_cpu_us=e.cpu_time_total) for e in prof.key_averages()]
    elif args.task == 'fit':
        result['fit'] = train(data, mode=args.mode, iterations=args.iterations, level=args.mesh,
            output=args.output.with_suffix('.training.json'), execution=args.execution, device=args.device,
            resume=parameter_values(model))
        result['initial_parameters'] = result['parameters']
        result['parameters'] = result['fit']['parameters']
    elif args.task == 'gradients':
        result['evaluation'] = run()
        named = dict(model.named_parameters())
        originals = {k: v.detach().clone() for k, v in named.items()}
        directions = {k: .05*torch.cos(torch.arange(v.numel(), dtype=torch.float64,
                            device=v.device)).reshape_as(v) for k, v in named.items()}
        result['gradient_checks'] = []
        for group in ['network', 'initial_state']:
            selected = [k for k in named if (k == 'raw_initial') == (group == 'initial_state')]
            ad = sum(float((torch.tensor(result['evaluation']['gradients'][k], dtype=torch.float64,
                             device=args.device)*directions[k]).sum()) for k in selected)
            for bump in [1e-3, 3e-4]:
                values = []
                with torch.no_grad():
                    for sign in [1., -1.]:
                        for k in selected:
                            named[k].copy_(originals[k]+sign*bump*directions[k])
                        value = problem.loss(model, model.initial_latent, mode=args.mode)[0]+model.penalty()
                        values.append(float(value))
                    for k in selected:
                        named[k].copy_(originals[k])
                fd = (values[0]-values[1])/(2*bump)
                result['gradient_checks'].append(dict(group=group, bump=bump, ad=ad,
                                                      finite_difference=fd, absolute_error=abs(ad-fd)))
    elif args.task == 'convergence':
        # Frozen network and initial-state parameters: per-price sensitivities,
        # not just an aggregate calibration gradient that can cancel errors.
        result['convergence'] = []
        for level in ['small', 'medium', 'training', 'time', 'variance', 'stock', 'refined', 'fine']:
            local = make_problem(data, level, execution='batched', device=args.device)
            params = tuple(model.parameters())
            start = time.perf_counter()
            prices = local.prices(model, model.initial_latent, mode=args.mode)
            directional, state = [], []
            for j in range(len(prices)):
                gradients = torch.autograd.grad(prices[j], params, retain_graph=j+1<len(prices))
                directional.append(sum(float((g*.05*torch.cos(torch.arange(g.numel(), device=g.device, dtype=torch.float64).reshape_as(g))).sum())
                                       for p, g in zip(params, gradients) if p is not model.raw_initial))
                state.append(gradients[0].detach().cpu().tolist())  # raw_initial is first named parameter
            row = dict(mesh=level, prices=prices.detach().cpu().tolist(), neural_directional=directional,
                       raw_initial_sensitivities=state, seconds=time.perf_counter()-start,
                       target_rmse=float(((prices-local.targets)**2).mean().sqrt().detach()))
            result['convergence'].append(row)
            del prices, local
            gc.collect(); save()
    save()
    print(json.dumps(dict(output=str(args.output), median=result.get('median'), peak_growth_bytes=result.get('peak_growth_bytes'))))


if __name__ == '__main__':
    # Importing torch here would initialize CPU pools before worker affinity.
    main()
