"""Measure full pricing gradients and Hessian-vector products in isolated workers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from investigate_fd import ROOT, native_fingerprint
from benchmark_fd_optimizations import FLAGS, LEVELS


def profile_curvature(torch, loss, base, direction):
    """Separate instrumentation from timed samples; keep all numerical outputs."""
    theta = base.clone().requires_grad_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with torch.profiler.record_function('curvature::forward'):
            value, objective = loss(theta)
        with torch.profiler.record_function('curvature::gradient'):
            gradient, = torch.autograd.grad(objective, theta, create_graph=True)
        with torch.profiler.record_function('curvature::hvp'):
            hvp, = torch.autograd.grad((gradient*direction).sum(), theta)
    numeric = dict(price=value.item(), gradient=gradient.detach().tolist(), hvp=hvp.tolist())
    del value, objective, gradient, hvp, theta
    operators, scopes, scoped_operators = (defaultdict(Counter) for _ in range(3))
    for event in profiler.events():
        if not event.name.startswith(('aten::', 'qlt::')):
            continue
        ancestor, phase, nearest, recursive = event.cpu_parent, 'unattributed', None, False
        while ancestor:
            if ancestor.name == event.name:
                recursive = True
            if nearest is None and ancestor.name.startswith('qlt::'):
                nearest = ancestor.name
            if ancestor.name.startswith('curvature::'):
                phase = ancestor.name.removeprefix('curvature::')
                break
            ancestor = ancestor.cpu_parent
        if event.name.startswith('qlt::'):
            if not recursive:
                entry = scopes[(phase, event.name)]
                entry['calls'] += 1
                entry['inclusive_cpu_us'] += event.cpu_time_total
        else:
            for entry in (operators[(phase, event.name)], scoped_operators[(phase, nearest, event.name)]):
                entry['calls'] += 1
                entry['self_cpu_us'] += event.self_cpu_time_total
    result = dict(numeric=numeric,
        scopes=[dict(phase=p, name=n, **v) for (p, n), v in scopes.items()],
        operators=[dict(phase=p, name=n, **v) for (p, n), v in operators.items()],
        scoped_operators=[dict(phase=p, scope=s, name=n, **v) for (p, s, n), v in scoped_operators.items()],
        aten_calls={p: sum(v['calls'] for (phase, _), v in operators.items() if phase == p)
                    for p, _ in operators})

    # Graph walks and saved-tensor hooks run outside profiling and timing.
    def graph(tensor):
        seen, counts, pending = set(), Counter(), [tensor.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            counts[type(node).__name__] += 1
            pending.extend(child for child, _ in node.next_functions)
        return dict(nodes=len(seen), node_types=dict(counts.most_common()))

    saved = []
    def pack(tensor):
        saved.append(tensor.numel()*tensor.element_size())
        return tensor
    theta = base.clone().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        value, objective = loss(theta)
        forward_saves = len(saved)
        result['forward_graph'] = graph(objective)
        gradient, = torch.autograd.grad(objective, theta, create_graph=True)
        result['gradient_graph'] = graph(gradient)
    result['saved_tensors'] = dict(forward_count=forward_saves, gradient_count=len(saved)-forward_saves,
        forward_bytes_with_repeated_saves=sum(saved[:forward_saves]),
        gradient_bytes_with_repeated_saves=sum(saved[forward_saves:]))
    hvp, = torch.autograd.grad((gradient*direction).sum(), theta)
    torch.testing.assert_close(value, torch.tensor(numeric['price'], dtype=base.dtype), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.tensor(numeric['gradient'], dtype=base.dtype), rtol=0, atol=0)
    torch.testing.assert_close(hvp, torch.tensor(numeric['hvp'], dtype=base.dtype), rtol=0, atol=0)
    return result


def worker(case):
    from runtime_controls import worker_qos
    qos = worker_qos(case['windows_qos'], case.get('affinity'))
    import torch
    sys.path.insert(0, str(ROOT/'python'))
    from quantlibtorch import fd_vanilla
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    times = [0., .37, 1.]
    logs = [math.log(s) for s in (40., 70., 100., 140., 250.)]
    ts, xs = torch.meshgrid(torch.tensor(times, dtype=torch.float64),
                           torch.tensor(logs, dtype=torch.float64)-math.log(100.), indexing='ij')
    coordinates = torch.stack((ts, xs), dim=-1)
    neural = case['surface'] == 'neural'
    base = (torch.linspace(-.3, .4, 17, dtype=torch.float64) if neural else
            (.21+.025*torch.sin(torch.arange(15, dtype=torch.float64))).reshape(3, 5))
    direction = torch.cos(torch.arange(base.numel(), dtype=torch.float64)).reshape_as(base)

    def loss(theta):
        surface = theta
        if neural:
            hidden = torch.tanh(coordinates@theta[:8].reshape(2, 4)+theta[8:12])
            surface = .1+.3*torch.sigmoid(hidden@theta[12:16]+theta[16])
        value = fd_vanilla(surface, times=times, log_spots=logs, kernels='auto', mode='full',
                           time_steps=case['steps'], grid_points=case['nodes'])
        return value, (value-9.).square() if neural else value

    def run():
        theta = base.clone().requires_grad_()
        start = time.perf_counter()
        value, objective = loss(theta)
        forward = time.perf_counter()
        gradient, = torch.autograd.grad(objective, theta, create_graph=True)
        first = time.perf_counter()
        hvp, = torch.autograd.grad((gradient*direction).sum(), theta)
        second = time.perf_counter()
        price, g, h = value.item(), gradient.detach().tolist(), hvp.tolist()
        cleanup = time.perf_counter()
        del value, objective, gradient, hvp, theta
        end = time.perf_counter()
        return dict(price=price, gradient=g, hvp=h, forward_seconds=forward-start,
                    gradient_seconds=first-forward, hvp_seconds=second-first, cleanup_seconds=end-cleanup,
                    total_seconds=(second-start)+(end-cleanup))

    warmup = [run() for _ in range(case['warmup'])]
    samples = [run() for _ in range(case['repeats'])]
    step = 1e-5
    differences = []
    for sign in (1., -1.):
        theta = (base+sign*step*direction).requires_grad_()
        value, objective = loss(theta)
        differences.append(torch.autograd.grad(objective, theta)[0])
    finite_difference = (differences[0]-differences[1])/(2*step)
    hvp = torch.tensor(samples[0]['hvp'], dtype=torch.float64)
    torch.testing.assert_close(hvp, finite_difference, rtol=2e-5, atol=3e-5)
    result = dict(case=case, worker_qos=qos, torch_version=torch.__version__, warmup=warmup, samples=samples,
                median={key: statistics.median(s[key] for s in samples) for key in samples[0] if key.endswith('_seconds')},
                finite_difference_max_error=(hvp-finite_difference).abs().max().item())
    if case.get('profile'):
        result['profile'] = profile_curvature(torch, loss, base, direction)
        for field in ('price', 'gradient', 'hvp'):
            torch.testing.assert_close(torch.tensor(result['profile']['numeric'][field], dtype=torch.float64),
                                       torch.tensor(samples[0][field], dtype=torch.float64), rtol=0, atol=0)
    return result


def flatten(value):
    if isinstance(value, list): return [x for part in value for x in flatten(part)]
    return [value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--grids', default='10x31,30x61')
    parser.add_argument('--surfaces', default='surface,neural')
    parser.add_argument('--strategies', default='pcr,vector', help='pcr,thomas,vector')
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--windows-qos', choices=('system', 'high'), default='high')
    parser.add_argument('--basis-cache', type=int, choices=(0, 1), default=0)
    parser.add_argument('--full-tensor-step', type=int, choices=(0, 1), default=0)
    parser.add_argument('--time-variance-arithmetic', type=int, choices=(0, 1), default=0)
    parser.add_argument('--step-preparation', type=int, choices=(0, 1), default=0)
    parser.add_argument('--assembled-bands', type=int, choices=(0, 1), default=0)
    parser.add_argument('--adjoint-assembled-bands', type=int, choices=(0, 1), default=0)
    parser.add_argument('--diffusion-selected-grads', type=int, choices=(0, 1), default=0)
    parser.add_argument('--profile', action='store_true', help='profile and inspect graphs after all timed samples')
    parser.add_argument('--affinity', type=int, help='pin only each benchmark worker to this logical CPU')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    grids = [tuple(map(int, grid.split('x'))) for grid in args.grids.split(',')]
    surfaces = args.surfaces.split(',')
    strategies = args.strategies.split(',')
    if (not args.output or min(args.warmup, args.repeats) < 1 or set(surfaces)-{'surface', 'neural'}
        or (args.affinity is not None and not 0 <= args.affinity < (os.cpu_count() or 1))
        or set(strategies)-{'pcr', 'thomas', 'vector'}
        or any(len(g) != 2 or g[0] < 1 or g[1] < 5 for g in grids)):
        parser.error('invalid grid/surface/count or missing output')
    features = {'QLT_'+name.upper().replace('-', '_'): str(value) for name, value in zip(FLAGS, LEVELS['phase9'])}
    features.update(QLT_PASSIVE_CONSTANTS='1', QLT_SAMPLE_CACHE='1', QLT_BASIS_CACHE=str(args.basis_cache),
                    QLT_FULL_TENSOR_STEP=str(args.full_tensor_step),
                    QLT_TIME_VARIANCE_ARITHMETIC=str(args.time_variance_arithmetic),
                    QLT_STEP_PREPARATION=str(args.step_preparation),
                    QLT_ASSEMBLED_BANDS=str(args.assembled_bands),
                    QLT_ADJOINT_ASSEMBLED_BANDS=str(args.adjoint_assembled_bands),
                    QLT_DIFFUSION_SELECTED_GRADS=str(args.diffusion_selected_grads))
    payload = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        baseline_environment=features, python_version=sys.version,
        native_source_sha256=native_fingerprint(), results=[], validation=dict(comparisons=0,
            max_price_difference=0., max_gradient_difference=0., max_hvp_difference=0.),
        notes=['Each case has a fresh worker; warmups and samples retain every price, gradient and HVP.',
               'All derivatives use full mode; direction is a fixed cosine vector in input-parameter space.',
               'HVP is checked against central differences of the pricing-loss gradient.',
               'Timed total includes forward, gradient recording, HVP and cleanup; excludes result serialization.'])
    references = {}
    for strategy in strategies:
        flag = int(strategy != 'pcr')
        for steps, nodes in grids:
            for surface in surfaces:
                case = dict(steps=steps, nodes=nodes, surface=surface, thomas_arithmetic=flag,
                            strategy=strategy, thomas_vector_backward=int(strategy == 'vector'),
                            warmup=args.warmup, repeats=args.repeats, windows_qos=args.windows_qos, affinity=args.affinity,
                            profile=args.profile)
                env = dict(os.environ, **features)
                env['QLT_THOMAS_ARITHMETIC'] = str(flag)
                env['QLT_THOMAS_VECTOR_BACKWARD'] = str(int(strategy == 'vector'))
                response = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                    input=json.dumps(case), env=env, capture_output=True, text=True, check=True)
                result = json.loads(response.stdout)
                if native_fingerprint() != payload['native_source_sha256']:
                    raise RuntimeError('Native sources changed during the curvature experiment')
                key = (steps, nodes, surface)
                reference = references.setdefault(key, result['samples'][0])
                for sample in result['warmup']+result['samples']:
                    for field in ('price', 'gradient', 'hvp'):
                        actual, expected = flatten(sample[field]), flatten(reference[field])
                        if len(actual) != len(expected) or any(not math.isfinite(a) or
                            abs(a-e) > 2e-8+2e-8*abs(e) for a, e in zip(actual, expected)):
                            raise RuntimeError(f'Curvature reference mismatch: {case}, {field}')
                        name = 'max_'+field+'_difference'
                        payload['validation'][name] = max(payload['validation'][name], max(abs(a-e) for a, e in zip(actual, expected)))
                    payload['validation']['comparisons'] += 1
                payload['results'].append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
                print(case, result['median'], flush=True)


if __name__ == '__main__':
    main()
