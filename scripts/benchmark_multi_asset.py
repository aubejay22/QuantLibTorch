"""Isolated end-to-end Phase 24 pricing/VJP, memory, and transfer measurements.

Serial uses optimized one-contract LatentBatch CPU plans and streamed gradients.
asset_batch measures the previous compatible-contract path serially over assets.
multi uses independent asset operators and tensor readout. All include neural
field evaluation, loss, initial-state mapping, and gradient materialization.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))


def fingerprints():
    paths = ['cpp/packed_pde.cpp', 'cpp/multi_asset.inc', 'cpp/parallel_solve.cpp',
             'python/quantlibtorch/batch.py', 'python/quantlibtorch/multi_asset.py',
             'examples/multi_asset_neural_sde.py', 'scripts/prepare_packed.py',
             'scripts/benchmark_multi_asset.py']
    binary = next(p for p in (ROOT/'python/quantlibtorch/_native').glob('*native.*')
                  if p.suffix in ('.dll', '.so', '.dylib'))
    return dict(source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['timing', 'memory', 'profile'])
    p.add_argument('--execution', choices=['serial', 'asset_batch', 'multi'], default='multi')
    p.add_argument('--device', default='cpu')
    p.add_argument('--assets', type=int, default=1)
    p.add_argument('--contracts', type=int, default=1)
    p.add_argument('--nx', type=int, default=21)
    p.add_argument('--ny', type=int, default=9)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--latents', type=int, choices=[1, 2], default=1)
    p.add_argument('--checkpoint-steps', type=int, default=4)
    p.add_argument('--mode', choices=['full', 'adjoint'], default='adjoint')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--affinity', type=int, default=8)
    args = p.parse_args()
    from runtime_controls import worker_qos
    runtime = worker_qos('high', args.affinity)
    import torch
    from quantlibtorch import LatentBatch, MultiAssetBatch
    from multi_asset_neural_sde import NeuralAssets, configuration
    from investigate_fd import process_memory
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cuda = torch.device(args.device).type == 'cuda'
    if args.execution == 'serial' and cuda:
        raise ValueError('serial baseline is CPU')
    def sync():
        if cuda: torch.cuda.synchronize()
    start = time.perf_counter()
    configs, times, spots, targets = configuration(args.assets, args.contracts, args.nx, args.ny, args.steps, args.latents)
    model = NeuralAssets([c['axes'] for c in configs], times, spots, args.device)
    target = torch.tensor(targets, dtype=torch.float64, device=args.device)
    work = []
    if args.execution == 'multi':
        plan = MultiAssetBatch(configs, device=args.device, checkpoint_steps=args.checkpoint_steps)
        work.append((plan, None, None))
    else:
        for i, config in enumerate(configs):
            if args.execution == 'asset_batch':
                work.append((LatentBatch(**config, device=args.device), i, None))
            else:
                for j, strike in enumerate(config['strikes']):
                    work.append((LatentBatch(**{**config, 'strikes': [strike]}), i, j))
    sync()
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), configuration=vars(args),
                  runtime=runtime, torch=torch.__version__, python=sys.version, **fingerprints(),
                  device_name=torch.cuda.get_device_name() if cuda else 'CPU', setup_seconds=time.perf_counter()-start,
                  baseline_memory=process_memory(), nodes=args.nx*args.ny**args.latents,
                  numerical_note='Fixed small-grid workload; throughput is not a convergence certificate.')
    if cuda: torch.cuda.reset_peak_memory_stats()
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str)+'\n', encoding='utf-8', newline='\n')
    save()

    def run(gradient=True, materialize=True):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter(); forward = backward = 0.
        rows, losses = [], []
        with torch.set_grad_enabled(gradient):
            for plan, i, j in work:
                begin = time.perf_counter()
                fields = model.fields(None if i is None else [i])
                if i is not None: fields = tuple(v[0] for v in fields)
                prices = plan.price(*fields, mode=args.mode)
                truth = target if i is None else target[i] if j is None else target[i, j:j+1]
                loss = (prices-truth).square().sum()/(args.assets*args.contracts)
                sync(); middle = time.perf_counter()
                forward += middle-begin
                if gradient: loss.backward()
                sync(); backward += time.perf_counter()-middle
                rows.append(prices.detach().reshape(-1)); losses.append(loss.detach())
                del fields, prices, loss
        data = {}
        if materialize:
            data['prices'] = torch.cat(rows).cpu().tolist()
            data['loss'] = float(torch.stack(losses).sum())
            if gradient: data['gradients'] = {k: v.grad.detach().cpu().tolist() for k, v in model.named_parameters()}
        del rows, losses
        sync(); cleanup = time.perf_counter()
        model.zero_grad(set_to_none=True); gc.collect(); sync()
        total = time.perf_counter()-start
        return dict(seconds=total, forward_seconds=forward, backward_seconds=backward,
                    cleanup_seconds=time.perf_counter()-cleanup,
                    contracts_per_second=args.assets*args.contracts/total, **data)

    if args.task == 'timing':
        result['warmups'] = [run() for _ in range(args.warmup)]
        result['samples'], result['inference_samples'] = [], []
        for _ in range(args.repeats):
            result['samples'].append(run()); save()
        run(False)
        for _ in range(args.repeats):
            result['inference_samples'].append(run(False)); save()
        result['median'] = {k: statistics.median(r[k] for r in result['samples']) for k in
                            ('seconds', 'forward_seconds', 'backward_seconds', 'cleanup_seconds', 'contracts_per_second')}
        result['inference_median_seconds'] = statistics.median(r['seconds'] for r in result['inference_samples'])
    elif args.task == 'memory':
        result['cycles'] = []
        for _ in range(args.repeats):
            row = run(); row['host_after_cleanup'] = process_memory()
            if cuda:
                row['cuda'] = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                                   peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            result['cycles'].append(row); save()
    else:
        run()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda: activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile:
            result['evaluation'] = run(materialize=False)
        result['events'] = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                                 device_us=e.device_time_total) for e in profile.key_averages()]
        trace = args.output.with_suffix('.trace.json')
        profile.export_chrome_trace(str(trace))
        raw = json.loads(trace.read_text())
        transfers = [e for e in raw.get('traceEvents', []) if 'memcpy' in e.get('cat', '').lower()]
        result['transfers'] = [dict(name=e.get('name'), category=e.get('cat'), duration_us=e.get('dur'), args=e.get('args')) for e in transfers]
        result['trace_file'] = str(trace)
        result['cuda_trace_available'] = any(e.get('cat')=='kernel' for e in raw.get('traceEvents', []))
    result['host_final'] = process_memory()
    result['host_peak_growth_bytes'] = result['host_final']['peak_rss_bytes']-result['baseline_memory']['peak_rss_bytes']
    if cuda:
        result['cuda_final'] = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                                   peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    save()
    print(json.dumps(dict(output=str(args.output), median=result.get('median'), host_peak_growth=result['host_peak_growth_bytes'])))


if __name__ == '__main__':
    main()
