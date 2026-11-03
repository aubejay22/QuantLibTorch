"""Small complete neural price/VJP benchmark for batched financial events.

Run after tests, without other CPU/GPU workloads. Setup includes event-map
construction. Samples include fields, prices, loss, backward, materialization,
gradient clearing and garbage collection. This is not an accuracy study.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks/financial_events_phase25.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--assets", type=int, default=4)
    parser.add_argument("--contracts", type=int, default=6)
    parser.add_argument("--affinity", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.assets < 1 or args.contracts < 1:
        parser.error("repeats, assets and contracts must be positive")
    from runtime_controls import worker_qos
    runtime = worker_qos("high", args.affinity)
    import torch
    from quantlibtorch import MultiAssetBatch
    from multi_asset_neural_sde import NeuralAssets, configuration
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    paths = ["cpp/packed_pde.cpp", "cpp/packed_events.inc", "cpp/multi_asset.inc",
             "python/quantlibtorch/batch.py", "python/quantlibtorch/multi_asset.py",
             "scripts/prepare_packed.py", "examples/multi_asset_neural_sde.py",
             "scripts/benchmark_financial_events.py"]
    binary = next(p for p in (ROOT / "python/quantlibtorch/_native").glob("*native.*")
                  if p.suffix in (".dll", ".so", ".dylib"))
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                  python=sys.version, runtime=runtime, assets=args.assets, contracts=args.contracts,
                  grid=[21, 9], time_steps=12, checkpoint_steps=4, warmups=1, repeats=args.repeats,
                  cpu_threads=1, samples_include="neural fields, price, loss, backward, logging, cleanup",
                  note="Small fixed-grid integration workload, not a convergence or broad GPU speedup claim.",
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), workloads=[])
    for kind in ("american", "bermudan"):
        references = None
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            def sync():
                if device == "cuda":
                    torch.cuda.synchronize()
            for mode in ("full", "adjoint"):
                begin = time.perf_counter()
                configs, times, spots, _ = configuration(args.assets, args.contracts)
                for i, config in enumerate(configs):
                    config.update(exercise=kind, option_types=["put"] * args.contracts,
                                  cash_dividends=[(date(2025, 5, 1), .5+.1*i),
                                                  (date(2025, 9, 1), .8+.1*i)])
                    if kind == "bermudan":
                        config["exercise_dates"] = [date(2025, 5, 1), date(2025, 9, 1), date(2026, 1, 1)]
                plan = MultiAssetBatch(configs, device=device, checkpoint_steps=4)
                model = NeuralAssets([c['axes'] for c in configs], times, spots, device)
                # Fixed nonzero target only defines a reproducible scalar loss.
                target = torch.full((args.assets, args.contracts), 8., dtype=torch.float64, device=device)
                sync()
                record = dict(exercise=kind, device=device, mode=mode, setup_seconds=time.perf_counter()-begin,
                              device_name=torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                              samples=[])
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.repeats+1):
                    sync()
                    begin = time.perf_counter()
                    prices = plan.price(*model.fields(), mode=mode)
                    loss = (prices-target).square().mean()
                    loss.backward()
                    sync()
                    values = prices.detach().cpu()
                    gradients = torch.cat([p.grad.detach().cpu().reshape(-1) for p in model.parameters()])
                    materialized = dict(prices=values.tolist(), gradients=gradients.tolist(), loss=loss.item())
                    model.zero_grad(set_to_none=True)
                    del prices, loss
                    gc.collect()
                    sync()
                    elapsed = time.perf_counter()-begin
                    if references is None:
                        references = values.clone(), gradients.clone()
                    price_error = (values-references[0]).abs().max().item()
                    gradient_error = (gradients-references[1]).abs().max().item()
                    torch.testing.assert_close(values, references[0], rtol=2e-10, atol=2e-10)
                    torch.testing.assert_close(gradients, references[1], rtol=2e-8, atol=2e-8)
                    if repeat:
                        record['samples'].append(dict(seconds=elapsed, max_price_error=price_error,
                                                      max_gradient_error=gradient_error, **materialized))
                record['median_seconds'] = statistics.median(s['seconds'] for s in record['samples'])
                if device == 'cuda':
                    record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                report['workloads'].append(record)
                print(f"{kind} {device} {mode}: {record['median_seconds']:.6f}s", flush=True)
                del plan, model, target
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    report['max_price_error'] = max(s['max_price_error'] for w in report['workloads'] for s in w['samples'])
    report['max_gradient_error'] = max(s['max_gradient_error'] for w in report['workloads'] for s in w['samples'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
