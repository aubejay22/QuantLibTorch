"""Run selected Phase 23 workload workers sequentially, with resumable evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'docs/benchmarks/packed_phase23_workloads')
    parser.add_argument('--devices', default='cpu')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    jobs = []
    if 'cpu' in args.devices.split(','):
        for mode in ['full', 'adjoint']:
            for count in [1, 3, 9]:
                for execution in ['serial', 'batched']:
                    jobs.append(('timing', 'small', count, 'common', mode, execution, 'cpu', 1))
        for mesh, count, maturities in [('training', 1, 'common'), ('training', 3, 'common'),
                ('training', 9, 'common'), ('refined', 3, 'common'), ('medium', 27, 'multiple')]:
            for execution in ['serial', 'batched']:
                jobs.append(('timing', mesh, count, maturities, 'adjoint', execution, 'cpu', 1))
        for mode in ['full', 'adjoint']:
            for execution in ['serial', 'batched']:
                jobs.append(('memory', 'small', 9, 'common', mode, execution, 'cpu', 1))
                jobs.append(('profile', 'diagnostic', 3, 'common', mode, execution, 'cpu', 1))
                jobs.append(('timing', 'three_dimensional', 3, 'common', mode, execution, 'cpu', 2))
        for execution in ['serial', 'batched']:
            jobs.append(('memory', 'training', 9, 'common', 'adjoint', execution, 'cpu', 1))
        # Larger device comparison after optimizing CPU. No extrapolation of
        # original serial runtime or financially converged full AD is made.
        jobs.append(('timing', 'fine', 9, 'common', 'adjoint', 'batched', 'cpu', 1))
        jobs.append(('memory', 'fine', 9, 'common', 'adjoint', 'batched', 'cpu', 1))
    if 'cuda' in args.devices.split(','):
        for mesh, count, maturities in [('small', 3, 'common'), ('training', 3, 'common'),
                ('training', 9, 'common'), ('refined', 3, 'common'), ('medium', 27, 'multiple')]:
            jobs.append(('timing', mesh, count, maturities, 'adjoint', 'batched', 'cuda', 1))
        jobs.append(('timing', 'small', 3, 'common', 'full', 'batched', 'cuda', 1))
        jobs.append(('memory', 'training', 9, 'common', 'adjoint', 'batched', 'cuda', 1))
        jobs.append(('timing', 'fine', 9, 'common', 'adjoint', 'batched', 'cuda', 1))
        jobs.append(('memory', 'fine', 9, 'common', 'adjoint', 'batched', 'cuda', 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands_path = args.output_dir/'commands.json'
    commands = json.loads(commands_path.read_text()) if commands_path.exists() else []
    for task, mesh, count, maturities, mode, execution, device, dimensions in jobs:
        name = f'{task}_{mesh}_{count}_{maturities}_{mode}_{execution}_{device}_{dimensions}'
        output = args.output_dir/(name+'.json')
        if output.exists():
            record = json.loads(output.read_text())
            done = ('median' in record if task == 'timing' else 'peak_growth_bytes' in record
                    if task == 'memory' else 'events' in record)
            if done:
                for source, expected in record['source_sha256'].items():
                    if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != expected:
                        raise RuntimeError(f'{output}: source changed ({source}); use a new output directory')
                native = next(p for p in (ROOT/'python/quantlibtorch/_native').glob('*native.*')
                              if p.suffix in ('.dll', '.so', '.dylib'))
                if hashlib.sha256(native.read_bytes()).hexdigest() != record['binary_sha256']:
                    raise RuntimeError(f'{output}: native binary changed; use a new output directory')
                continue
        command = [sys.executable, str(ROOT/'scripts/benchmark_packed_calibration.py'), task,
            '--mesh', mesh, '--contracts', str(count), '--maturities', maturities, '--mode', mode,
            '--execution', execution, '--device', device, '--latent-dimensions', str(dimensions),
            '--repeats', str(args.repeats), '--output', str(output)]
        start = time.time()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        row = dict(name=name, command=command, seconds=time.time()-start, returncode=completed.returncode,
                   stdout=completed.stdout, stderr=completed.stderr)
        commands.append(row)
        commands_path.write_text(json.dumps(commands, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps({k: row[k] for k in ['name', 'seconds', 'returncode']}), flush=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr)


if __name__ == '__main__':
    main()
