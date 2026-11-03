"""Run isolated Phase 24 workers sequentially; resume only matching evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark_multi_asset import ROOT, fingerprints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'docs/benchmarks/multi_asset_phase24_workloads')
    p.add_argument('--suite', choices=['pilot', 'scaling', 'diagnostics'], default='scaling')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.suite in ('pilot', 'scaling'):
        sizes = [(1, 1), (1, 16), (8, 16), (20, 30), (32, 32)] if args.suite=='scaling' else [(8, 16)]
        for a, c in sizes:
            for execution, device in [('serial', 'cpu'), ('asset_batch', 'cpu'), ('multi', 'cpu'), ('multi', 'cuda')]:
                cases.append(('timing', a, c, 21, 9, 12, execution, device, 1, 'adjoint', 4))
        if args.suite=='scaling':
            for execution, device in [('serial', 'cpu'), ('asset_batch', 'cpu'), ('multi', 'cpu'), ('multi', 'cuda')]:
                cases.append(('timing', 20, 30, 41, 17, 24, execution, device, 1, 'adjoint', 4))
    else:
        for execution, device in [('serial', 'cpu'), ('asset_batch', 'cpu'), ('multi', 'cpu'), ('multi', 'cuda')]:
            cases.append(('memory', 20, 30, 41, 17, 24, execution, device, 1, 'adjoint', 4))
        for device in ['cpu', 'cuda']:
            cases.append(('profile', 8, 16, 21, 9, 12, 'multi', device, 1, 'adjoint', 4))
            cases.append(('timing', 1, 16, 21, 9, 12, 'multi', device, 1, 'full', 4))
            cases.append(('timing', 2, 3, 9, 5, 4, 'multi', device, 2, 'adjoint', 2))
        for stride in [1, 12]:
            cases.append(('memory', 20, 30, 41, 17, 24, 'multi', 'cuda', 1, 'adjoint', stride))
    commands = []
    signature = fingerprints()
    for task, a, c, nx, ny, steps, execution, device, m, mode, stride in cases:
        name = f'{task}_{a}x{c}_{nx}x{ny}_{steps}_{execution}_{device}_{m}_{mode}_k{stride}'
        path = args.output_dir/(name+'.json')
        flags = dict(execution=execution, device=device, assets=a, contracts=c, nx=nx, ny=ny, steps=steps,
                     latents=m, mode=mode, checkpoint_steps=stride, repeats=args.repeats)
        command = [sys.executable, str(ROOT/'scripts/benchmark_multi_asset.py'), task, '--output', str(path)]
        for key, value in flags.items(): command += ['--'+key.replace('_', '-'), str(value)]
        if path.exists():
            old = json.loads(path.read_text())
            if all(old.get(k)==v for k, v in signature.items()) and all(old['configuration'][k]==v for k, v in flags.items()) and 'host_final' in old:
                commands.append(dict(command=command, returncode=0, resumed=True)); continue
            raise RuntimeError(f'Stale or incomplete report: {path}; use a fresh output directory')
        print(name, flush=True)
        with path.with_suffix('.log').open('w', encoding='utf-8') as log:
            run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        commands.append(dict(command=command, returncode=run.returncode, resumed=False))
        (args.output_dir/f'commands_{args.suite}.json').write_text(json.dumps(commands, indent=2)+'\n', encoding='utf-8', newline='\n')
        if run.returncode:
            raise RuntimeError(f'Worker failed; see {path.with_suffix(".log")}')
    (args.output_dir/f'commands_{args.suite}.json').write_text(json.dumps(commands, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
