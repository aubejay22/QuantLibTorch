"""Run Phase 23 fits, frozen-model refinements and independent derivative checks."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/benchmarks'


def main():
    jobs = []
    for mode in ['adjoint', 'full']:
        jobs.append((f'fit_{mode}', 'benchmark_packed_calibration.py', ['fit', '--market',
            str(OUT/'calibration_phase22_market.json'), '--mesh', 'space', '--maturities', 'multiple',
            '--contracts', '9', '--execution', 'batched', '--mode', mode, '--iterations', '15']))
    jobs.append(('neural_convergence', 'benchmark_packed_calibration.py', ['convergence',
        '--maturities', 'multiple', '--contracts', '9', '--parameters', str(OUT/'calibration_phase22_fit.json')]))
    jobs.append(('accuracy', 'diagnose_packed_accuracy.py', []))
    for name, mesh, mode, device in [('gradients_cpu', 'training', 'adjoint', 'cpu'),
                                    ('gradients_full', 'small', 'full', 'cpu'),
                                    ('gradients_cuda', 'training', 'adjoint', 'cuda')]:
        jobs.append((name, 'benchmark_packed_calibration.py', ['gradients', '--mesh', mesh,
            '--contracts', '3', '--mode', mode, '--device', device,
            '--parameters', str(OUT/'calibration_phase22_fit.json')]))
    commands = []
    for name, script, args in jobs:
        output = OUT/f'packed_phase23_{name}.json'
        command = [sys.executable, str(ROOT/'scripts'/script), *args, '--output', str(output)]
        start = time.time()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        row = dict(name=name, command=command, seconds=time.time()-start, returncode=completed.returncode,
                   stdout=completed.stdout, stderr=completed.stderr)
        commands.append(row)
        (OUT/'packed_phase23_validation_commands.json').write_text(json.dumps(commands, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps({k: row[k] for k in ['name', 'seconds', 'returncode']}), flush=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr)


if __name__ == '__main__':
    main()
