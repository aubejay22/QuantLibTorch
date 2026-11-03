"""Isolated CPU/CUDA solve+VJP measurements, not option-pricing throughput.

Validation uses independent CPU dense solutions and every band/RHS VJP in its
own worker. Timings retain all warmups/samples and synchronize CUDA. Resident
inputs and a host-to-device/CPU-result-and-gradient roundtrip are distinguished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from investigate_fd import ROOT, native_fingerprint, process_memory


def worker(case):
    from runtime_controls import worker_qos
    qos = worker_qos(case.get('windows_qos', 'system'))
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from quantlibtorch import tridiagonal_solve_pcr

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device, transfer = case['device'], case['transfer']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    batch, n, nrhs = case['batch'], case['nodes'], case['rhs']
    generator = torch.Generator().manual_seed(277)
    band_batch = () if case.get('shared_bands', False) else (batch,)
    cpu = tuple(t.requires_grad_() for t in (
        .1*torch.randn(*band_batch, n-1, dtype=torch.float64, generator=generator),
        2+torch.rand(*band_batch, n, dtype=torch.float64, generator=generator),
        -.1*torch.randn(*band_batch, n-1, dtype=torch.float64, generator=generator),
        torch.randn(batch, n, nrhs, dtype=torch.float64, generator=generator)))
    weights = torch.linspace(-.7, 1.1, batch*n*nrhs, dtype=torch.float64).reshape(batch, n, nrhs)
    if case.get('unbatched', False):
        cpu = (*cpu[:3], cpu[3].squeeze(0)) if case.get('shared_bands', False) else tuple(t.squeeze(0) for t in cpu)
        weights = weights.squeeze(0)
    resident = tuple(t.detach().to(device).requires_grad_() for t in cpu)
    device_weights = weights.to(device)

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    def solve(args):
        if case.get('solver', 'pcr') == 'thomas':
            return torch.ops.quantlibtorch._thomas_arithmetic(*args, case['checked'], True)
        return tridiagonal_solve_pcr(*args, check_pivots=case['checked'])

    result = {'case': case, 'worker_qos': qos, 'input_bytes': sum(t.numel()*t.element_size() for t in cpu)}
    if device == 'cuda':
        result['gpu'] = torch.cuda.get_device_name()
    if case['task'] == 'validation':
        lo, di, up, rhs = cpu
        reference = torch.linalg.solve(torch.diag_embed(lo, offset=-1)+torch.diag_embed(di)
                                       +torch.diag_embed(up, offset=1), rhs)
        expected = torch.autograd.grad(reference, cpu, weights)
        actual = solve(resident)
        gradients = torch.autograd.grad(actual, resident, device_weights)
        torch.testing.assert_close(actual.cpu(), reference, rtol=2e-12, atol=2e-13)
        for gradient, ref in zip(gradients, expected):
            torch.testing.assert_close(gradient.cpu(), ref, rtol=2e-11, atol=2e-12)
        result['validation'] = dict(max_value_difference=(actual.cpu()-reference).abs().max().item(),
            max_vjp_difference=max((g.cpu()-ref).abs().max().item() for g, ref in zip(gradients, expected)))
        return result

    if case['task'] == 'memory':
        gc.collect()
        sync()
        result['before'] = process_memory()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            result['cuda_allocated_before_bytes'] = torch.cuda.memory_allocated()
        args = tuple(t.to(device) for t in cpu) if transfer == 'roundtrip' else resident
        value = solve(args)
        if transfer == 'roundtrip':
            value = value.cpu()
        sync()
        result['after_forward'] = process_memory()
        gradients = torch.autograd.grad(value, cpu if transfer == 'roundtrip' else resident,
                                        weights if transfer == 'roundtrip' else device_weights)
        sync()
        result['after_backward'] = process_memory()
        del value, gradients, args
        gc.collect()
        sync()
        result['after_cleanup'] = process_memory()
        result['peak_rss_growth_bytes'] = max(0, result['after_cleanup']['peak_rss_bytes']
                                               - result['before']['peak_rss_bytes'])
        if device == 'cuda':
            result['cuda_peak_allocated_growth_bytes'] = (torch.cuda.max_memory_allocated()
                                                        - result['cuda_allocated_before_bytes'])
        return result

    def run():
        sync()
        start = time.perf_counter()
        args = tuple(t.to(device) for t in cpu) if transfer == 'roundtrip' else resident
        value = solve(args)
        if transfer == 'roundtrip':
            value = value.cpu()
        sync()
        forward_end = time.perf_counter()
        gradients = torch.autograd.grad(value, cpu if transfer == 'roundtrip' else resident,
                                        weights if transfer == 'roundtrip' else device_weights)
        sync()
        backward_end = time.perf_counter()
        del value, gradients, args
        sync()
        end = time.perf_counter()
        return dict(forward_seconds=forward_end-start, backward_seconds=backward_end-forward_end,
                    cleanup_seconds=end-backward_end, total_seconds=end-start)

    result['warmup'] = [run() for _ in range(case['warmup'])]
    result['samples'] = [run() for _ in range(case['repeats'])]
    result['median'] = {key: statistics.median(s[key] for s in result['samples'])
                        for key in result['samples'][0]}
    result['systems_per_second'] = batch/result['median']['total_seconds']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--shapes', default='1x61x1,32x61x1,256x257x4', help='batch x nodes x RHS')
    parser.add_argument('--devices', default='cpu,cuda')
    parser.add_argument('--solvers', default='pcr', help='pcr or experimental full-AD thomas recurrence')
    parser.add_argument('--checks', default='checked,unchecked')
    parser.add_argument('--transfers', default='resident,roundtrip')
    parser.add_argument('--tasks', default='validation,timing')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--pcr-fused', type=int, choices=(0, 1), default=0)
    parser.add_argument('--pcr-native-stage', type=int, choices=(0, 1), default=1)
    parser.add_argument('--unbatched', action='store_true', help='omit the batch axis; requires batch=1')
    parser.add_argument('--shared-bands', action='store_true', help='one shared matrix broadcast across RHS batches')
    parser.add_argument('--windows-qos', choices=('system', 'high'), default='system')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    shapes = [tuple(map(int, shape.split('x'))) for shape in args.shapes.split(',')]
    devices, checks = args.devices.split(','), args.checks.split(',')
    solvers = args.solvers.split(',')
    transfers, tasks = args.transfers.split(','), args.tasks.split(',')
    if (any(len(s) != 3 or s[0] < 1 or s[1] < 2 or s[2] < 1 for s in shapes)
        or set(devices)-{'cpu', 'cuda'} or set(checks)-{'checked', 'unchecked'} or set(solvers)-{'pcr', 'thomas'}
        or set(transfers)-{'resident', 'roundtrip'} or set(tasks)-{'validation', 'timing', 'memory'}
        or min(args.warmup, args.repeats) < 1 or args.output is None
        or (args.unbatched and any(s[0] != 1 for s in shapes))):
        parser.error('invalid shapes/devices/checks/transfers/tasks/counts or missing output')
    import torch
    import os
    os.environ['QLT_PCR_FUSED'] = str(args.pcr_fused)
    os.environ['QLT_PCR_NATIVE_STAGE'] = str(args.pcr_native_stage)
    payload = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        native_source_sha256=native_fingerprint(), pcr_roll_environment=os.environ.get('QLT_PCR_ROLL'),
        pcr_fused_environment=os.environ.get('QLT_PCR_FUSED'),
        pcr_native_stage_environment=os.environ.get('QLT_PCR_NATIVE_STAGE'),
        windows_qos=args.windows_qos,
        environment=dict(python=platform.python_version(), torch=torch.__version__, os=platform.platform(),
                         cpu=platform.processor(), torch_threads=1, torch_interop_threads=1, dtype='float64'),
        notes=['Every case/task gets a fresh subprocess; imports/startup excluded.',
               'CUDA wall times synchronize after forward, backward and cleanup.',
               'Checked mode validates finite inputs, elimination pivots and the output; tensor checks can synchronize devices.',
               'Thomas uses native CPU recurrence arithmetic and its full chain rule; unsupported cases use tensor recurrence.',
               'Resident timings exclude transfers. Roundtrip includes input H2D, result D2H and all gradient D2H copies.',
               'Roundtrip uses ordinary unpinned host tensors. It does not represent an overlapped transfer pipeline.',
               'Each validation worker compares against independent CPU dense solves and all band/RHS VJPs.',
               'Memory workers measure a cold first solve; process peak RSS includes allocator retention and native allocations.',
               'CUDA peak allocated growth excludes existing resident inputs and uses the PyTorch allocator counter.',
               'This is a numerical solve benchmark; no batched financial engine or GPU pricing is claimed.'], results=[])
    for (batch, n, nrhs), solver, device, checked, transfer, task in itertools.product(shapes, solvers, devices, checks, transfers, tasks):
        if transfer == 'roundtrip' and (device == 'cpu' or task == 'validation'):
            continue
        case = dict(batch=batch, nodes=n, rhs=nrhs, solver=solver, device=device, checked=checked == 'checked',
                    transfer=transfer, task=task, warmup=args.warmup, repeats=args.repeats,
                    windows_qos=args.windows_qos, unbatched=args.unbatched, shared_bands=args.shared_bands)
        response = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                  input=json.dumps(case), capture_output=True, text=True, check=True)
        result = json.loads(response.stdout)
        payload['results'].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
        print(case, result.get('median', result.get('validation', result.get('peak_rss_growth_bytes'))), flush=True)


if __name__ == '__main__':
    main()
