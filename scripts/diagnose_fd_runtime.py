"""Record repeated-price runtime regimes, memory, and OS CPU observations.

Diagnostic observations are outside each timed price/VJP/cleanup. They still
perturb the interval between calls, so these are not headline pricing timings.
Optional affinity affects only this diagnostic process and its child threads.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from investigate_fd import ROOT, native_fingerprint, process_memory
from runtime_controls import windows_process_qos, set_affinity


def windows_cpu():
    if sys.platform != 'win32':
        return None
    from ctypes import wintypes
    class Power(ctypes.Structure):
        _fields_ = [(name, wintypes.ULONG) for name in
                    ('processor', 'max_mhz', 'current_mhz', 'limit_mhz', 'max_idle', 'current_idle')]
    kernel = ctypes.WinDLL('kernel32')
    kernel.GetCurrentProcessorNumber.restype = wintypes.DWORD
    processor = kernel.GetCurrentProcessorNumber()
    data = (Power * os.cpu_count())()
    library = ctypes.WinDLL('powrprof')
    library.CallNtPowerInformation.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.ULONG,
                                             ctypes.c_void_p, wintypes.ULONG]
    library.CallNtPowerInformation.restype = wintypes.ULONG
    status = library.CallNtPowerInformation(11, None, 0, ctypes.byref(data), ctypes.sizeof(data))
    return dict(processor=processor, status=status, process_qos=windows_process_qos(),
                power=None if status else [dict((name, getattr(p, name)) for name, _ in Power._fields_)
                                           for p in data])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calls', type=int, default=60)
    parser.add_argument('--affinity', type=int)
    parser.add_argument('--pause-ms', type=float, default=0)
    parser.add_argument('--disable-throttling', action='store_true',
                        help='opt this diagnostic process out of Windows execution-speed throttling')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.calls < 2 or args.pause_ms < 0 or (args.affinity is not None and not 0 <= args.affinity < os.cpu_count()):
        parser.error('invalid calls, pause, or processor')
    set_affinity(args.affinity)
    previous_qos = windows_process_qos()
    if args.disable_throttling:
        windows_process_qos(disable=True)
    import torch
    sys.path.insert(0, str(ROOT / 'python'))
    from quantlibtorch import fd_vanilla
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = (.21+.025*torch.sin(torch.arange(15, dtype=torch.float64))).reshape(3, 5)
    kwargs = dict(times=[0., .37, 1.], log_spots=[math.log(s) for s in (40., 70., 100., 140., 250.)],
                  time_steps=30, grid_points=61, mode='adjoint', kernels='auto')
    payload = dict(command=sys.argv, measured_at_utc=datetime.now(timezone.utc).isoformat(),
                   case=dict(steps=30, nodes=61, surface='surface', mode='adjoint', kernels='auto', contract='european'),
                   native_source_sha256=native_fingerprint(), torch=torch.__version__,
                   platform=platform.platform(), logical_processors=os.cpu_count(),
                   process_qos_before_control=previous_qos,
                   process_qos_after_control=windows_process_qos(),
                   performance_environment={k: v for k, v in os.environ.items() if k.startswith('QLT_')},
                   notes=['CPU observations and process-memory reads are outside timed work.',
                          'OS-reported MHz is not a calibrated measurement of instantaneous core frequency.',
                          'Processor numbers sample only the calling thread at the observation point.',
                          'Affinity, pauses and power-throttling opt-out affect only this diagnostic process.',
                          'No machine power plan or production pricing policy is changed.'], samples=[])
    for index in range(args.calls):
        observation = windows_cpu()
        sigma = base.clone().requires_grad_()
        cpu, start = time.process_time(), time.perf_counter()
        price = fd_vanilla(sigma, **kwargs)
        gradient, = torch.autograd.grad(price, sigma)
        numeric = (price.item(), gradient.tolist())
        del price, gradient, sigma
        elapsed, used_cpu = time.perf_counter()-start, time.process_time()-cpu
        if index:
            torch.testing.assert_close(torch.tensor(numeric[0], dtype=torch.float64),
                                       torch.tensor(payload['samples'][0]['price'], dtype=torch.float64), rtol=0, atol=0)
            torch.testing.assert_close(torch.tensor(numeric[1], dtype=torch.float64),
                                       torch.tensor(payload['samples'][0]['gradient'], dtype=torch.float64), rtol=0, atol=0)
        payload['samples'].append(dict(index=index, total_seconds=elapsed, total_cpu_seconds=used_cpu,
                                       price=numeric[0], gradient=numeric[1],
                                       cpu_before=observation, cpu_after=windows_cpu(), memory=process_memory()))
        if args.pause_ms:
            time.sleep(args.pause_ms/1000.)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print([(s['index'], round(s['total_seconds']*1000, 2)) for s in payload['samples']])


if __name__ == '__main__':
    main()
