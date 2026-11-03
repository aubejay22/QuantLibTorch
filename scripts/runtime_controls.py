"""Explicit, worker-local runtime controls for reproducible benchmarks only."""
import ctypes
import os
import sys


def windows_process_qos(disable=False):
    if sys.platform != 'win32':
        if disable:
            raise RuntimeError('Process power-throttling control requires Windows')
        return None
    from ctypes import wintypes
    # https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
    # SDK 10.0.26100.0: ProcessPowerThrottling=4, version=1, EXECUTION_SPEED=0x1.
    class Policy(ctypes.Structure):
        _fields_ = [(name, wintypes.ULONG) for name in ('version', 'control_mask', 'state_mask')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    signature = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.GetProcessInformation.argtypes = signature
    kernel.GetProcessInformation.restype = wintypes.BOOL
    kernel.SetProcessInformation.argtypes = signature
    kernel.SetProcessInformation.restype = wintypes.BOOL
    policy = Policy(1, 0, 0)
    process = kernel.GetCurrentProcess()
    if not kernel.GetProcessInformation(process, 4, ctypes.byref(policy), ctypes.sizeof(policy)):
        raise ctypes.WinError(ctypes.get_last_error())
    if disable:
        policy.control_mask |= 1
        policy.state_mask &= ~1
        if not kernel.SetProcessInformation(process, 4, ctypes.byref(policy), ctypes.sizeof(policy)):
            raise ctypes.WinError(ctypes.get_last_error())
        return windows_process_qos()
    return dict((name, getattr(policy, name)) for name, _ in Policy._fields_)


def set_affinity(processor):
    if processor is None:
        return
    if not 0 <= processor < (os.cpu_count() or 1):
        raise ValueError('affinity processor is outside the logical CPU range')
    if sys.platform == 'win32':
        from ctypes import wintypes
        if processor >= ctypes.sizeof(ctypes.c_size_t)*8:
            raise ValueError('affinity processor exceeds the supported Windows mask width')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel.SetProcessAffinityMask.restype = wintypes.BOOL
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << processor):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {processor})
    else:
        raise RuntimeError('Affinity unsupported on this platform')


def process_affinity():
    if sys.platform == 'win32':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetCurrentProcessorNumber.restype = wintypes.DWORD
        kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                 ctypes.POINTER(ctypes.c_size_t)]
        kernel.GetProcessAffinityMask.restype = wintypes.BOOL
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(mask=process_mask.value, system_mask=system_mask.value, processor=kernel.GetCurrentProcessorNumber())
    if hasattr(os, 'sched_getaffinity'):
        return dict(cpus=sorted(os.sched_getaffinity(0)))
    return None


def worker_qos(choice, affinity=None):
    if choice not in ('system', 'high'):
        raise ValueError('unknown worker QoS policy')
    affinity_before = process_affinity() if affinity is not None else None
    set_affinity(affinity)
    affinity_after = process_affinity() if affinity is not None else None
    if affinity is not None:
        pinned = (affinity_after.get('mask') == 1 << affinity if 'mask' in affinity_after
                  else affinity_after.get('cpus') == [affinity])
        if not pinned:
            raise RuntimeError('worker affinity does not match the requested processor')
    before = windows_process_qos()
    if choice == 'high':
        windows_process_qos(disable=True)
    result = dict(requested=choice, before=before, after=windows_process_qos())
    if affinity is not None:
        result['affinity'] = dict(requested=affinity, before=affinity_before, after=affinity_after)
    return result
