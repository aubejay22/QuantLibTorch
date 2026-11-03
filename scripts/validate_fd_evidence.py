"""Cross-check stored pricing runs, including every warmup/timed price and VJP.

Record source/binary/test/log fingerprints alongside the comparisons. Reports
named with --require-current must match the current native-source fingerprint;
historical reports are labelled explicitly rather than silently relabelled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from investigate_fd import ROOT, native_fingerprint, validate_results


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_tail(path):
    raw = path.read_bytes()
    encoding = 'utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    return raw.decode(encoding, errors='replace').splitlines()[-6:]


def validate_curvature(payload):
    def flatten(value):
        return [x for part in value for x in flatten(part)] if isinstance(value, list) else [value]
    references, observed = {}, {}
    result = dict(comparisons=0, max_price_difference=0., max_gradient_difference=0., max_hvp_difference=0.)
    for record in payload['results']:
        case = record['case']
        key = (case['steps'], case['nodes'], case['surface'])
        observed.setdefault(key, set()).add(case['thomas_arithmetic'])
        if len(record['warmup']) != case['warmup'] or len(record['samples']) != case['repeats']:
            raise RuntimeError('Incomplete curvature samples')
        if not math.isfinite(record['finite_difference_max_error']):
            raise RuntimeError('Invalid curvature finite-difference diagnostic')
        for sample in record['warmup']+record['samples']:
            reference = references.setdefault(key, sample)
            for field in ('price', 'gradient', 'hvp'):
                actual, expected = flatten(sample[field]), flatten(reference[field])
                if len(actual) != len(expected) or any(not math.isfinite(a) or
                    abs(a-e) > 2e-8+2e-8*abs(e) for a, e in zip(actual, expected)):
                    raise RuntimeError(f'Curvature reference mismatch: {key}, {field}')
                name = 'max_'+field+'_difference'
                result[name] = max(result[name], max(abs(a-e) for a, e in zip(actual, expected)))
            result['comparisons'] += 1
    if not observed or any(flags != {0, 1} for flags in observed.values()):
        raise RuntimeError('Missing paired curvature configurations')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--require-current', type=Path, action='append', default=[])
    parser.add_argument('--log', type=Path, action='append', default=[])
    parser.add_argument('--overlay-check', type=Path)
    parser.add_argument('--tensor-report', type=Path, action='append', default=[])
    parser.add_argument('--historical-tensor-report', type=Path, action='append', default=[])
    parser.add_argument('--runtime-report', type=Path, action='append', default=[])
    parser.add_argument('--curvature-report', type=Path, action='append', default=[])
    parser.add_argument('--historical-curvature-report', type=Path, action='append', default=[])
    parser.add_argument('--source-archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    current = native_fingerprint()
    required = {p.resolve() for p in args.require_current}
    if required-{p.resolve() for p in args.reports}:
        parser.error('--require-current paths must also appear among reports')
    combined, records = {'results': [], 'stock': []}, {}
    for path in args.reports:
        payload = json.loads(path.read_text(encoding='utf-8'))
        matched = payload['native_source_sha256'] == current
        if path.resolve() in required and not matched:
            raise RuntimeError(f'Native source mismatch: {path}')
        records[path.as_posix()] = dict(sha256=digest(path), matches_current_native_source=matched,
                                       validation=validate_results(payload))
        combined['results'].extend(payload['results'])
        combined['stock'].extend(payload['stock'])
        for result in payload['results']:
            for sample in result.get('warmup', [])+result.get('samples', []):
                combined['results'].append(dict(case=result['case'], **sample))
    library_paths = [p for p in (ROOT/'python/quantlibtorch/_native').glob('quantlibtorch_native.*')
                     if p.suffix in {'.dll', '.so', '.dylib'}]
    logs = {p.as_posix(): dict(sha256=digest(p), tail=log_tail(p))
            for p in args.log}
    overlay = None
    if args.overlay_check:
        original = ROOT/'build/native/quantlib-overlay'
        paths = [p.relative_to(original) for p in original.rglob('*') if p.is_file()]
        compared = [p.relative_to(args.overlay_check) for p in args.overlay_check.rglob('*') if p.is_file()]
        if set(paths) != set(compared) or any(digest(original/p) != digest(args.overlay_check/p) for p in paths):
            raise RuntimeError('Freshly generated QuantLib overlay differs from the build overlay')
        overlay = dict(directory=args.overlay_check.as_posix(), identical_files=len(paths))
    tensors = {}
    for path in args.tensor_report+args.historical_tensor_report:
        payload = json.loads(path.read_text(encoding='utf-8'))
        matched = payload['native_source_sha256'] == current
        if path in args.tensor_report and not matched:
            raise RuntimeError(f'Tensor benchmark native-source mismatch: {path}')
        checks = [r['validation'] for r in payload['results'] if r['case']['task'] == 'validation']
        if not checks or any(not math.isfinite(x) or x > 1e-11 for check in checks for x in check.values()):
            raise RuntimeError(f'Invalid tensor benchmark checks: {path}')
        tensors[path.as_posix()] = dict(sha256=digest(path), comparisons=len(checks),
            matches_current_native_source=matched,
            max_value_difference=max(c['max_value_difference'] for c in checks),
            max_vjp_difference=max(c['max_vjp_difference'] for c in checks))
    runtimes = {}
    for path in args.runtime_report:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload['native_source_sha256'] != current:
            raise RuntimeError(f'Runtime diagnostic native-source mismatch: {path}')
        if not payload['samples']:
            raise RuntimeError(f'Empty runtime diagnostic: {path}')
        combined['results'].extend(dict(case=payload['case'], **sample) for sample in payload['samples'])
        runtimes[path.as_posix()] = dict(sha256=digest(path), samples=len(payload['samples']))
    curvatures = {}
    for path in args.curvature_report+args.historical_curvature_report:
        payload = json.loads(path.read_text(encoding='utf-8'))
        matched = payload['native_source_sha256'] == current
        if path in args.curvature_report and not matched:
            raise RuntimeError(f'Curvature native-source mismatch: {path}')
        curvatures[path.as_posix()] = dict(sha256=digest(path), matches_current_native_source=matched,
                                         validation=validate_curvature(payload))
    flags = ['PASSIVE_CONSTANTS', 'SAMPLE_CACHE', 'NATIVE_ADJOINT', 'PACKED_ARRAYS', 'LAZY_ARRAYS',
             'VECTOR_SAMPLING', 'ARRAY_KERNELS', 'PRICE_ONLY', 'SPLINE_KERNELS', 'TRIDIAGONAL_ADJOINT', 'PCR_ROLL',
             'SPATIAL_CACHE', 'OPERATOR_ADJOINT', 'STEP_ADJOINT', 'ASSEMBLY_ADJOINT', 'PCR_FUSED', 'PCR_NATIVE_STAGE',
             'NATIVE_SOLVE_CHECKS', 'ASSEMBLY_ARITHMETIC', 'OPERATOR_ARITHMETIC', 'ARITHMETIC_NATIVE_FORWARD',
             'THOMAS_ARITHMETIC', 'THOMAS_VECTOR_BACKWARD', 'ADJOINT_ARITHMETIC_FORWARD', 'STEP_NATIVE_ARITHMETIC',
             'BASIS_CACHE', 'FULL_TENSOR_STEP', 'TIME_VARIANCE_ARITHMETIC', 'STEP_PREPARATION', 'ASSEMBLED_BANDS',
             'ADJOINT_ASSEMBLED_BANDS', 'DIFFUSION_SELECTED_GRADS']
    python_hashes = {p.relative_to(ROOT).as_posix(): digest(p)
                     for directory in ('python/quantlibtorch', 'tests', 'scripts')
                     for p in sorted((ROOT/directory).glob('*.py'))}
    build_hashes = {p.relative_to(ROOT).as_posix(): digest(p)
                    for p in [ROOT/'CMakeLists.txt', ROOT/'pyproject.toml', *sorted((ROOT/'cmake').glob('*.cmake'))]}
    archive = None
    if args.source_archive:
        expected = {**current, **python_hashes, **build_hashes}
        with zipfile.ZipFile(args.source_archive) as snapshot:
            if snapshot.testzip() is not None:
                raise RuntimeError('Source archive CRC mismatch')
            for name, fingerprint in expected.items():
                if hashlib.sha256(snapshot.read(name)).hexdigest() != fingerprint:
                    raise RuntimeError(f'Source archive differs from current workspace: {name}')
        archive = dict(path=args.source_archive.as_posix(), sha256=digest(args.source_archive),
                       verified_source_files=len(expected))
    output = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        native_source_sha256=current,
        native_binary_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in library_paths},
        python_and_test_sha256=python_hashes, build_configuration_sha256=build_hashes,
        generated_patch_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                                for p in sorted((ROOT/'build/native/quantlib-overlay').glob('*.patch'))},
        quantlib_commit=subprocess.check_output(['git', '-C', str(ROOT/'third_party/QuantLib'), 'rev-parse', 'HEAD'], text=True).strip(),
        quantlib_worktree=subprocess.check_output(['git', '-C', str(ROOT/'third_party/QuantLib'), 'status', '--porcelain'], text=True),
        performance_environment={'QLT_'+flag: os.environ.get('QLT_'+flag) for flag in flags},
        reports=records, all_samples_validation=validate_results(combined), logs=logs,
        overlay_comparison=overlay, tensor_reports=tensors, runtime_reports=runtimes, curvature_reports=curvatures,
        source_archive=archive,
        notes=['Historical source mismatches are retained and explicitly marked.',
               'Cross-report validation uses one fixed reference per financial case and includes every raw warmup/timed price/VJP, not timing ratios.',
               'Hashes associate the current workspace, binary and supplied logs; this script does not rebuild or rerun tests.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output['all_samples_validation'], indent=2))


if __name__ == '__main__':
    main()
