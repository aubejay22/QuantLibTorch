"""Verify Phase 23 evidence, preserve sources, and audit upstream/Phase 22 hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/benchmarks'


def main():
    def git(*args, raw=False):
        output = subprocess.check_output(['git', *args], cwd=ROOT)
        return output if raw else output.decode().strip()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline_git=git('rev-parse', '12ec6ee'), cpu_checkpoint=git('rev-parse', '93cd935'),
        implementation_git=git('rev-parse', 'HEAD'), quantlib_pin=git('-C', 'third_party/QuantLib', 'rev-parse', 'HEAD'))
    assert manifest['quantlib_pin'] == '06edddbdb85f006c43a449b5fc846a00c67477f3'
    assert not git('-C', 'third_party/QuantLib', 'status', '--porcelain')
    assert not git('diff', '12ec6ee', '--', 'scripts/prepare_quantlib.py', 'cmake/quantlib_sources.cmake',
                   'cpp/coefficient_pde.cpp', 'cpp/structured_ops.cpp', 'python/quantlibtorch/latent.py')
    assert not git('diff', '12ec6ee', '--', 'docs/benchmarks/calibration_phase22*')
    manifest['unchanged_original_coefficient_solver_and_compatibility_overlay'] = True
    previous = json.loads((OUT/'calibration_phase22_preservation.json').read_text())
    for path, expected in previous['artifact_sha256'].items():
        assert sha((OUT/path).read_bytes()) == expected, path
    manifest['verified_phase22_artifacts'] = len(previous['artifact_sha256'])
    native = next(p for p in (ROOT/'python/quantlibtorch/_native').glob('*native.*') if p.suffix in ('.dll', '.so', '.dylib'))
    manifest['binary_sha256'] = sha(native.read_bytes())
    summary = json.loads((OUT/'packed_phase23_summary.json').read_text())
    assert summary['verified_reports'] == 47
    assert summary['maximum_price_difference'] < 1e-10
    assert summary['maximum_gradient_difference'] < 1e-8
    for commands in [OUT/'packed_phase23_workloads/commands.json', OUT/'packed_phase23_validation_commands.json']:
        assert all(r['returncode'] == 0 for r in json.loads(commands.read_text())), commands
    baseline_fit = json.loads((OUT/'calibration_phase22_fit.json').read_text())['fit']
    for mode in ('full', 'adjoint'):
        fitted = json.loads((OUT/f'packed_phase23_fit_{mode}.json').read_text())['fit']
        assert fitted['accepted_iterations'] == 15 and len(fitted['history']) == 16
        assert abs(fitted['final_rmse']-baseline_fit['final_rmse']) < 1e-10
        assert max(abs(a-b) for a,b in zip(fitted['final_prices'],baseline_fit['final_prices'])) < 1e-10
    for kind in ('cpu', 'full', 'cuda'):
        record = json.loads((OUT/f'packed_phase23_gradients_{kind}.json').read_text())
        assert max(r['absolute_error'] for r in record['gradient_checks']) < 1e-6
    accuracy = json.loads((OUT/'packed_phase23_accuracy.json').read_text())['records']
    assert accuracy[-1]['heston_max_price_error'] < accuracy[0]['heston_max_price_error']/20
    assert accuracy[-1]['black_max_vega_error'] < accuracy[0]['black_max_vega_error']/20
    assert all(a < b/20 for a,b in zip(accuracy[-1]['heston_max_sensitivity_errors'],accuracy[0]['heston_max_sensitivity_errors']))
    manifest['numerical_and_workload_evidence_checks_passed'] = True
    reports = list((OUT/'packed_phase23_workloads').glob('*.json'))+list(OUT.glob('packed_phase23_*.json'))
    manifest['verified_reports'] = []
    for path in sorted(reports):
        if path.name.endswith('preservation.json'):
            continue
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or 'source_sha256' not in record:
            continue
        for source, expected in record['source_sha256'].items():
            assert sha((ROOT/source).read_bytes()) == expected, (path, source)
        assert record['binary_sha256'] == manifest['binary_sha256'], path
        manifest['verified_reports'].append(path.relative_to(ROOT).as_posix())
    manifest['verified_cpu_pilot_reports'] = []
    for path in sorted((OUT/'packed_phase23_cpu_pilot').glob('*.json')):
        record = json.loads(path.read_text())
        for source, expected in record['source_sha256'].items():
            name = Path(source).as_posix()
            assert sha(git('show', f'93cd935:{name}', raw=True)) == expected, (path, source)
        manifest['verified_cpu_pilot_reports'].append(path.relative_to(ROOT).as_posix())
    logs = {'phase23_cpu_targeted_tests.txt': 'packed_phase23_cpu_tests.txt',
            'phase23_cpu_full_tests.txt': 'packed_phase23_cpu_full_tests.txt',
            'phase23_mutation_test.txt': 'packed_phase23_mutation_test.txt',
            'phase23_gpu_targeted_tests.txt': 'packed_phase23_targeted_tests.txt',
            'phase23_full_tests.txt': 'packed_phase23_tests.txt',
            'phase23_build_retry.txt': 'packed_phase23_cpu_build.txt',
            'phase23_gpu_build.txt': 'packed_phase23_build.txt'}
    for source, target in logs.items():
        raw = (ROOT/'build'/source).read_bytes()
        content = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        (OUT/target).write_text('\n'.join(line.rstrip() for line in content.splitlines())+'\n', encoding='utf-8', newline='\n')
    assert '1865 passed' in (OUT/'packed_phase23_tests.txt').read_text()
    assert '109 passed' in (OUT/'packed_phase23_targeted_tests.txt').read_text()
    check = ROOT/'build/phase23_regenerated_stages'
    subprocess.run([sys.executable, 'scripts/prepare_packed.py', 'third_party/QuantLib', str(check)], cwd=ROOT, check=True)
    generated = ROOT/'build/native/packed-quantlib'
    for name in ('packed_stages.inc', 'packed_stages.json'):
        assert (check/name).read_bytes() == (generated/name).read_bytes(), name
    manifest['generated_stages'] = json.loads((generated/'packed_stages.json').read_text())
    fixture = ROOT/'build/phase23_market_check.json'
    subprocess.run([sys.executable, 'scripts/benchmark_packed_calibration.py', 'prepare', '--output', str(fixture)], cwd=ROOT, check=True)
    assert fixture.read_bytes() == (OUT/'packed_phase23_market.json').read_bytes()
    manifest['independent_market_fixture_reproduced_byte_for_byte'] = True
    sources = [ROOT/n for n in git('ls-files').splitlines() if n.startswith(('cpp/', 'python/', 'scripts/', 'tests/', 'examples/', 'cmake/', 'LICENSES/'))
               and Path(n).suffix in ('.cpp', '.hpp', '.py', '.cmake', '.txt', '.md')]
    sources += [ROOT/n for n in ['CMakeLists.txt', 'PROJECT_CONTEXT.md', 'README.md', 'AGENTS.md', 'docs/architecture.md', 'pyproject.toml']]
    manifest['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in sorted(set(sources))}
    archive = OUT/'packed_phase23_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(sources)):
            z.write(p, p.relative_to(ROOT).as_posix())
        for p in sorted(generated.glob('*')):
            z.write(p, 'generated/'+p.name)
    manifest['artifact_sha256'] = {p.relative_to(OUT).as_posix(): sha(p.read_bytes()) for p in sorted(OUT.glob('packed_phase23*'))
                                  if p.is_file() and not p.name.endswith('preservation.json')}
    manifest['workload_sha256'] = {p.relative_to(OUT).as_posix(): sha(p.read_bytes()) for p in sorted((OUT/'packed_phase23_workloads').glob('*.json'))}
    (OUT/'packed_phase23_preservation.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(sources=len(sources), reports=len(manifest['verified_reports']),
                         prior_artifacts=manifest['verified_phase22_artifacts'], archive=str(archive))))


if __name__ == '__main__':
    main()
