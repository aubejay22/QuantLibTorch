"""Audit prior evidence/provenance and archive the validated Phase 24 milestone."""
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/benchmarks'


def main():
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    baseline='27ba330'
    assert git('-C', 'third_party/QuantLib', 'rev-parse', 'HEAD')=='06edddbdb85f006c43a449b5fc846a00c67477f3'
    assert not git('-C', 'third_party/QuantLib', 'status', '--porcelain')
    untouched = ['cpp/coefficient_pde.cpp', 'cpp/structured_ops.cpp', 'cpp/parallel_solve.cpp',
                 'cpp/thomas_arithmetic.cpp', 'scripts/prepare_quantlib.py', 'cmake/quantlib_sources.cmake',
                 'python/quantlibtorch/latent.py', 'python/quantlibtorch/calibration.py', 'CMakeLists.txt']
    assert not git('diff', baseline, '--', *untouched)
    previous = json.loads((OUT/'packed_phase23_preservation.json').read_text())
    for path, expected in {**previous['artifact_sha256'], **previous['workload_sha256']}.items():
        assert sha(OUT/path)==expected, path
    assert sha(ROOT/'build/phase24_baseline/quantlibtorch_native.dll')==previous['binary_sha256']
    generated=ROOT/'build/native/packed-quantlib'
    with zipfile.ZipFile(OUT/'packed_phase23_source.zip') as old:
        assert old.read('generated/packed_stages.inc')==(generated/'packed_stages.inc').read_bytes()
    assert (generated/'multi_stages.inc').read_text().replace('void MultiEvolver::', 'void PackedEvolver::')==(
        generated/'packed_stages.inc').read_text()
    logs={'phase24_build.txt':'multi_asset_phase24_build.txt',
          'phase24_targeted_tests.txt':'multi_asset_phase24_targeted_tests.txt',
          'phase24_full_tests.txt':'multi_asset_phase24_tests.txt',
          'phase24_training_example.txt':'multi_asset_phase24_training_example.txt'}
    for source, target in logs.items():
        raw=(ROOT/'build'/source).read_bytes()
        value=raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        (OUT/target).write_text('\n'.join(line.rstrip() for line in value.splitlines())+'\n', encoding='utf-8', newline='\n')
    assert '1940 passed' in (OUT/'multi_asset_phase24_tests.txt').read_text()
    summary=json.loads((OUT/'multi_asset_phase24_summary.json').read_text())
    assert not summary['pending_reports']
    assert summary['maximum_price_difference']<1e-9 and summary['maximum_gradient_difference']<1e-7
    for commands in (OUT/'multi_asset_phase24_workloads').glob('commands_*.json'):
        assert all(row['returncode']==0 for row in json.loads(commands.read_text())), commands
    # Windows subprocess stdout and older command ledgers use CRLF. Preserve
    # their text in canonical LF form so Git's whitespace checks remain useful.
    for path in (OUT/'multi_asset_phase24_workloads').iterdir():
        if path.suffix=='.log' or path.name.startswith('commands_'):
            value='\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n'
            path.write_text(value, encoding='utf-8', newline='\n')
    # Keep exact raw profiler evidence compact in Git. The uncompressed originals
    # remain in build; verify both paths are inside this workspace before moving.
    trace_dir=ROOT/'build/phase24_traces'
    trace_dir.mkdir(exist_ok=True)
    for trace in (OUT/'multi_asset_phase24_workloads').glob('*.trace.json'):
        archive_path=trace.with_suffix(trace.suffix+'.gz')
        raw=trace.read_bytes()
        archive_path.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        assert gzip.decompress(archive_path.read_bytes())==raw
        target=trace_dir/trace.name
        assert trace.resolve().is_relative_to(ROOT.resolve()) and target.resolve().is_relative_to(ROOT.resolve())
        assert not target.exists() or sha(target)==sha(trace)
        original_hash=sha(trace)
        trace.replace(target)
        report=trace.with_name(trace.name.replace('.trace.json', '.json'))
        data=json.loads(report.read_text())
        data['trace_file']=archive_path.relative_to(ROOT).as_posix()
        data['trace_preservation']=dict(compression='gzip', uncompressed_sha256=original_hash,
                                        compressed_sha256=sha(archive_path))
        report.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
    # Read tracked and newly authored source files, excluding caches/dependencies.
    sources=[]
    for folder in ['cpp', 'python', 'scripts', 'tests', 'examples', 'cmake', 'LICENSES']:
        sources += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in
                    ('.cpp', '.hpp', '.inc', '.py', '.cmake', '.txt', '.md') and '__pycache__' not in p.parts]
    sources += [ROOT/p for p in ['CMakeLists.txt', 'README.md', 'PROJECT_CONTEXT.md', 'AGENTS.md', 'pyproject.toml', 'docs/architecture.md']]
    archive=OUT/'multi_asset_phase24_source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources): z.write(p, p.relative_to(ROOT).as_posix())
        for p in sorted(generated.glob('*')): z.write(p, 'generated/'+p.name)
    binary=ROOT/'python/quantlibtorch/_native/quantlibtorch_native.dll'
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(), baseline_git=git('rev-parse', baseline),
                  implementation_checkpoint='43d6113', archive_parent_git=git('rev-parse', 'HEAD'),
                  archive_source_is_hashed_worktree=True,
                  binary_sha256=sha(binary), preserved_baseline_binary_sha256=previous['binary_sha256'],
                  quantlib_pin=git('-C', 'third_party/QuantLib', 'rev-parse', 'HEAD'),
                  unchanged_sources=untouched, unchanged_phase23_generated_stages=True,
                  unchanged_prior_artifacts=len(previous['artifact_sha256'])+len(previous['workload_sha256']),
                  test_evidence=dict(initial_targeted_source_git='43d6113',
                      final_full_suite='1940 passed, 1 skipped', final_multi_asset_test_sha256=sha(ROOT/'tests/test_multi_asset.py')),
                  source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
                  artifact_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.glob('multi_asset_phase24*'))
                                   if p.is_file() and not p.name.endswith('preservation.json')},
                  workloads_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT/'multi_asset_phase24_workloads').iterdir())
                                    if p.is_file() and not p.name.endswith('.trace.json')})
    for path in list((OUT/'multi_asset_phase24_workloads').glob('*.json'))+list(OUT.glob('multi_asset_phase24_convergence*.json'))+list(OUT.glob('multi_asset_phase24_gradients*.json')):
        if path.name.endswith('.trace.json'): continue
        data=json.loads(path.read_text())
        if not isinstance(data, dict) or 'source_sha256' not in data: continue
        assert data['binary_sha256']==manifest['binary_sha256'], path
        for source, digest in {**data['source_sha256'], **data.get('diagnostic_source_sha256', {})}.items():
            assert sha(ROOT/source)==digest, (path, source)
    (OUT/'multi_asset_phase24_preservation.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(sources=len(sources), previous_artifacts=manifest['unchanged_prior_artifacts'], archive=str(archive))))


if __name__=='__main__':
    main()
