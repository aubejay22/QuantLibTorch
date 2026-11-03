"""Verify and archive Phase 22 evidence after a validated implementation commit."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/benchmarks'


def main():
    sha=lambda data:hashlib.sha256(data).hexdigest()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    baseline='3e66775'
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),baseline_git=git('rev-parse',baseline),
        implementation_git=git('rev-parse','HEAD'),quantlib_pin=git('-C','third_party/QuantLib','rev-parse','HEAD'))
    fit=json.loads((OUT/'calibration_phase22_fit.json').read_text())
    manifest['environment']=dict(platform=platform.platform(),processor=platform.processor(),
        logical_cpus=os.cpu_count(),python=fit['python'],torch=fit['torch'])
    assert manifest['quantlib_pin']=='06edddbdb85f006c43a449b5fc846a00c67477f3'
    assert not git('-C','third_party/QuantLib','status','--porcelain')
    assert not git('diff',baseline,'--','cmake/quantlib_sources.cmake','CMakeLists.txt','scripts/prepare_quantlib.py')
    assert not git('diff',baseline,'--','docs/benchmarks/neural_sde_phase21*')
    prior=json.loads((OUT/'neural_sde_phase21_preservation.json').read_text())
    for path,expected in prior['artifact_sha256'].items():
        assert sha((OUT/path).read_bytes())==expected,path
    manifest['preserved_phase21_artifacts']=len(prior['artifact_sha256'])
    old_cpp=git('ls-tree','-r','--name-only',baseline,'--','cpp').splitlines()
    unchanged=[p for p in old_cpp if p!='cpp/coefficient_pde.cpp']
    assert not git('diff',baseline,'--',*unchanged)
    manifest['unchanged_prior_cpp_files']=unchanged
    manifest['native_change']='coefficient_pde.cpp: profiling scopes and an internal financial-generator validation endpoint only'
    binary=ROOT/'python/quantlibtorch/_native/quantlibtorch_native.dll'
    if not binary.exists():
        binary=next(p for p in binary.parent.glob('*native.*') if p.suffix in ('.so','.dylib'))
    manifest['binary_sha256']=sha(binary.read_bytes())
    verified=[]
    reports=list(OUT.glob('calibration_phase22_*.json'))+list((OUT/'calibration_phase22_workloads').glob('*.json'))
    for path in sorted(reports):
        if path.name=='calibration_phase22_preservation.json':continue
        record=json.loads(path.read_text())
        if not isinstance(record,dict) or 'source_sha256' not in record:continue
        for source,expected in record['source_sha256'].items():
            assert sha((ROOT/source).read_bytes())==expected,(path,source)
        assert record['binary_sha256']==manifest['binary_sha256'],path
        verified.append(str(path.relative_to(OUT)))
    manifest['verified_fingerprinted_reports']=verified
    fixture=ROOT/'build/phase22_market_check.json'
    subprocess.run([sys.executable,'scripts/prepare_calibration_market.py','--output',str(fixture)],cwd=ROOT,check=True)
    assert fixture.read_bytes()==(OUT/'calibration_phase22_market.json').read_bytes()
    manifest['independent_market_fixture_reproduced_byte_for_byte']=True
    commands=json.loads((OUT/'calibration_phase22_commands.json').read_text())
    assert all(r['returncode']==0 for r in commands)
    assert {'convergence','finance','quote_audit','gradients_refined','gradients_full','gradients_adjoint',
            'workloads','full_tests','refinement','truth','python_profile'} <= {r['name'] for r in commands}
    logs={'build/phase22_build.log':'calibration_phase22_build.txt',
          'build/phase22_full_tests.log':'calibration_phase22_tests.txt',
          'build/phase22_targeted_tests.log':'calibration_phase22_targeted_tests.txt',
          'build/phase22_fit.log':'calibration_phase22_fit_log.txt',
          'build/phase22_workloads.log':'calibration_phase22_workloads.txt'}
    for source,destination in logs.items():
        raw=(ROOT/source).read_bytes()
        contents=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        contents='\n'.join(line.rstrip() for line in contents.splitlines())+'\n'
        (OUT/destination).write_text(contents,encoding='utf-8',newline='\n')
    assert '1756 passed' in (OUT/'calibration_phase22_tests.txt').read_text()
    assert '14 passed' in (OUT/'calibration_phase22_targeted_tests.txt').read_text()
    overlay=ROOT/'build/native/quantlib-overlay'
    check=ROOT/'build/phase22_overlay_check'
    subprocess.run([sys.executable,'scripts/prepare_quantlib.py','third_party/QuantLib',str(check)],cwd=ROOT,check=True)
    files=sorted(p.relative_to(overlay) for p in overlay.rglob('*') if p.is_file())
    assert files==sorted(p.relative_to(check) for p in check.rglob('*') if p.is_file())
    for path in files:assert (overlay/path).read_bytes()==(check/path).read_bytes(),path
    manifest['regenerated_overlay_files']=len(files)
    names=git('ls-files').splitlines()
    sources=[n for n in names if n.startswith(('cpp/','python/','tests/','scripts/','examples/','cmake/','LICENSES/'))
             and Path(n).suffix in ('.cpp','.hpp','.py','.cmake','.txt','.md')]
    sources+=['CMakeLists.txt','pyproject.toml','AGENTS.md','PROJECT_CONTEXT.md','README.md','docs/architecture.md']
    sources=sorted(set(sources))
    assert 'scripts/preserve_calibration.py' in sources,'Commit the implementation sources before preserving evidence'
    manifest['source_sha256']={p:sha((ROOT/p).read_bytes()) for p in sources}
    archive_path=OUT/'calibration_phase22_source.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:archive.write(ROOT/path,path)
        for path in files:archive.write(overlay/path,'quantlib-overlay/'+path.as_posix())
        for path in ['build/phase22_finish.py','build/phase22_deep.py','build/phase22_run_initial.py']:
            archive.write(ROOT/path,path)
    with zipfile.ZipFile(archive_path) as archive:
        for path,expected in manifest['source_sha256'].items():assert sha(archive.read(path))==expected
    manifest['archive_sha256']=sha(archive_path.read_bytes())
    manifest['artifact_sha256']={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.glob('calibration_phase22*'))
        if p.is_file() and p.name!='calibration_phase22_preservation.json'}
    manifest['artifact_sha256'].update({str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted((OUT/'calibration_phase22_workloads').glob('*.json'))})
    (OUT/'calibration_phase22_preservation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(source_files=len(sources),reports=len(verified),overlays=len(files),
                         archive_sha256=manifest['archive_sha256'],implementation_git=manifest['implementation_git']),indent=2))


if __name__=='__main__':main()
