"""Further independent mesh/time refinement of the frozen calibrated model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/benchmarks'


def main():
    from runtime_controls import worker_qos
    runtime=worker_qos('high' if sys.platform=='win32' else 'system',8 if sys.platform=='win32' else None)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'examples'))
    from calibrate_neural_sde import NeuralVarianceModel,restore,make_problem
    market=OUT/'calibration_phase22_market.json'
    fit=OUT/'calibration_phase22_fit.json'
    data=json.loads(market.read_text())
    request=dict(meshes={'fine_space':dict(nx=161,nv=33,steps_per_year=48),
                         'fine':dict(nx=161,nv=65,steps_per_year=96)})
    mesh_response=json.loads(subprocess.run([sys.executable,str(ROOT/'scripts/calibration_stock_reference.py')],
        input=json.dumps(request),text=True,capture_output=True,check=True).stdout)
    data['meshes'].update(mesh_response['meshes'])
    data['meshes']['fine_time']=dict(data['meshes']['refined'],steps_per_year=96)
    model=NeuralVarianceModel();restore(model,json.loads(fit.read_text())['parameters'])
    paths=[Path(__file__),ROOT/'examples/calibrate_neural_sde.py',ROOT/'python/quantlibtorch/calibration.py',
           ROOT/'scripts/calibration_stock_reference.py',ROOT/'cpp/coefficient_pde.cpp',market,fit]
    native=next(p for p in (ROOT/'python/quantlibtorch/_native').glob('*native.*') if p.suffix in ('.dll','.so','.dylib'))
    result=dict(runtime=runtime,torch=torch.__version__,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        binary_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),meshes={k:data['meshes'][k] for k in ['fine_time','fine_space','fine']},
        note='Frozen fit; inference only. Time and space controls followed by combined refinement. No optimizer updates.',results=[])
    def save():
        (OUT/'calibration_phase22_refinement.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    for level in ['fine_time','fine_space','fine']:
        for heldout in [False,True]:
            problem=make_problem(data,level,heldout=heldout)
            start=time.perf_counter()
            with torch.no_grad():prices=problem.prices(model,model.initial_latent,mode='adjoint')
            row=dict(mesh=level,heldout=heldout,prices=prices.tolist(),
                rmse=float((prices-problem.targets).square().mean().sqrt()),
                max_target_error=float((prices-problem.targets).abs().max()),seconds=time.perf_counter()-start)
            result['results'].append(row);save();print(json.dumps(row),flush=True)
    # Analytic Heston supplies an independent numerical-error control on the
    # deepest mesh as well as the frozen neural-model convergence experiment.
    import math
    with torch.no_grad():
        model.net[-1].weight.zero_();model.net[-1].bias.zero_()
        model.raw_initial.fill_(math.log(.045/(.12-.045)))
        problem=make_problem(data,'fine')
        start=time.perf_counter();prices=problem.prices(model,model.initial_latent,mode='adjoint')
    result['heston_control']=dict(prices=prices.tolist(),rmse=float((prices-problem.targets).square().mean().sqrt()),
        max_target_error=float((prices-problem.targets).abs().max()),seconds=time.perf_counter()-start)
    save();print(json.dumps(result['heston_control']),flush=True)


if __name__=='__main__':main()
