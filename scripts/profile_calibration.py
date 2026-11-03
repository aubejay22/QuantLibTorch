"""Bounded Python/native timing attribution for the actual calibration grid.

ATen event traces on tiny meshes identify kernel/graph costs separately. This
profile measures one complete fitted-model objective/VJP on 81 x 17 without
retaining an event for every tensor operation. Times include cProfile overhead.
"""
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    from runtime_controls import worker_qos
    runtime=worker_qos('high' if sys.platform=='win32' else 'system',8 if sys.platform=='win32' else None)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'examples'))
    from calibrate_neural_sde import NeuralVarianceModel,restore,make_problem
    market=ROOT/'docs/benchmarks/calibration_phase22_market.json'
    fit=ROOT/'docs/benchmarks/calibration_phase22_fit.json'
    model=NeuralVarianceModel();restore(model,json.loads(fit.read_text())['parameters'])
    problem=make_problem(json.loads(market.read_text()),'space')
    def evaluate():
        model.zero_grad(set_to_none=True)
        loss,prices=problem.backward(model,model.initial_latent,mode='adjoint')
        penalty=model.penalty();penalty.backward()
        return dict(prices=prices.tolist(),scaled_loss=float(loss),penalty=float(penalty.detach()),
                    gradients={k:p.grad.tolist() for k,p in model.named_parameters()})
    profiler=cProfile.Profile()
    result=profiler.runcall(evaluate)
    stats=pstats.Stats(profiler)
    events=[]
    for (file,line,function),(primitive,calls,self_time,total_time,callers) in stats.stats.items():
        events.append(dict(file=file,line=line,function=function,calls=calls,primitive_calls=primitive,
                           self_seconds=self_time,inclusive_seconds=total_time))
    events.sort(key=lambda row:row['inclusive_seconds'],reverse=True)
    paths=[Path(__file__),ROOT/'python/quantlibtorch/calibration.py',ROOT/'examples/calibrate_neural_sde.py',
           ROOT/'cpp/coefficient_pde.cpp',market,fit]
    native=next(p for p in (ROOT/'python/quantlibtorch/_native').glob('*native.*') if p.suffix in ('.dll','.so','.dylib'))
    report=dict(runtime=runtime,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        binary_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),evaluation=result,events=events,
        total_seconds=stats.total_tt,total_calls=stats.total_calls,
        note='One streamed adjoint objective/VJP, fitted model, all nine contracts, 81 x 17, 24 steps/year. cProfile overhead included; no tensor event retention.')
    (ROOT/'docs/benchmarks/calibration_phase22_python_profile.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(total_seconds=stats.total_tt,events=events[:18]),indent=2))


if __name__=='__main__':main()
