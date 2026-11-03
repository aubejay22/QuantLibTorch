"""Reproducible multi-maturity calibration and isolated workload diagnostics.

Every invocation is one worker. Run expensive workers sequentially, especially
memory and profiler jobs. Source and binary fingerprints accompany each report.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["truth", "convergence", "fit", "gradients", "timing", "memory", "profile", "finance"])
    parser.add_argument("--market", type=Path, default=ROOT/'docs/benchmarks/calibration_phase22_market.json')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--mesh", default="calibration")
    parser.add_argument("--mode", choices=["full", "adjoint"], default="adjoint")
    parser.add_argument("--execution", choices=["streamed", "stacked"], default="streamed")
    parser.add_argument("--contracts", type=int, default=9)
    parser.add_argument("--latent-dimensions", type=int, choices=[1,2], default=1)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--windows-qos", choices=["system", "high"], default="system")
    parser.add_argument("--affinity", type=int)
    args = parser.parse_args()
    from runtime_controls import worker_qos
    runtime = worker_qos(args.windows_qos, args.affinity)
    import torch
    from quantlibtorch import LatentCalibration, OptionQuote, fd_latent, state_grid
    from calibrate_neural_sde import NeuralVarianceModel, make_problem, parameter_values, restore, train
    from investigate_fd import process_memory
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = json.loads(args.market.read_text())
    model = NeuralVarianceModel(args.latent_dimensions)
    if args.parameters:
        restore(model, json.loads(args.parameters.read_text())["parameters"])
    if args.task == "truth":
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
            model.raw_initial.fill_(math.log(.045/(.12-.045)))
    problem = make_problem(data, args.mesh, args.contracts, args.latent_dimensions)
    paths = [ROOT/'python/quantlibtorch/calibration.py',ROOT/'cpp/coefficient_pde.cpp',Path(__file__),
             ROOT/'examples/calibrate_neural_sde.py',ROOT/'scripts/calibration_stock_reference.py',args.market]
    native = next(p for p in (ROOT/'python/quantlibtorch/_native').glob('*native.*') if p.suffix in ('.dll','.so','.dylib'))
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), configuration=vars(args).copy(),
        runtime=runtime, python=sys.version, torch=torch.__version__, baseline_git="3e66775",
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        binary_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        parameters=parameter_values(model), nodes=len(problem.states))
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,default=str)+"\n",encoding="utf-8",newline="\n")

    def gradients():
        return {k: (torch.zeros_like(v) if v.grad is None else v.grad).detach().tolist() for k,v in model.named_parameters()}

    def run(context=None):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        if args.execution == "streamed":
            loss, prices = problem.backward(model,model.initial_latent,mode=args.mode,contract_context=context)
        else:
            loss, prices = problem.loss(model,model.initial_latent,mode=args.mode)
            loss.backward()
        penalty = model.penalty()
        penalty.backward()
        values = dict(loss=float(loss.detach()),penalty=float(penalty.detach()),prices=prices.detach().tolist(),gradients=gradients())
        del loss,prices,penalty
        values["seconds"] = time.perf_counter()-start
        return values

    if args.task in ("truth","convergence"):
        records=[]
        for level in ["coarse","calibration","time","space","variance","refined"]:
            p = make_problem(data,level,args.contracts,args.latent_dimensions)
            with torch.no_grad():
                start=time.perf_counter()
                prices=p.prices(model,model.initial_latent,mode=args.mode)
            records.append(dict(mesh=level,prices=prices.tolist(),rmse=float((prices-p.targets).square().mean().sqrt()),
                                max_target_error=float((prices-p.targets).abs().max()),seconds=time.perf_counter()-start))
            result["convergence"]=records
            print(json.dumps(records[-1]),flush=True);save()
        # Enlarge the domain without altering any previous interior coordinates.
        base=data["meshes"]["calibration"]
        extended=json.loads(json.dumps(data))
        extended["meshes"]["domain"]=dict(base,x=[math.log(5.),math.log(10.),*base["x"],math.log(800.),math.log(1600.)],
                                         v=[*base["v"],.8,1.2])
        for name, modified in [("domain",extended),("coefficient_times",dict(data,times=[i/16 for i in range(17)]))]:
            p=make_problem(modified,"domain" if name=="domain" else "calibration",args.contracts,args.latent_dimensions)
            with torch.no_grad(): prices=p.prices(model,model.initial_latent,mode=args.mode)
            records.append(dict(mesh=name,prices=prices.tolist(),rmse=float((prices-p.targets).square().mean().sqrt())))
            save()
        if args.task=="convergence":
            for level in ["calibration","refined"]:
                p=make_problem(data,level,latent_dimensions=args.latent_dimensions,heldout=True)
                with torch.no_grad(): prices=p.prices(model,model.initial_latent,mode=args.mode)
                result.setdefault("heldout",[]).append(dict(mesh=level,prices=prices.tolist(),rmse=float((prices-p.targets).square().mean().sqrt())))
    elif args.task == "fit":
        result["fit"]=train(data,mode=args.mode,iterations=args.iterations,level=args.mesh,
                             output=args.output.with_suffix('.training.json'),resume=parameter_values(model))
        result["parameters"]=result["fit"]["parameters"]
    elif args.task == "gradients":
        result["evaluation"]=run()
        named=dict(model.named_parameters())
        original={k:p.detach().clone() for k,p in named.items()}
        directions={k:torch.cos(torch.arange(p.numel(),dtype=torch.float64)).reshape_as(p)*.05 for k,p in named.items()}
        checks=[]
        for group in ["network","initial_state"]:
            selected=[k for k in named if (k=="raw_initial")==(group=="initial_state")]
            ad=sum(float((torch.tensor(result["evaluation"]["gradients"][k],dtype=torch.float64)*directions[k]).sum()) for k in selected)
            for h in [1e-3,3e-4]:
                vals=[]
                with torch.no_grad():
                    for sign in [1.,-1.]:
                        for k in selected: named[k].copy_(original[k]+sign*h*directions[k])
                        value=problem.loss(model,model.initial_latent,mode=args.mode)[0]+model.penalty()
                        vals.append(float(value))
                    for k in selected: named[k].copy_(original[k])
                fd=(vals[0]-vals[1])/(2*h)
                checks.append(dict(group=group,h=h,ad=ad,finite_difference=fd,absolute_error=abs(ad-fd)))
                result["gradient_checks"]=checks;save()
    elif args.task == "timing":
        result["warmups"]=[run() for _ in range(args.warmup)]
        samples=[]
        for _ in range(args.repeats):
            gc.collect();samples.append(run())
            result["samples"]=samples;save()
        result["median_seconds"]=statistics.median(s["seconds"] for s in samples)
    elif args.task == "memory":
        gc.collect();result["before"]=process_memory()
        stages=[]
        @contextmanager
        def memory(i):
            yield
            stages.append(dict(contract=i,**process_memory()))
        result["evaluation"]=run(memory)
        gc.collect();result["after"]=process_memory()
        result["contract_memory"]=stages
        result["peak_growth_bytes"]=result["after"]["peak_rss_bytes"]-result["before"]["peak_rss_bytes"]
    elif args.task == "profile":
        totals={};per_contract=[]
        @contextmanager
        def profile(i):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,
                                        profile_memory=False,with_stack=False) as prof:
                yield
            events=prof.key_averages()
            per_contract.append(dict(contract=i,aten_calls=sum(e.count for e in events if e.key.startswith('aten::'))))
            for e in events:
                if e.key.startswith(('qlt::','calibration::','aten::')) or 'Backward' in e.key:
                    row=totals.setdefault(e.key,dict(name=e.key,calls=0,self_cpu_us=0.,inclusive_cpu_us=0.))
                    row['calls']+=e.count;row['self_cpu_us']+=e.self_cpu_time_total;row['inclusive_cpu_us']+=e.cpu_time_total
            del events,prof
            gc.collect()
        if args.execution!='streamed': raise ValueError("bounded profiling requires streamed execution")
        result["evaluation"]=run(profile)
        result["events"]=sorted(totals.values(),key=lambda e:e['self_cpu_us'],reverse=True)
        result["contracts"]=per_contract
    elif args.task == "finance":
        # Independently analytic BS quotes, call-put parity and strike shape.
        contracts=[]
        for q in data['quotes']:
            contracts.extend([dict(strike=q['strike'],maturity=q['maturity'],option_type=kind) for kind in ['call','put']])
        oracle=json.loads(subprocess.run([sys.executable,str(ROOT/'scripts/calibration_stock_reference.py')],
            input=json.dumps(dict(contracts=contracts,black_scholes=True)),text=True,capture_output=True,check=True).stdout)
        records=[]
        for level in ['calibration','refined']:
            mesh=data['meshes'][level]
            axes=[mesh['x'],mesh['v']]
            nodes=len(mesh['x'])*len(mesh['v'])
            b=torch.zeros(1,nodes,1,dtype=torch.float64)
            factor=torch.tensor([[.2,0.],[-.06,.1]],dtype=torch.float64).expand(1,nodes,2,2)
            initial=torch.tensor([math.log(100.),.045],dtype=torch.float64)
            rows=[]
            from datetime import date
            with torch.no_grad():
                for q in oracle['quotes']:
                    maturity=date.fromisoformat(q['maturity'])
                    t=(maturity-date(2025,1,1)).days/365
                    value=float(fd_latent(b,factor,initial,axes=axes,strike=q['strike'],maturity_date=maturity,
                        option_type=q['option_type'],time_steps=max(4,math.ceil(mesh['steps_per_year']*t)),mode=args.mode))
                    rows.append(dict(q,actual=value,error=value-q['price']))
            parities=[rows[i]['actual']-rows[i+1]['actual']-(100.*math.exp(-.01*((date.fromisoformat(rows[i]['maturity'])-date(2025,1,1)).days/365))
                -rows[i]['strike']*math.exp(-.03*((date.fromisoformat(rows[i]['maturity'])-date(2025,1,1)).days/365))) for i in range(0,len(rows),2)]
            records.append(dict(mesh=level,quotes=rows,max_price_error=max(abs(q['error']) for q in rows),
                                parity_residuals=parities,max_parity_error=max(map(abs,parities))))
            result['financial_checks']=records;save()
    save()
    print(json.dumps({k:v for k,v in result.items() if k in ('median_seconds','peak_growth_bytes','gradient_checks')},indent=2))


if __name__=='__main__':
    main()
