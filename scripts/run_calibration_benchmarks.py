"""Run bounded, sequential Phase 22 timing/memory/profile workers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows-qos',choices=['system','high'],default='system')
    parser.add_argument('--affinity',type=int)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/benchmarks/calibration_phase22_workloads')
    args=parser.parse_args()
    # A 9 x 5 x 5 full-AD trace exceeded 7 GB of profiler working memory.
    # Keep the two-latent event trace to one contract on 4 x 4 x 4; ordinary
    # timing and cold-memory measurements retain their original larger grids.
    market=ROOT/'docs/benchmarks/calibration_phase22_market.json'
    profile_market=ROOT/'docs/benchmarks/calibration_phase22_profile_market.json'
    data=json.loads(market.read_text())
    diagnostic=data['meshes']['diagnostic']
    data['meshes']['profile_3d']=dict(nx=4,nv=4,steps_per_year=4,
        x=[diagnostic['x'][i] for i in [0,3,5,-1]],
        v=[diagnostic['v'][i] for i in [0,1,2,-1]])
    profile_market.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    cases=[]
    for mode in ['full','adjoint']:
        for execution in ['stacked','streamed']:
            for count in [1,3,9]:
                cases.append(dict(task='timing',mesh='diagnostic',mode=mode,execution=execution,contracts=count,m=1))
            for count in [1,9]:
                cases.append(dict(task='memory',mesh='diagnostic',mode=mode,execution=execution,contracts=count,m=1))
        for m,mesh in [(1,'scaling'),(2,'three_dimensional')]:
            for count in [1,3,9]:
                cases.append(dict(task='timing',mesh=mesh,mode=mode,execution='streamed',contracts=count,m=m))
            cases.append(dict(task='memory',mesh=mesh,mode=mode,execution='streamed',contracts=9,m=m))
        cases.append(dict(task='timing',mesh='three_dimensional',mode=mode,execution='streamed',contracts=9,m=1))
        cases.append(dict(task='profile',mesh='diagnostic',mode=mode,execution='streamed',contracts=9,m=1))
        cases.append(dict(task='profile',mesh='profile_3d',mode=mode,execution='streamed',contracts=1,m=2))
    cases.append(dict(task='memory',mesh='space',mode='adjoint',execution='streamed',contracts=9,m=1))
    cases.append(dict(task='timing',mesh='space',mode='adjoint',execution='streamed',contracts=9,m=1))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for case in cases:
        key='_'.join(map(str,[case[k] for k in ['task','mesh','mode','execution','contracts','m']]))
        output=args.output_dir/(key+'.json')
        if output.exists():
            # Completed reports are resumable; incomplete reports are rerun.
            old=json.loads(output.read_text())
            if any(k in old for k in ['median_seconds','peak_growth_bytes','events']):
                continue
        command=[sys.executable,str(ROOT/'scripts/benchmark_calibration.py'),case['task'],
            '--mesh',case['mesh'],'--mode',case['mode'],'--execution',case['execution'],
            '--contracts',str(case['contracts']),'--latent-dimensions',str(case['m']),
            '--windows-qos',args.windows_qos,'--output',str(output)]
        if case['mesh']=='profile_3d':command+=['--market',str(profile_market)]
        if args.affinity is not None: command+=['--affinity',str(args.affinity)]
        print(json.dumps(dict(case=case,command=command)),flush=True)
        with (ROOT/'build'/('phase22_'+key+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(output.read_text())
        print(json.dumps({k:report[k] for k in ['median_seconds','peak_growth_bytes'] if k in report}),flush=True)


if __name__=='__main__': main()
