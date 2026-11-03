"""Create independent analytic Heston quotes and pinned QuantLib mesh fixtures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/benchmarks/calibration_phase22_market.json')
    args=parser.parse_args()
    contracts=[dict(strike=k,maturity=t,scale=.05) for t in ['2025-04-01','2025-07-01','2026-01-01'] for k in [85.,100.,115.]]
    held=[dict(strike=k,maturity=t,scale=.05) for t in ['2025-05-15','2025-10-01'] for k in [90.,100.,110.]]
    meshes={n:dict(nx=x,nv=v,steps_per_year=s) for n,x,v,s in [
        ('coarse',21,9,12),('calibration',41,17,24),('refined',81,33,48),
        ('time',41,17,48),('space',81,17,24),('variance',41,33,24),
        ('diagnostic',9,5,4),('scaling',21,9,12),('three_dimensional',13,5,6)]}
    completed=subprocess.run([sys.executable,str(ROOT/'scripts/calibration_stock_reference.py')],
        input=json.dumps(dict(contracts=contracts+held,meshes=meshes)),text=True,capture_output=True,check=True)
    data=json.loads(completed.stdout)
    data.update(valuation_date='2025-01-01',spot=100.,rate=.03,dividend_yield=.01,times=[i/8 for i in range(9)],
                target_model='Independent analytic Heston, QuantLib 1.40',heston=[1.5,.04,.3,-.6,.045])
    data['heldout']=data['quotes'][9:]
    data['quotes']=data['quotes'][:9]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__': main()
