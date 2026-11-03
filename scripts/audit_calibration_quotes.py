"""Audit fitted prices with independent implied vols, bounds and put-call parity."""
from datetime import date
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def stock_ivs(request):
    import QuantLib as ql
    if ql.__version__!='1.40': raise RuntimeError('QuantLib 1.40 required')
    ref=ql.Date(1,1,2025)
    saved=ql.SavedSettings()
    ql.Settings.instance().evaluationDate=ref
    dc=ql.Actual365Fixed()
    process=ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(100.)),
        ql.YieldTermStructureHandle(ql.FlatForward(ref,.01,dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(ref,.03,dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref,ql.NullCalendar(),.2,dc)))
    output=[]
    for q in request:
        d=date.fromisoformat(q['maturity'])
        option=ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call,q['strike']),
                                ql.EuropeanExercise(ql.Date(d.day,d.month,d.year)))
        output.append(dict(q,implied_volatility=option.impliedVolatility(q['price'],process,1e-11,1000,1e-4,4.)))
    del saved
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock',action='store_true')
    parser.add_argument('--fit',type=Path,default=ROOT/'docs/benchmarks/calibration_phase22_fit.json')
    parser.add_argument('--convergence',type=Path,default=ROOT/'docs/benchmarks/calibration_phase22_convergence.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/benchmarks/calibration_phase22_quote_audit.json')
    args=parser.parse_args()
    if args.stock:
        print(json.dumps(stock_ivs(json.load(sys.stdin))));return
    # Stock QuantLib above is used only in a distinct subprocess, never linked
    # to/imported in the TorchReal pricing worker below.
    import torch
    from runtime_controls import worker_qos
    worker_qos('high' if sys.platform=='win32' else 'system',8 if sys.platform=='win32' else None)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'examples'))
    from calibrate_neural_sde import NeuralVarianceModel,restore,make_problem
    from quantlibtorch import fd_latent
    data=json.loads((ROOT/'docs/benchmarks/calibration_phase22_market.json').read_text())
    fit=json.loads(args.fit.read_text())
    convergence=json.loads(args.convergence.read_text())
    model=NeuralVarianceModel();restore(model,fit['parameters'])
    refined=next(r['prices'] for r in convergence['convergence'] if r['mesh']=='refined')
    heldout=next(r['prices'] for r in convergence['heldout'] if r['mesh']=='refined')
    targets=data['quotes']+data['heldout']
    fitted=refined+heldout
    quotes=[dict(strike=q['strike'],maturity=q['maturity'],price=q['price']) for q in targets]
    quotes += [dict(strike=q['strike'],maturity=q['maturity'],price=p) for q,p in zip(targets,fitted)]
    output=json.loads(subprocess.run([sys.executable,__file__,'--stock'],input=json.dumps(quotes),
                                     text=True,capture_output=True,check=True).stdout)
    n=len(targets)
    errors=[(b['implied_volatility']-a['implied_volatility'])*1e4 for a,b in zip(output[:n],output[n:])]
    problem=make_problem(data,'refined')
    puts=[]
    with torch.no_grad():
        fields=model(problem.time_tensor,problem.states)
        initial=torch.cat((torch.tensor([math.log(100.)],dtype=torch.float64),model.initial_latent()))
        for quote in problem.quotes:
            t=(quote.maturity-problem.valuation_date).days/365
            puts.append(float(fd_latent(*fields,initial,axes=problem.axes,times=problem.times,
                strike=quote.strike,maturity_date=quote.maturity,option_type='put',
                time_steps=max(problem.minimum_steps,math.ceil(problem.steps_per_year*t)),mode='adjoint')))
    parity=[];bound_violation=[]
    for q,call,put in zip(data['quotes'],refined,puts):
        t=(date.fromisoformat(q['maturity'])-date(2025,1,1)).days/365
        forward=100*math.exp(-.01*t)-q['strike']*math.exp(-.03*t)
        parity.append(call-put-forward)
        bound_violation.append(max(0.,max(forward,0.)-call,call-100*math.exp(-.01*t),
                                   max(-forward,0.)-put,put-q['strike']*math.exp(-.03*t)))
    monotonic=[];convex=[]
    for i in range(0,9,3):
        a,b,c=refined[i:i+3]
        monotonic.extend([max(0.,b-a),max(0.,c-b)])
        convex.append(max(0.,2*b-a-c))
    result=dict(targets=output[:n],fitted=output[n:],implied_vol_error_basis_points=errors,
        train_iv_rmse_bp=math.sqrt(sum(v*v for v in errors[:9])/9),heldout_iv_rmse_bp=math.sqrt(sum(v*v for v in errors[9:])/6),
        put_prices=puts,parity_residuals=parity,max_parity_error=max(map(abs,parity)),
        max_bound_violation=max(bound_violation),max_strike_monotonicity_violation=max(monotonic),
        max_strike_convexity_violation=max(convex),
        note='Synthetic analytic-Heston quotes; fitted model repriced on 81 x 33 grid. Implied-vol errors are absolute volatility basis points.')
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith(('max_','train_','heldout_'))},indent=2))


if __name__=='__main__': main()
