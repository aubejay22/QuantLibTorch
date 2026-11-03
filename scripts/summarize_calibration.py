"""Validate recorded calibration results and summarize matched CPU workloads."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'docs/benchmarks'


def flat(value):
    if isinstance(value,dict): return [x for key in sorted(value) for x in flat(value[key])]
    if isinstance(value,list): return [x for item in value for x in flat(item)]
    return [value]


def main():
    read=lambda name: json.loads((REPORTS/('calibration_phase22_'+name+'.json')).read_text())
    fit=read('fit')['fit']; con=read('convergence'); audit=read('quote_audit')
    grid={r['mesh']:r for r in con['convergence']}
    difference=lambda a,b: max(abs(x-y) for x,y in zip(a,b))
    assert fit['final_rmse'] < .01 and fit['final_latent_state'] != fit['history'][0]['latent_state']
    assert grid['refined']['max_target_error'] < .02
    assert difference(fit['final_prices'],grid['space']['prices']) < 2e-11
    assert con['heldout'][-1]['rmse'] < .01
    assert audit['max_parity_error'] < .01
    assert max(audit[k] for k in ['max_bound_violation','max_strike_monotonicity_violation','max_strike_convexity_violation'])==0
    summary=dict(initial_rmse=fit['history'][0]['price_rmse'],training_rmse=fit['final_rmse'],
        refined_training_rmse=grid['refined']['rmse'],refined_heldout_rmse=con['heldout'][-1]['rmse'],
        elapsed_training_seconds=fit['elapsed_seconds'],optimizer_evaluations=len(fit['history']),
        final_latent_state=fit['final_latent_state'],last_scaled_objective_gradient_norm=fit['history'][-1]['gradient_norm'],
        max_training_grid_refinement_change=difference(grid['space']['prices'],grid['refined']['prices']),
        max_domain_change=difference(grid['domain']['prices'],grid['calibration']['prices']),
        max_coefficient_time_change=difference(grid['coefficient_times']['prices'],grid['calibration']['prices']),
        quote_audit={k:v for k,v in audit.items() if k.startswith(('max_','train_','heldout_'))})
    refs={};comparisons=0;max_price=max_gradient=0.
    timings=[];memory=[];profiles=[]
    for path in sorted((REPORTS/'calibration_phase22_workloads').glob('*.json')):
        record=json.loads(path.read_text());c=record['configuration']
        samples=record.get('warmups',[])+record.get('samples',[])
        if 'evaluation' in record:samples.append(record['evaluation'])
        key=(c['mesh'],c['latent_dimensions'],c['contracts'])
        for sample in samples:
            if key not in refs: refs[key]=sample
            reference=refs[key]
            dp=difference(sample['prices'],reference['prices'])
            a,b=flat(sample['gradients']),flat(reference['gradients'])
            assert len(a)==len(b)
            dg=max(abs(x-y) for x,y in zip(a,b))
            assert dp<2e-10
            assert all(math.isfinite(x) and abs(x-y)<2e-8*(1+abs(y)) for x,y in zip(a,b)),path
            max_price=max(max_price,dp);max_gradient=max(max_gradient,dg);comparisons+=len(a)
        label={k:c[k] for k in ['mesh','mode','execution','contracts','latent_dimensions']}
        label.update(nodes=record['nodes'],source=path.name)
        if 'median_seconds' in record:timings.append(dict(label,seconds=record['median_seconds']))
        if 'peak_growth_bytes' in record:memory.append(dict(label,peak_growth_mb=record['peak_growth_bytes']/1e6,
            absolute_peak_mb=record['after']['peak_rss_bytes']/1e6))
        if 'events' in record:
            scopes=[e for e in record['events'] if e['name'].startswith(('calibration::','qlt::'))]
            backward=[e for e in record['events'] if 'Backward' in e['name'] and not e['name'].startswith('autograd::engine')]
            aten=[e for e in record['events'] if e['name'].startswith('aten::')]
            profiles.append(dict(label,aten_calls=sum(e['calls'] for e in aten),scopes=scopes,
                                  top_aten=aten[:12],top_backward=backward[:10]))
    assert len(timings)==27 and len(memory)==13 and len(profiles)==4,(len(timings),len(memory),len(profiles))
    summary.update(timings=timings,memory=memory,profiles=profiles,gradient_comparisons=comparisons,
                   max_price_difference=max_price,max_gradient_difference=max_gradient)
    full,adj=read('gradients_full'),read('gradients_adjoint')
    summary['matched_calibration_gradient_difference']=difference(flat(full['evaluation']['gradients']),flat(adj['evaluation']['gradients']))
    assert summary['matched_calibration_gradient_difference'] < 1e-8
    checks=[]
    for name in ['gradients_full','gradients_adjoint','gradients_refined']:
        r=read(name)
        for c in r['gradient_checks']:
            assert c['absolute_error'] < 2e-5*(1+abs(c['ad'])),(name,c)
            checks.append(dict(report=name,**c))
    summary['independent_gradient_checks']=checks
    deep=read('refinement')
    fine={row['heldout']:row for row in deep['results'] if row['mesh']=='fine'}
    assert max(row['rmse'] for row in fine.values()) < .05
    assert deep['heston_control']['max_target_error'] < .01
    summary['further_refinement']=dict(training_rmse=fine[False]['rmse'],heldout_rmse=fine[True]['rmse'],
        training_max_target_error=fine[False]['max_target_error'],heldout_max_target_error=fine[True]['max_target_error'],
        max_change_from_81_by_33=difference(fine[False]['prices'],grid['refined']['prices']),
        max_change_from_training_grid=difference(fine[False]['prices'],grid['space']['prices']),
        heston_control=deep['heston_control'])
    python_profile=read('python_profile')
    assert difference(python_profile['evaluation']['prices'],fit['final_prices']) < 2e-11
    assert difference(flat(python_profile['evaluation']['gradients']),flat(read('gradients_refined')['evaluation']['gradients'])) < 1e-9
    summary['calibration_grid_python_profile']=dict(total_seconds=python_profile['total_seconds'],events=python_profile['events'][:20])
    (REPORTS/'calibration_phase22_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('timings','memory','profiles','independent_gradient_checks','calibration_grid_python_profile')},indent=2))


if __name__=='__main__':main()
