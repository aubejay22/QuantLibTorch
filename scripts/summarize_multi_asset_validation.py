"""Check CPU/CUDA refinement and actual-workload gradient evidence."""
import json
from pathlib import Path
from summarize_multi_asset import flatten

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/benchmarks'


def difference(a, b):
    a,b=flatten(a),flatten(b)
    assert len(a)==len(b)
    return max(abs(x-y) for x,y in zip(a,b))


def main():
    records={d: json.loads((OUT/f'multi_asset_phase24_convergence_{d}.json').read_text()) for d in ['cpu', 'cuda']}
    result=dict(maximum_cpu_cuda_price_difference=0., maximum_cpu_cuda_sensitivity_difference=0.,
                neural_refinement=[], heston_refinement=[], gradient_checks={})
    for family in ['heston', 'neural']:
        assert len(records['cpu'][family])==len(records['cuda'][family])==7
        for a,b in zip(records['cpu'][family], records['cuda'][family]):
            assert a['mesh']==b['mesh']
            result['maximum_cpu_cuda_price_difference']=max(result['maximum_cpu_cuda_price_difference'], difference(a['prices'], b['prices']))
            for key in (['sensitivities'] if family=='heston' else ['neural_directional', 'raw_initial_sensitivities']):
                result['maximum_cpu_cuda_sensitivity_difference']=max(result['maximum_cpu_cuda_sensitivity_difference'], difference(a[key], b[key]))
    assert result['maximum_cpu_cuda_price_difference']<1e-9
    assert result['maximum_cpu_cuda_sensitivity_difference']<1e-8
    fine=records['cuda']['neural'][-1]
    for row in records['cuda']['neural']:
        result['neural_refinement'].append(dict(mesh=row['mesh'], nodes=row['nodes'], steps=row['steps'],
            price_change_to_finest=difference(row['prices'],fine['prices']),
            neural_sensitivity_change_to_finest=difference(row['neural_directional'],fine['neural_directional']),
            initial_sensitivity_change_to_finest=difference(row['raw_initial_sensitivities'],fine['raw_initial_sensitivities'])))
    for row in records['cuda']['heston']:
        result['heston_refinement'].append({key: row[key] for key in
            ['mesh', 'max_price_error', 'max_sensitivity_error', 'max_existing_cpu_price_difference']})
    coarse, fine_h=records['cuda']['heston'][0],records['cuda']['heston'][-1]
    assert fine_h['max_price_error']<coarse['max_price_error']/20
    assert all(f<c/20 for f,c in zip(fine_h['max_sensitivity_error'],coarse['max_sensitivity_error']))
    for device in ['cpu', 'cuda']:
        data=json.loads((OUT/f'multi_asset_phase24_gradients_{device}.json').read_text())
        assert len(data['checks'])==6
        assert max(c['absolute_error'] for c in data['checks'])<1e-7
        result['gradient_checks'][device]=data['checks']
    result['qualifications']=[
        'Two independent Heston assets, three strikes each; analytic prices and central bumps from an isolated stock QuantLib 1.40 process.',
        'Frozen joint-state neural network and raw initial parameters; changes relative to finest tested grid, not certified continuous-model errors.',
        'Separate price and per-price network-direction/raw-initial sensitivity refinements; time-only refinement need not improve spatial errors.',
        'No financially converged two-latent calibration or real-market fit is claimed.']
    (OUT/'multi_asset_phase24_validation_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
