"""Verify matched Phase 23 workloads and summarize measured timings/memory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def flatten(value):
    if isinstance(value, dict):
        return [x for k in sorted(value) for x in flatten(value[k])]
    if isinstance(value, list):
        return [x for v in value for x in flatten(v)]
    return [float(value)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT/'docs/benchmarks/packed_phase23_workloads')
    p.add_argument('--output', type=Path, default=ROOT/'docs/benchmarks/packed_phase23_summary.json')
    args = p.parse_args()
    records = []
    for path in sorted(args.input_dir.glob('*.json')):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or 'config' not in record:
            continue
        for source, expected in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == expected, (path, source)
        record['path'] = str(path.relative_to(ROOT))
        records.append(record)
    assert records
    assert len({r['binary_sha256'] for r in records}) == 1
    groups = {}
    result = dict(binary_sha256=records[0]['binary_sha256'], timings=[], memory=[], profiles=[],
                  maximum_price_difference=0., maximum_gradient_difference=0., verified_reports=len(records))
    for r in records:
        c = r['config']
        key = (c['mesh'], c['contracts'], c['maturities'], c['mode'], c['latent_dimensions'])
        if c['task'] == 'timing':
            groups.setdefault(key, {})[(c['execution'], c['device'])] = r
        elif c['task'] == 'memory':
            result['memory'].append(dict(configuration=key, execution=c['execution'], device=c['device'],
                peak_growth_bytes=r['peak_growth_bytes'], setup_seconds=r['setup_seconds'],
                cleanup_rss_bytes=[a['after_cleanup']['rss_bytes'] for a in r['cycles']],
                cleanup_private_bytes=[a['after_cleanup'].get('private_bytes') for a in r['cycles']],
                cuda=[a.get('cuda') for a in r['cycles']], path=r['path']))
        elif c['task'] == 'profile':
            result['profiles'].append(dict(configuration=key, execution=c['execution'], device=c['device'],
                aten_calls=sum(e['count'] for e in r['events'] if e['name'].startswith('aten::')),
                scopes=[e for e in r['events'] if e['name'].startswith(('qlt::', 'calibration::'))], path=r['path']))
    for key, group in groups.items():
        ref = group.get(('serial', 'cpu'), group.get(('batched', 'cpu')))
        assert ref is not None, key
        comparison = dict(configuration=key, paths={}, median={}, setup_seconds={},
                          maximum_price_difference=0., maximum_gradient_difference=0.)
        base = ref['samples'][0]
        for (execution, device), r in group.items():
            name = execution+'_'+device
            comparison['paths'][name] = r['path']
            comparison['median'][name] = r['median']
            comparison['setup_seconds'][name] = r['setup_seconds']
            for sample in r['samples']:
                for field, tolerance, relative in [('prices', 3e-10, 3e-11), ('gradients', 2e-8, 3e-9)]:
                    actual, expected = flatten(sample[field]), flatten(base[field])
                    assert len(actual) == len(expected)
                    error = max(abs(a-b) for a, b in zip(actual, expected))
                    assert all(abs(a-b) <= tolerance+relative*abs(b) for a, b in zip(actual, expected)), (key, name, field, error)
                    target = 'maximum_price_difference' if field == 'prices' else 'maximum_gradient_difference'
                    comparison[target] = max(comparison[target], error)
                    result[target] = max(result[target], error)
        if ('serial', 'cpu') in group and ('batched', 'cpu') in group:
            comparison['serial_over_batched_cpu'] = group[('serial', 'cpu')]['median']['seconds']/group[('batched', 'cpu')]['median']['seconds']
        if ('batched', 'cuda') in group:
            comparison['cuda_over_batched_cpu'] = group[('batched', 'cuda')]['median']['seconds']/group[('batched', 'cpu')]['median']['seconds']
        result['timings'].append(comparison)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('verified_reports', 'maximum_price_difference', 'maximum_gradient_difference')}))
    for row in result['timings']:
        print(row['configuration'], {k: round(v['seconds'], 6) for k, v in row['median'].items()},
              row.get('serial_over_batched_cpu'), row.get('cuda_over_batched_cpu'))


if __name__ == '__main__':
    main()
