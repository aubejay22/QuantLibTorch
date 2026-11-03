"""Verify matched price/gradient evidence and summarize complete workloads."""
import argparse
import json
from pathlib import Path

from benchmark_multi_asset import fingerprints


def flatten(value):
    if isinstance(value, dict):
        return [v for k in sorted(value) for v in flatten(value[k])]
    if isinstance(value, list):
        return [v for x in value for v in flatten(x)]
    return [value]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    signature = fingerprints()
    reports, groups, pending = [], {}, []
    for path in sorted(args.input_dir.glob('*.json')):
        if path.name.startswith('commands_') or path.name.endswith('.trace.json'):
            continue
        data = json.loads(path.read_text())
        if 'configuration' not in data: continue
        if 'host_final' not in data:
            pending.append(path.name)
            continue
        assert all(data.get(k)==v for k, v in signature.items()), f'stale evidence: {path}'
        config = data['configuration']
        key = tuple(config[k] for k in ('assets', 'contracts', 'nx', 'ny', 'steps', 'latents'))
        reports.append((path, data))
        for row in data.get('samples', [])+data.get('cycles', []):
            groups.setdefault(key, []).append((path.name, row))
    comparisons, max_price, max_gradient = 0, 0., 0.
    details = []
    for key, records in groups.items():
        ref_name, ref = records[0]
        pg, gg = 0., 0.
        for name, row in records[1:]:
            for kind in ('prices', 'gradients'):
                a, b = flatten(row[kind]), flatten(ref[kind])
                assert len(a)==len(b)
                difference = max(abs(x-y) for x, y in zip(a, b))
                assert all(abs(x-y)<=1e-8+1e-9*abs(y) for x, y in zip(a, b)), (name, ref_name, kind, difference)
                if kind=='prices': pg=max(pg, difference)
                else: gg=max(gg, difference)
            comparisons += 1
        max_price=max(max_price, pg); max_gradient=max(max_gradient, gg)
        details.append(dict(configuration=key, samples=len(records), price_difference=pg, gradient_difference=gg))
    result = dict(verified_reports=len(reports), pending_reports=pending, matched_comparisons=comparisons,
                  maximum_price_difference=max_price, maximum_gradient_difference=max_gradient,
                  comparisons=details, timing=[], memory=[], profiles=[])
    for path, data in reports:
        config=data['configuration']
        common=dict(report=path.name, configuration=config)
        if 'median' in data:
            result['timing'].append(dict(**common, median=data['median'],
                inference_seconds=data['inference_median_seconds'], setup_seconds=data['setup_seconds']))
        if config['task']=='memory':
            result['memory'].append(dict(**common, baseline=data['baseline_memory'], final=data['host_final'],
                peak_host_growth=data['host_peak_growth_bytes'], cuda=data.get('cuda_final'),
                cycles=[dict(host=r['host_after_cleanup'], cuda=r.get('cuda')) for r in data['cycles']]))
        if 'events' in data:
            result['profiles'].append(dict(**common, cuda_trace_available=data['cuda_trace_available'],
                transfers=data['transfers'], events=[e for e in data['events'] if any(s in e['name'].lower()
                    for s in ('synchroniz', 'memcpy', 'local_scalar', 'aten::item', 'qlt::multi_asset'))]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('verified_reports', 'matched_comparisons', 'maximum_price_difference', 'maximum_gradient_difference')}))


if __name__ == '__main__':
    main()
