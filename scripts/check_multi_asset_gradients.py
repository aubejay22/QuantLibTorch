"""Central differences on the actual 20-asset, 30-contract neural workload."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from benchmark_multi_asset import ROOT, fingerprints
sys.path.insert(0, str(ROOT/'examples'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    from runtime_controls import worker_qos
    runtime=worker_qos('high', 8)
    import torch
    from quantlibtorch import MultiAssetBatch
    from multi_asset_neural_sde import NeuralAssets, configuration
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    configs,times,spots,targets=configuration(20,30,41,17,24)
    model=NeuralAssets([c['axes'] for c in configs], times, spots, args.device)
    plan=MultiAssetBatch(configs, device=args.device, checkpoint_steps=4)
    targets=torch.tensor(targets, dtype=torch.float64, device=args.device)
    named=dict(model.named_parameters())
    originals={name: p.detach().clone() for name,p in named.items()}
    direction={name: .05*torch.cos(torch.arange(p.numel(), dtype=p.dtype, device=p.device)).reshape_as(p)
               for name,p in named.items()}
    def loss():
        return (plan.price(*model.fields())-targets).square().mean()
    value=loss()
    gradients=dict(zip(named, torch.autograd.grad(value, tuple(named.values()))))
    result=dict(**fingerprints(), device=args.device, runtime=runtime, assets=20, contracts=30,
                nodes=[41,17], time_steps=24, loss=float(value.detach()), checks=[],
                diagnostic_source_sha256={'scripts/check_multi_asset_gradients.py':
                    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for group in ['network', 'asset_embeddings', 'initial_states']:
        names=[n for n in named if (n.startswith('net.') if group=='network' else
                                   n=='embedding' if group=='asset_embeddings' else n=='raw_initial')]
        ad=float(sum((gradients[n]*direction[n]).sum() for n in names))
        for h in [1e-3, 3e-4]:
            values=[]
            with torch.no_grad():
                for sign in [1., -1.]:
                    for n in names: named[n].copy_(originals[n]+sign*h*direction[n])
                    values.append(float(loss()))
                for n in names: named[n].copy_(originals[n])
            fd=(values[0]-values[1])/(2*h)
            result['checks'].append(dict(group=group, bump=h, ad=ad, finite_difference=fd, absolute_error=abs(ad-fd)))
            assert abs(ad-fd)<1e-7, result['checks'][-1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result['checks']))


if __name__=='__main__':
    main()
