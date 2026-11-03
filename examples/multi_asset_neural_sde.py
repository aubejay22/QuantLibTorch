"""Train a shared, jointly state-dependent neural SDE over independent assets.

Run: python examples/multi_asset_neural_sde.py --device cuda --assets 20 --contracts 30
The deliberately small default grid demonstrates throughput, not convergence.
"""
from __future__ import annotations
import argparse
import math

import torch
from quantlibtorch import MultiAssetBatch, state_grid


class NeuralAssets(torch.nn.Module):
    def __init__(self, axes, times, spots, device="cpu"):
        super().__init__()
        torch.manual_seed(241)
        m = len(axes[0])-1
        self.latents = m
        self.net = torch.nn.Sequential(torch.nn.Linear(m+4, 8), torch.nn.Tanh(),
                                       torch.nn.Linear(8, m+(m+1)**2)).double()
        self.embedding = torch.nn.Parameter(.05*torch.cos(torch.arange(len(axes)*2, dtype=torch.float64)).reshape(-1, 2))
        self.raw_initial = torch.nn.Parameter(.12*torch.sin(torch.arange(len(axes)*m, dtype=torch.float64)).reshape(-1, m))
        self.register_buffer("log_spots", torch.tensor(spots, dtype=torch.float64).log())
        states = torch.stack([state_grid(a) for a in axes])
        states = torch.cat((states[..., :1]-self.log_spots[:, None, None], states[..., 1:]), -1)
        clock = torch.tensor(times, dtype=torch.float64)[None, :, None, None].expand(len(axes), -1, states.shape[1], -1)
        self.register_buffer("features", torch.cat((clock, states[:, None].expand(-1, len(times), -1, -1)), -1))
        self.to(device)

    def fields(self, indices=None):
        features = self.features if indices is None else self.features[indices]
        embedding = self.embedding if indices is None else self.embedding[indices]
        raw_initial = self.raw_initial if indices is None else self.raw_initial[indices]
        log_spots = self.log_spots if indices is None else self.log_spots[indices]
        embed = embedding[:, None, None].expand(*features.shape[:-1], 2)
        raw = self.net(torch.cat((features, embed), -1))
        m, d = self.latents, self.latents+1
        drift = -.1*features[..., 2:2+m]+.08*raw[..., :m].tanh()
        loading = .18*torch.eye(d, dtype=raw.dtype, device=raw.device)+.035*raw[..., m:].reshape(*raw.shape[:-1], d, d).tanh()
        initial = torch.cat((log_spots[:, None], .3*raw_initial.tanh()), -1)
        return drift, loading, initial


def configuration(assets, contracts, nx=21, ny=9, steps=12, latents=1):
    times = [0., .5, 1.]
    spots = [90.+20.*i/max(1, assets-1) for i in range(assets)]
    configs, targets = [], []
    normal = lambda x: .5*(1.+math.erf(x/math.sqrt(2.)))
    for i, spot in enumerate(spots):
        x = torch.linspace(math.log(spot*.35), math.log(spot*2.8), nx, dtype=torch.float64).tolist()
        y = torch.linspace(-1., 1., ny, dtype=torch.float64).tolist()
        strikes = [spot*(.8+.4*j/max(1, contracts-1)) for j in range(contracts)] if contracts>1 else [spot]
        r, q = .02+.015*i/max(1, assets-1), .005+.005*i/max(1, assets-1)
        configs.append(dict(axes=[x]+[y]*latents, times=times, strikes=strikes, time_steps=steps,
                            rate=r, dividend_yield=q))
        # Analytic Black-Scholes synthetic targets, separate from PDE pricing.
        sigma = .2+.01*i/max(1, assets-1)
        row = []
        for strike in strikes:
            d1 = (math.log(spot/strike)+r-q+.5*sigma*sigma)/sigma
            row.append(spot*math.exp(-q)*normal(d1)-strike*math.exp(-r)*normal(d1-sigma))
        targets.append(row)
    return configs, times, spots, targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--assets', type=int, default=8)
    p.add_argument('--contracts', type=int, default=16)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--checkpoint-steps', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(1)
    configs, times, spots, targets = configuration(args.assets, args.contracts)
    plan = MultiAssetBatch(configs, device=args.device, checkpoint_steps=args.checkpoint_steps)
    model = NeuralAssets([c['axes'] for c in configs], times, spots, args.device)
    targets = torch.tensor(targets, dtype=torch.float64, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for step in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        prices = plan.price(*model.fields())
        loss = (prices-targets).square().mean()
        loss.backward()
        optimizer.step()
        print(f"iteration={step} price_rmse={loss.detach().sqrt().item():.8f}")


if __name__ == '__main__':
    main()
