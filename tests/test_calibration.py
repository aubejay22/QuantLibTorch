from datetime import date
import math

import pytest
import torch

from quantlibtorch import LatentCalibration, OptionQuote, fd_latent, state_grid
from test_latent_pde import spatial


DT = torch.float64


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([.1, -.2, .25, .15], dtype=DT))
        self.z = torch.nn.Parameter(torch.tensor(.05, dtype=DT))

    def initial(self):
        return self.z.reshape(1)

    def forward(self, times, states):
        t = times[:, None]
        x, y = states[:, 0]-math.log(100.), states[:, 1]
        drift = .04*torch.tanh(self.p[0]+t+x*y)[..., None]
        vol = .18+.04*torch.tanh(self.p[1]+t*.1+x*.2+y*.3)
        a = .13+.02*torch.tanh(self.p[2]+t*.2+y-x*.1)
        rho = .6*torch.tanh(self.p[3])
        zero = torch.zeros_like(vol)
        return drift, torch.stack((torch.stack((vol, zero), -1),
            torch.stack((a*rho, a*torch.sqrt(1-rho*rho)), -1)), -2)


def problem():
    return LatentCalibration([
        OptionQuote(95., date(2025, 4, 1), 8.1, .7),
        OptionQuote(105., date(2025, 7, 1), 4.2, 1.3),
        OptionQuote(100., date(2026, 1, 1), 8.8, .9)],
        axes=[[math.log(s) for s in [40., 70., 95., 110., 140., 200.]], [-1., -.3, .3, 1.]],
        times=[0., .3, 1.], steps_per_year=3, minimum_steps=2)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_streamed_objective_matches_single_graph_and_independent_parameter_and_state_bumps(mode):
    p, model = problem(), TinyModel()
    loss, prices = p.loss(model, model.initial, mode=mode)
    expected = torch.autograd.grad(loss, (model.p, model.z))
    loss2, prices2 = p.backward(model, model.initial, mode=mode)
    torch.testing.assert_close(loss, loss2, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(prices, prices2, rtol=0, atol=0)
    for actual, ref in zip((model.p.grad, model.z.grad), expected):
        torch.testing.assert_close(actual, ref, rtol=2e-11, atol=2e-11)
    h = 2e-5
    with torch.no_grad():
        baseline = model.p.clone()
        direction = torch.tensor([.3, -.2, .4, .1], dtype=DT)
        model.p.copy_(baseline+h*direction)
        plus = p.loss(model, model.initial, mode=mode)[0]
        model.p.copy_(baseline-h*direction)
        minus = p.loss(model, model.initial, mode=mode)[0]
        model.p.copy_(baseline)
        initial = model.z.clone()
        model.z.copy_(initial+h)
        zplus = p.loss(model, model.initial, mode=mode)[0]
        model.z.copy_(initial-h)
        zminus = p.loss(model, model.initial, mode=mode)[0]
        model.z.copy_(initial)
    torch.testing.assert_close((expected[0]*direction).sum(), (plus-minus)/(2*h), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(expected[1], (zplus-zminus)/(2*h), rtol=2e-6, atol=2e-7)


def test_streamed_gradients_accumulate_and_common_absolute_time_is_used():
    p, model = problem(), TinyModel()
    loss, prices = p.backward(model, model.initial)
    first = model.p.grad.clone()
    p.backward(model, model.initial)
    torch.testing.assert_close(model.p.grad, 2*first, rtol=2e-13, atol=2e-13)
    with torch.no_grad():
        b, l = model(p.time_tensor, p.states)
        initial = torch.cat((torch.tensor([math.log(100.)], dtype=DT), model.initial()))
        ref = torch.stack([fd_latent(b, l, initial, axes=p.axes, times=p.times,
            strike=q.strike, maturity_date=q.maturity,
            time_steps=max(2, math.ceil(3*(q.maturity-p.valuation_date).days/365)), mode="adjoint") for q in p.quotes])
    torch.testing.assert_close(prices, ref, rtol=0, atol=0)


def test_full_objective_preserves_second_derivatives():
    p, model = problem(), TinyModel()
    loss, _ = p.loss(model, model.initial, mode="full")
    g, = torch.autograd.grad(loss, model.z, create_graph=True)
    h, = torch.autograd.grad(g, model.z)
    assert torch.isfinite(h) and abs(float(h)) > 1e-6


@pytest.mark.parametrize("d", [2, 3])
def test_actual_equity_generator_ito_covariance_discount_and_boundaries(d):
    axes = [[3.5, 4.2, 4.7, 5.4], [-.6, -.1, .5], [-.4, .3]][:d]
    x = state_grid(axes)
    n = len(x)
    b = .03*torch.sin(x[:, 1:])[None].expand(2, n, d-1)
    factor = (.2*torch.eye(d, dtype=DT)[None, None]+
              .02*torch.sin(x[None, :, :, None]+torch.tensor([0., .4], dtype=DT)[:, None, None, None]))
    r, q = .037, .014
    t1, t2 = .05, .25  # covariance midpoint, between samples at 0 and 1
    c = (factor@factor.transpose(-1,-2))
    c = .85*c[0]+.15*c[1]
    drift = .85*b[0]+.15*b[1]
    edge = (x[:, 0] == axes[0][0]) | (x[:, 0] == axes[0][-1])
    stock_drift = r-q-.5*c[:, 0, 0]*(~edge)
    full_drift = torch.cat((stock_drift[:, None], drift), -1)
    first = [spatial(axes, i, 1) for i in range(d)]
    second = [spatial(axes, i, 2) for i in range(d)]
    matrix = sum(full_drift[:, i, None]*first[i]+.5*c[:, i, i, None]*second[i] for i in range(d))
    matrix = matrix+sum((c[:, i, j, None]*(first[i]@first[j]) for i in range(d) for j in range(i+1,d)), torch.zeros(n,n,dtype=DT))-r*torch.eye(n,dtype=DT)
    def native(u):
        return torch.ops.quantlibtorch._equity_generator_probe(b, factor, u, axes, [0., 1.], t1, t2, r, q)
    u = torch.cos(x.sum(-1))+x[:, 0]*x[:, -1]
    torch.testing.assert_close(native(u), matrix@u, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(native(torch.ones(n,dtype=DT)), torch.full((n,),-r,dtype=DT), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(native(x[:, 0]), stock_drift-r*x[:, 0], rtol=2e-12, atol=2e-12)
    # Removing Ito would differ by C00/2 on every interior stock row.
    wrong = r-q-r*x[:, 0]
    assert (native(x[:, 0])[~edge]-wrong[~edge]).abs().min() > .01
    assert torch.linalg.eigvalsh(c).min() > 0


def test_log_price_generator_converges_to_discounted_stock_carry():
    errors = []
    for nx in [11, 21, 41]:
        axes = [torch.linspace(3.8, 5.4, nx, dtype=DT).tolist(), [-1., -.3, .3, 1.]]
        x = state_grid(axes)
        b = torch.zeros(1,len(x),1,dtype=DT)
        l = torch.tensor([[.2,0.],[-.08,.15]],dtype=DT).expand(1,len(x),2,2)
        value = torch.ops.quantlibtorch._equity_generator_probe(b,l,x[:,0].exp(),axes,[0.],0.,.1,.03,.01)
        interior = (x[:,0]>axes[0][0]) & (x[:,0]<axes[0][-1])
        error = ((value+.01*x[:,0].exp())/x[:,0].exp())[interior].abs().max()
        errors.append(float(error))
    assert errors[-1] < errors[0]/12


@pytest.mark.parametrize("kwargs", [dict(scale=0.),dict(price=-1.),dict(strike=0.),dict(option_type="straddle")])
def test_quote_validation(kwargs):
    values = dict(strike=100., maturity=date(2026,1,1),price=8.)
    values.update(kwargs)
    with pytest.raises(ValueError): OptionQuote(**values)
