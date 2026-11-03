"""Asset independence, pinned engine references, device VJPs and checkpointing."""
from datetime import date
import gc
import math
import weakref

import pytest
import torch

from quantlibtorch import LatentBatch, MultiAssetBatch, fd_latent, state_grid
from test_fd_latent import joint_problem, neural_fields

DT = torch.float64
SCHEMES = ["douglas", "hundsdorfer", "craig_sneyd", "modified_craig_sneyd"]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def problem(m=1, device="cpu", scheme="hundsdorfer", checkpoint=2, count=2):
    axes, features, net, initial = joint_problem(m)
    # Distinct nonuniform spacings and curves, with identical topology/clocks.
    # Translations alone would not detect accidentally sharing derivative bases.
    configs = [dict(strikes=[93.+i, 107.+2*i],
                    axes=[[x+.01*i+.004*i*(j-2)**2 for j, x in enumerate(axes[0])]]+
                         [[y*(1.+.06*i) for y in axis] for axis in axes[1:]],
                    times=[0., .4, 1.], time_steps=3, scheme=scheme,
                    option_types=["call", "put"], rate=.02+.01*i, dividend_yield=.005+.003*i,
                    **(dict(discount_curve=[(date(2025, 1, 1), 1.),
                       (date(2025, 7, 1), .981), (date(2026, 1, 1), .953)]) if i==1 else {}))
               for i in range(count)]
    plan = MultiAssetBatch(configs, checkpoint_steps=checkpoint, device=device)
    bs, ls = [], []
    with torch.no_grad():
        for i in range(count):
            b, l = neural_fields(net, features+.03*i, m)
            bs.append(b*(1.+.1*i)); ls.append(l*(1.+.04*i))
    b, l = torch.stack(bs).to(device).requires_grad_(), torch.stack(ls).to(device).requires_grad_()
    z = torch.stack([initial.detach()+.01*i for i in range(count)]).to(device).requires_grad_()
    return plan, configs, (b, l, z)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_asset_prices_and_all_field_state_vjps_against_single_asset(device, m, scheme, mode):
    plan, configs, fields = problem(m, device, scheme)
    actual = plan.price(*fields, mode=mode)
    seed = torch.tensor([[.7, -.3], [.2, .9]], dtype=DT, device=device)
    gradients = torch.autograd.grad(actual, fields, seed)
    refs = [x.detach().cpu().requires_grad_() for x in fields]
    expected = torch.stack([LatentBatch(**config).price(*[v[i] for v in refs], mode=mode)
                            for i, config in enumerate(configs)])
    reference_grads = torch.autograd.grad(expected, refs, seed.cpu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-11, atol=2e-11)
    for a, e in zip(gradients, reference_grads):
        torch.testing.assert_close(a.cpu(), e, rtol=3e-9, atol=3e-9)
    assert plan.groups == ((0, 1),)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_original_scalar_quantlib_engine_remains_an_independent_reference(mode):
    plan, configs, fields = problem()
    value = plan.price(*fields, mode=mode)
    expected = []
    for i, config in enumerate(configs):
        config = dict(config)
        strikes, kinds = config.pop("strikes"), config.pop("option_types")
        expected.append(torch.stack([fd_latent(*[v[i] for v in fields], strike=k, option_type=kind,
                                               mode=mode, **config) for k, kind in zip(strikes, kinds)]))
    torch.testing.assert_close(value, torch.stack(expected), rtol=3e-12, atol=1e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("checkpoint", [1, 3, 8])
def test_checkpoint_lengths_and_asset_independence(device, checkpoint):
    plan, _, fields = problem(device=device, checkpoint=checkpoint)
    value = plan.price(*fields)
    seed = torch.tensor([[.2, .8], [0., 0.]], dtype=DT, device=device)
    gradients = torch.autograd.grad(value, fields, seed)
    for g in gradients:
        assert torch.count_nonzero(g[1]) == 0
    refs = [x.detach().requires_grad_() for x in fields]
    full = plan.price(*refs, mode="full")
    ref_grad = torch.autograd.grad(full, refs, seed)
    torch.testing.assert_close(value, full, rtol=0, atol=0)
    for a, b in zip(gradients, ref_grad):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_strided_field_vjps_against_independent_central_differences(device, mode):
    plan, _, fields = problem(device=device)
    storage = [torch.stack((v.detach(), v.detach()), -1).requires_grad_() for v in fields]
    views = [v[..., 0] for v in storage]
    before = [v.detach().clone() for v in storage]
    seed = torch.tensor([[.3, .4], [-.1, .7]], dtype=DT, device=device)
    def loss(inputs):
        return (plan.price(*inputs, mode=mode)*seed).sum()
    grads = torch.autograd.grad(loss(views), views)
    directions = [torch.cos(torch.arange(v.numel(), dtype=DT, device=device)).reshape_as(v)*.01 for v in views]
    for selected in range(3):
        h = 1e-5
        with torch.no_grad():
            plus, minus = list(views), list(views)
            plus[selected] = views[selected]+h*directions[selected]
            minus[selected] = views[selected]-h*directions[selected]
            difference = (loss(plus)-loss(minus))/(2*h)
        torch.testing.assert_close((grads[selected]*directions[selected]).sum(), difference, rtol=2e-6, atol=3e-8)
    for a, b in zip(storage, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_shared_neural_parameters_distinct_initials_and_training(device):
    plan, _, _ = problem(device=device)
    axes, features, net, _ = joint_problem()
    net = net.to(device)
    features = torch.stack((features, features+.03)).to(device)
    initial = torch.tensor([[math.log(100.), .03], [math.log(102.), -.02]], dtype=DT,
                           device=device, requires_grad=True)
    parameters = (*net.parameters(), initial)
    values, grads = [], []
    for mode in ("full", "adjoint"):
        raw = net(features)
        b = .08*raw[..., :1].tanh()
        l = .15*torch.eye(2, dtype=DT, device=device)+.08*raw[..., 1:].reshape(*raw.shape[:-1], 2, 2).tanh()
        price = plan.price(b, l, initial, mode=mode)
        loss = (price-torch.tensor([[10., 9.], [9., 10.]], device=device)).square().mean()
        values.append(loss.detach()); grads.append(torch.autograd.grad(loss, parameters))
    torch.testing.assert_close(values[0], values[1], rtol=2e-12, atol=2e-12)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=3e-10)
    assert grads[1][-1][:, 1].abs().min() > 1e-5
    # A concrete optimization step uses the network and both latent initials.
    optimizer = torch.optim.SGD(parameters, lr=1e-5)
    for p, g in zip(parameters, grads[1]):
        p.grad = g
    optimizer.step()
    raw = net(features)
    b = .08*raw[..., :1].tanh()
    l = .15*torch.eye(2, dtype=DT, device=device)+.08*raw[..., 1:].reshape(*raw.shape[:-1], 2, 2).tanh()
    after = (plan.price(b, l, initial)-torch.tensor([[10., 9.], [9., 10.]], device=device)).square().mean()
    assert after < values[1]


def dense_natural_spline(x, y, z):
    """Independent second-derivative linear system, axis at y[-2]."""
    n = len(x); h = x[1:]-x[:-1]
    matrix = torch.zeros((n, n), dtype=y.dtype, device=y.device)
    matrix[0, 0] = matrix[-1, -1] = 1.
    for i in range(1, n-1):
        matrix[i, i-1] = h[i-1]; matrix[i, i] = 2*(h[i-1]+h[i]); matrix[i, i+1] = h[i]
    slope = (y[..., 1:, :]-y[..., :-1, :])/h[:, None]
    rhs = torch.cat((torch.zeros_like(y[..., :1, :]), 6*(slope[..., 1:, :]-slope[..., :-1, :]),
                     torch.zeros_like(y[..., :1, :])), -2)
    second = torch.linalg.solve(matrix, rhs)
    i = int(torch.searchsorted(x, z.detach(), right=True).clamp(1, n-1))-1
    left, right = (x[i+1]-z)/h[i], (z-x[i])/h[i]
    return left*y[..., i, :]+right*y[..., i+1, :]+h[i].square()/6*(
        (left**3-left)*second[..., i, :]+(right**3-right)*second[..., i+1, :])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("knots", [False, True])
def test_device_spline_against_independent_dense_natural_system_with_curvature(device, m, knots):
    plan, configs, fields = problem(m=m, device=device)
    z = fields[-1]
    if knots:
        z = torch.tensor([[a[2] for a in p.axes] for p in plan.plans], dtype=DT, device=device, requires_grad=True)
    n = math.prod(map(len, plan.plans[0].axes))
    u = torch.sin(torch.arange(2*n*2, dtype=DT, device=device)*.2).reshape(2, n, 2).requires_grad_()
    actual = plan._native[0].readout(u, z)
    expected = []
    for a, config in enumerate(configs):
        value = u[a]
        for i, axis in enumerate(config['axes']):
            value = dense_natural_spline(torch.tensor(axis, dtype=DT, device=device),
                                         value.reshape(-1, len(axis), 2), z[a, i])
        expected.append(value.reshape(2))
    expected = torch.stack(expected)
    ga = torch.autograd.grad(actual.square().sum(), (u, z), create_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (u, z), create_graph=True)
    ha = torch.autograd.grad(ga[1].sum(), z)[0]
    he = torch.autograd.grad(ge[1].sum(), z)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    for a, e in zip((*ga, ha), (*ge, he)):
        torch.testing.assert_close(a, e, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_full_price_curvature_and_adjoint_higher_order_rejection(device):
    plan, _, fields = problem(device=device)
    def evaluate(p, mode):
        return plan.price(fields[0]*p, fields[1]*(.8+.2*p), fields[2], mode=mode).square().sum()
    p = torch.tensor(1., dtype=DT, device=device, requires_grad=True)
    first = torch.autograd.grad(evaluate(p, "full"), p, create_graph=True)[0]
    second = torch.autograd.grad(first, p)[0]
    def first_at(v):
        t = torch.tensor(v, dtype=DT, device=device, requires_grad=True)
        return torch.autograd.grad(evaluate(t, "full"), t)[0]
    diff = (first_at(1.+1e-4)-first_at(1.-1e-4))/2e-4
    torch.testing.assert_close(second, diff, rtol=2e-6, atol=2e-6)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(evaluate(p, "adjoint"), p, create_graph=True)


def test_incompatible_asset_clocks_and_topologies_execute_separately_in_order():
    _, configs, fields = problem(count=3)
    configs[1]['time_steps'] = 4
    plan = MultiAssetBatch(configs)
    assert plan.groups == ((0, 2), (1,))
    actual = plan.price(*fields)
    expected = torch.stack([LatentBatch(**c).price(*[v[i] for v in fields]) for i, c in enumerate(configs)])
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-11)
    # Different node counts are supplied as per-asset lists.
    configs[1]['axes'][0] = configs[1]['axes'][0]+[math.log(230.)]
    plan = MultiAssetBatch(configs)
    data = []
    for i, config in enumerate(configs):
        n = math.prod(map(len, config['axes']))
        data.append((torch.zeros((3, n, 1), dtype=DT),
                     torch.eye(2, dtype=DT).expand(3, n, 2, 2)*.2, fields[2][i]))
    actual = plan.price(*map(list, zip(*data)))
    expected = torch.stack([LatentBatch(**c).price(*v) for c, v in zip(configs, data)])
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-11)


@pytest.mark.parametrize("field,value", [("cash_dividends", [(date(2025, 7, 1), -1.)]),
    ("exercise", "invalid"), ("exercise_dates", [date(2025, 7, 1)]),
    ("damping_steps", 1), ("boundary_conditions", ["dirichlet"])])
def test_unsupported_events_and_boundaries_rejected(field, value):
    _, configs, _ = problem()
    configs[1][field] = value
    with pytest.raises(ValueError):
        MultiAssetBatch(configs)


def test_contract_compatibility_validation_and_copied_configuration():
    plan, configs, _ = problem()
    rows = [[dict(strike=k, **{a: v for a, v in c.items() if a not in ('strikes', 'option_types')})
             for k in c['strikes']] for c in configs]
    assert MultiAssetBatch.from_contracts(rows).groups == ((0, 1),)
    rows[0][1]['rate'] = .09
    with pytest.raises(ValueError, match="incompatible"):
        MultiAssetBatch.from_contracts(rows)
    configs[0]['axes'][0][0] = -1000.
    assert plan.plans[0].axes[0][0] != -1000.
    with pytest.raises(ValueError, match="empty"):
        MultiAssetBatch([])
    with pytest.raises(ValueError, match="equal contract"):
        MultiAssetBatch([dict(configs[0]), dict(configs[1], strikes=[100.], option_types=['call'])])
    with pytest.raises(ValueError):
        MultiAssetBatch(configs, checkpoint_steps=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", ["nan", "overflow", "state", "shape", "dtype", "dominance"])
def test_input_validation_and_failed_call_recovery(device, bad):
    if bad == 'dominance' and device == 'cpu':
        pytest.skip('CPU Thomas is not restricted to strict dominance')
    plan, _, fields = problem(device=device)
    b, l, z = fields
    if bad == 'nan': b = b*float('nan')
    if bad == 'overflow': l = l*1e300
    if bad == 'state': z = z+100.
    if bad == 'shape': b = b[..., :0]
    if bad == 'dtype': l = l.float()
    if bad == 'dominance': b = b*0.+1000.
    with pytest.raises((ValueError, RuntimeError)):
        plan.price(b, l, z)
    assert torch.isfinite(plan.price(*fields)).all()
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(plan.price(*fields).sum(), fields))


def test_repeated_fresh_graphs_do_not_stick_to_plan():
    plan, _, fields = problem()
    refs = []
    for _ in range(3):
        inputs = [v.detach().clone().requires_grad_() for v in fields]
        value = plan.price(*inputs)
        refs.extend(weakref.ref(v) for v in (*inputs, value))
        value.sum().backward()
        del value, inputs
        gc.collect()
        assert all(r() is None for r in refs)
