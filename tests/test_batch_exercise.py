"""Batched financial events against original QuantLib engines and bump VJPs."""
from datetime import date
import math

import pytest
import torch

from quantlibtorch import LatentBatch, MultiAssetBatch, fd_latent
from test_fd_latent import joint_problem, neural_fields, heston_fields, AXES, PARAMS, stock

DT = torch.float64
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]
SCHEMES = ["douglas", "hundsdorfer", "craig_sneyd", "modified_craig_sneyd"]


def exercise_config(kind):
    return dict(exercise=kind, exercise_dates=[date(2025, 5, 1), date(2025, 9, 1),
        date(2026, 1, 1)] if kind == "bermudan" else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("kind", ["american", "bermudan"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_exercise_prices_neural_and_state_vjps_match_original(device, scheme, kind, mode):
    axes, features, net, initial = joint_problem()
    config = dict(axes=axes, times=[0., .4, 1.], time_steps=3, scheme=scheme,
                  discount_curve=[(date(2025, 1, 1), 1.), (date(2025, 7, 1), .98),
                                  (date(2026, 1, 1), .945)], **exercise_config(kind))
    parameters = (*net.parameters(), initial)
    weights = torch.tensor([.7, -.3], dtype=DT)
    b, l = neural_fields(net, features, 1)
    expected = torch.stack([fd_latent(b, l, initial, strike=k, option_type=t, mode=mode, **config)
                            for k, t in [(93., "call"), (107., "put")]])
    expected_grads = torch.autograd.grad(expected, parameters, weights)
    for multi in (False, True):
        single = LatentBatch([93., 107.], option_types=["call", "put"], device=device, **config)
        plan = MultiAssetBatch([single], checkpoint_steps=2) if multi else single
        b, l = neural_fields(net, features, 1)
        inputs = [b.to(device), l.to(device), initial.to(device)]
        if multi:
            inputs = [v.unsqueeze(0) for v in inputs]
        value = plan.price(*inputs, mode=mode).reshape(-1)
        grads = torch.autograd.grad(value, parameters, weights.to(device))
        torch.testing.assert_close(value.cpu(), expected, rtol=3e-11, atol=3e-11)
        for actual, reference in zip(grads, expected_grads):
            torch.testing.assert_close(actual, reference, rtol=4e-9, atol=4e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["american", "bermudan"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_two_latent_exercise_preserves_original_solver(device, kind, mode):
    axes, features, net, initial = joint_problem(2)
    config = dict(axes=axes, times=[0., .4, 1.], time_steps=2, **exercise_config(kind))
    b, l = neural_fields(net, features, 2)
    reference = fd_latent(b, l, initial, strike=103., option_type="put", mode=mode, **config)
    parameters = (*net.parameters(), initial)
    reference_grads = torch.autograd.grad(reference, parameters)
    b, l = neural_fields(net, features, 2)
    plan = MultiAssetBatch([dict(strikes=[103.], option_types=["put"], **config)], device=device,
                           checkpoint_steps=1)
    actual = plan.price(b[None].to(device), l[None].to(device), initial[None].to(device), mode=mode).sum()
    grads = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual.cpu(), reference, rtol=3e-11, atol=3e-11)
    for a, b in zip(grads, reference_grads):
        torch.testing.assert_close(a, b, rtol=4e-9, atol=4e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["european", "american", "bermudan"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_dividends_and_exercise_against_stock_quantlib_and_all_parameter_bumps(device, kind, mode):
    # Exercise and a dividend coincide; maturity also has both events. The
    # first cash date exercises the solver's earliest-event snapshot rule.
    cash = [(date(2025, 1, 2), .3), (date(2025, 5, 1), 1.2), (date(2026, 1, 1), .6)]
    config = dict(axes=AXES, time_steps=5, option_types=["put"], cash_dividends=cash,
                  **exercise_config(kind))
    p = torch.tensor(PARAMS, dtype=DT, device=device, requires_grad=True)
    b, l = heston_fields(p.cpu())
    plan = MultiAssetBatch([dict(strikes=[103.], **config)], device=device, checkpoint_steps=3)
    result = plan.price(b[None].to(device), l[None].to(device), p[None, 4:], mode=mode).sum()
    grads, = torch.autograd.grad(result, p)
    oracle = stock(dict(axes=AXES, parameters=PARAMS, strike=103., option_type="put", exercise=kind,
        exercise_dates=list(map(str, config['exercise_dates'])),
        cash_dividends=[(str(d), v) for d, v in cash], time_steps=5, bump=1e-5))
    torch.testing.assert_close(result.cpu(), torch.tensor(oracle['price'], dtype=DT), rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(grads.cpu(), torch.tensor(oracle['gradients'], dtype=DT), rtol=6e-5, atol=6e-6)
    single = LatentBatch([103.], device=device, **config)
    expected = single.price(b.to(device), l.to(device), p[4:], mode=mode)
    torch.testing.assert_close(result.cpu(), expected.squeeze().cpu(), rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("checkpoint", [1, 2, 7])
def test_asset_specific_dividend_amounts_and_checkpoint_vjps(device, checkpoint):
    configs = [dict(strikes=[95., 108.], option_types=["call", "put"], rate=.03+.01*i,
                    cash_dividends=[(date(2025, 5, 1), 1.+i), (date(2026, 1, 1), .2*i)],
                    **exercise_config('bermudan')) for i in range(2)]
    plan = MultiAssetBatch(configs, axes=AXES, time_steps=4, device=device, checkpoint_steps=checkpoint)
    assert plan.groups == ((0, 1),)
    p = torch.tensor([PARAMS, PARAMS], dtype=DT, device=device, requires_grad=True)
    def price(mode):
        fields = [heston_fields(row.cpu()) for row in p]
        return plan.price(torch.stack([f[0] for f in fields]).to(device),
                          torch.stack([f[1] for f in fields]).to(device), p[:, 4:], mode=mode)
    seed = torch.tensor([[.7, -.3], [.2, .8]], dtype=DT, device=device)
    result = price('adjoint')
    gradient, = torch.autograd.grad(result, p, seed)
    full = price('full')
    full_gradient, = torch.autograd.grad(full, p, seed)
    torch.testing.assert_close(result, full, rtol=0, atol=0)
    torch.testing.assert_close(gradient, full_gradient, rtol=4e-9, atol=4e-9)
    # Each asset is separately priced by the stock-double operator and events.
    for i, config in enumerate(configs):
        for j, (strike, kind) in enumerate(zip(config['strikes'], config['option_types'])):
            oracle = stock(dict(axes=AXES, parameters=PARAMS, time_steps=4, strike=strike, option_type=kind,
                rate=config['rate'], exercise='bermudan', exercise_dates=list(map(str, config['exercise_dates'])),
                cash_dividends=[(str(d), v) for d, v in config['cash_dividends']]))
            torch.testing.assert_close(result[i, j].cpu(), torch.tensor(oracle['price'], dtype=DT), rtol=3e-11, atol=3e-11)


def test_different_exercise_and_dividend_schedules_form_separate_groups():
    rows = [dict(strikes=[100.]), dict(strikes=[100.], exercise='american'),
            dict(strikes=[100.], **exercise_config('bermudan')),
            dict(strikes=[100.], cash_dividends=[(date(2025, 5, 1), 1.)]),
            dict(strikes=[110.], cash_dividends=[(date(2025, 5, 1), 2.)])]
    plan = MultiAssetBatch(rows, axes=AXES, time_steps=3)
    assert plan.groups == ((0,), (1,), (2,), (3, 4))
    p = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(p)
    actual = plan.price(b.expand(5, *b.shape), l.expand(5, *l.shape), p[4:].expand(5, -1))
    expected = torch.stack([LatentBatch(axes=AXES, time_steps=3, **row).price(b, l, p[4:]) for row in rows])
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("kwargs", [dict(exercise='bermudan', exercise_dates=[date(2025, 5, 1)]),
    dict(cash_dividends=[(date(2025, 1, 1), 1.)]), dict(cash_dividends=[(date(2026, 1, 2), 1.)]),
    dict(cash_dividends=[(date(2025, 5, 1), 1.), (date(2025, 5, 1), 2.)]),
    dict(cash_dividends=[(date(2025, 5, 1), torch.tensor(1., requires_grad=True))])])
def test_invalid_event_configuration_fails_explicitly(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LatentBatch([100.], axes=AXES, **kwargs)


def test_dividends_with_two_latent_factors_are_explicitly_unsupported():
    with pytest.raises(ValueError, match="one latent"):
        LatentBatch([100.], axes=AXES+[[-1., -.2, .2, 1.]], cash_dividends=[(date(2025, 5, 1), 1.)])


@pytest.mark.parametrize("device", DEVICES)
def test_exercise_full_curvature_against_original_quantlib(device):
    config = dict(axes=AXES, time_steps=4, **exercise_config('american'))
    results = []
    for multi in (False, True):
        p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
        b, l = heston_fields(p)
        if multi:
            plan = MultiAssetBatch([dict(strikes=[104.], option_types=['put'], **config)], device=device)
            result = plan.price(b[None].to(device), l[None].to(device), p[None, 4:].to(device), mode='full').sum()
        else:
            result = fd_latent(b, l, p[4:], strike=104., option_type='put', mode='full', **config)
        first, = torch.autograd.grad(result, p, create_graph=True)
        second, = torch.autograd.grad(first.square().sum(), p)
        results.append((result, first, second))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-8, atol=1e-7)
