"""Market graph preservation across scalar, packed and checkpointed pricing."""
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantlibtorch import fd_latent, LatentBatch, MultiAssetBatch
from test_fd_latent import AXES, PARAMS, heston_fields

DT = torch.float64
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]
DATES = [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)]


def market(p, curves):
    if curves:
        return dict(discount_curve=list(zip(DATES, [1., p[0], p[1]])),
                    dividend_curve=list(zip(DATES, [1., p[2], p[3]])))
    return dict(rate=p[0], dividend_yield=p[1])


def events(kind, cash=False):
    result = dict(exercise=kind)
    if kind == "bermudan":
        result["exercise_dates"] = DATES[1:]
    if cash:
        result["cash_dividends"] = [(DATES[1], 1.1), (DATES[-1], .4)]
    return result


def oracle(p, curves, kind, cash=False):
    kwargs = {**market(p.tolist(), curves), **events(kind, cash)}
    for name in ("discount_curve", "dividend_curve", "cash_dividends"):
        if name in kwargs:
            kwargs[name] = [[str(d), v] for d, v in kwargs[name]]
    if "exercise_dates" in kwargs:
        kwargs["exercise_dates"] = list(map(str, kwargs["exercise_dates"]))
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name("stock_financial_reference.py"))],
        input=json.dumps(dict(axes=AXES, parameters=PARAMS, time_steps=4, option_type="put", **kwargs)),
        text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


@pytest.mark.parametrize("curves", [False, True])
@pytest.mark.parametrize("kind", ["european", "american", "bermudan"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_scalar_market_sensitivities_match_stock_bumps(curves, kind, mode):
    p = torch.tensor([.981, .953, .995, .982] if curves else [.034, .012], dtype=DT, requires_grad=True)
    params = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(params)
    value = fd_latent(b, l, params[4:], axes=AXES, time_steps=4, option_type="put",
                      mode=mode, **events(kind), **market(p, curves))
    grad, = torch.autograd.grad(value, p)
    expected = oracle(p.detach(), curves, kind)
    torch.testing.assert_close(value, torch.tensor(expected["price"], dtype=DT), rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(grad, torch.tensor(expected["gradients"], dtype=DT), rtol=3e-5, atol=4e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("curves", [False, True])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_packed_market_graphs_events_and_repeated_calls(device, curves, mode):
    p = torch.tensor([.981, .953, .995, .982] if curves else [.034, .012], dtype=DT, requires_grad=True)
    params = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(params)
    config = dict(axes=AXES, time_steps=4, option_types=["put"], **events("bermudan", cash=True))
    single = LatentBatch([100.], device=device, **config)
    multi = MultiAssetBatch([single, single], checkpoint_steps=2)
    fields = [v.to(device) for v in (b, l, params[4:])]
    expected = oracle(p.detach(), curves, "bermudan", cash=True)
    for repeat in range(2):
        values = single.price(*fields, mode=mode, **market(p+0., curves))
        gradients, = torch.autograd.grad(values.sum(), p)
        torch.testing.assert_close(values.cpu().squeeze(), torch.tensor(expected["price"], dtype=DT), rtol=3e-11, atol=3e-11)
        torch.testing.assert_close(gradients, torch.tensor(expected["gradients"], dtype=DT), rtol=3e-5, atol=4e-6)
        result = multi.price(*[torch.stack([v, v]) for v in fields], mode=mode,
                             markets=[market(p+0., curves), market(p+0., curves)])
        grad, = torch.autograd.grad(result.sum()/2, p)
        torch.testing.assert_close(result.cpu(), values.cpu().expand(2, 1), rtol=3e-11, atol=3e-11)
        torch.testing.assert_close(grad, gradients, rtol=3e-9, atol=3e-9)


@pytest.mark.parametrize("engine", ["scalar", "batch", "multi"])
@pytest.mark.parametrize("curves", [False, True])
def test_market_full_higher_derivatives(engine, curves):
    p = torch.tensor([.981, .953, .995, .982] if curves else [.033, .013], dtype=DT, requires_grad=True)
    params = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(params)
    plan = LatentBatch([100.], axes=AXES, time_steps=3)
    multi = MultiAssetBatch([plan])
    def price(p):
        if engine == "scalar":
            return fd_latent(b, l, params[4:], axes=AXES, time_steps=3, **market(p, curves))
        if engine == "batch":
            return plan.price(b, l, params[4:], mode="full", **market(p, curves))
        return multi.price(b[None], l[None], params[None, 4:], mode="full", markets=[market(p, curves)])
    assert torch.autograd.gradcheck(price, (p,), atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(price, (p,), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
@pytest.mark.parametrize("checkpoint", [1, 7])
def test_distinct_asset_markets_groups_and_joint_vjps(mode, checkpoint):
    plans = [LatentBatch([97., 106.], axes=AXES, time_steps=3, scheme=scheme)
             for scheme in ("douglas", "hundsdorfer", "douglas")]
    plan = MultiAssetBatch(plans, checkpoint_steps=checkpoint)
    assert plan.groups == ((0, 2), (1,))
    p = torch.tensor([[.02, .01], [.04, .015], [.025, .005]], dtype=DT, requires_grad=True)
    model = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
    seed = torch.tensor([[.7, -.2], [.3, .9], [-.1, .4]], dtype=DT)
    b, l = heston_fields(model)
    value = plan.price(b.expand(3, *b.shape), l.expand(3, *l.shape), model[4:].expand(3, -1),
                       mode=mode, markets=[market(row, False) for row in p])
    grads = torch.autograd.grad(value, (p, model), seed)
    b, l = heston_fields(model)
    reference = torch.stack([single.price(b, l, model[4:], mode=mode, **market(row, False))
                             for single, row in zip(plans, p)])
    expected = torch.autograd.grad(reference, (p, model), seed)
    torch.testing.assert_close(value, reference, rtol=3e-11, atol=3e-11)
    for actual, target in zip(grads, expected):
        torch.testing.assert_close(actual, target, rtol=3e-9, atol=3e-9)


@pytest.mark.parametrize("active", ["rate", "dividend_yield", "neither"])
def test_checkpoint_requests_only_active_market_vjps(active):
    model = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
    value = torch.tensor(.032 if active == "rate" else .012, dtype=DT, requires_grad=True)
    overrides = {} if active == "neither" else {active: value}
    plan = MultiAssetBatch([dict(strikes=[97., 106.])], axes=AXES, time_steps=3, checkpoint_steps=2)
    gradients = []
    for mode in ("full", "adjoint"):
        b, l = heston_fields(model)
        price = plan.price(b[None], l[None], model[None, 4:], mode=mode, markets=[overrides])
        inputs = (model,) if active == "neither" else (model, value)
        gradients.append(torch.autograd.grad(price.sum(), inputs))
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-9, atol=3e-9)


def test_market_validation_and_plan_defaults():
    params = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(params)
    curve = list(zip(DATES, [1., .98, .95]))
    plan = LatentBatch([100.], axes=AXES, time_steps=3, discount_curve=curve)
    before = plan.price(b, l, params[4:])
    with pytest.raises(ValueError, match="overridden"):
        plan.price(b, l, params[4:], rate=torch.tensor(.03, dtype=DT, requires_grad=True))
    with pytest.raises(ValueError, match="fixed one"):
        plan.price(b, l, params[4:], discount_curve=[(DATES[0], torch.tensor(1., dtype=DT, requires_grad=True)), (DATES[-1], .95)])
    with pytest.raises(TypeError, match="unsupported"):
        plan.price(b, l, params[4:], spot=100.)
    active = torch.tensor(.03, dtype=DT, requires_grad=True)
    cleared = plan.price(b, l, params[4:], discount_curve=None, rate=active)
    assert torch.autograd.grad(cleared.sum(), active)[0].abs()>1.
    torch.testing.assert_close(before, plan.price(b, l, params[4:]), rtol=0, atol=0)


@pytest.mark.parametrize("engine", ["scalar", "batch", "multi"])
@pytest.mark.parametrize("curves", [False, True])
def test_adjoint_market_forward_tangents_rejected(engine, curves):
    params = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(params)
    plan = LatentBatch([100.], axes=AXES, time_steps=2)
    multi = MultiAssetBatch([plan])
    p = torch.tensor([.981, .953, .995, .982] if curves else [.033, .013], dtype=DT)
    with torch.autograd.forward_ad.dual_level():
        p = torch.autograd.forward_ad.make_dual(p, torch.ones_like(p))
        with pytest.raises(RuntimeError, match="forward tangents"):
            if engine == "scalar":
                fd_latent(b, l, params[4:], axes=AXES, time_steps=2, mode="adjoint", **market(p, curves))
            elif engine == "batch":
                plan.price(b, l, params[4:], **market(p, curves))
            else:
                multi.price(b[None], l[None], params[None, 4:], markets=[market(p, curves)])


def test_shadowed_rate_and_fixed_first_discount_forward_tangents_rejected():
    from quantlibtorch.financial import _market
    from quantlibtorch.fd import _serial
    reference, maturity = (_serial("date", d) for d in (DATES[0], DATES[-1]))
    with torch.autograd.forward_ad.dual_level():
        active = torch.autograd.forward_ad.make_dual(torch.tensor(1., dtype=DT), torch.tensor(1., dtype=DT))
        with pytest.raises(ValueError, match="fixed one"):
            _market(.03, .01, [(DATES[0], active), (DATES[-1], .95)], None, reference, maturity)
        with pytest.raises(ValueError, match="overridden"):
            _market(active*.03, .01, [(DATES[0], 1.), (DATES[-1], .95)], None,
                    reference, maturity)
