"""Independent dense, original QuantLib, financial and derivative controls."""
from datetime import date
import math

import pytest
import torch

from quantlibtorch import (LatentBatch, LatentCalibration, OptionQuote,
                          diffusion_rollback_batch, fd_latent)
from test_latent_pde import problem, dense_reference, SCHEMES
from test_fd_latent import joint_problem, neural_fields, heston_fields, AXES, PARAMS, stock
from test_calibration import TinyModel

DT = torch.float64


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_packed_equations_and_all_shared_coefficient_rhs_vjps(d, scheme, mode):
    axes, times, fields = problem(d)
    b, l, k, u = fields
    terminal = torch.stack((u, .7*u.square(), torch.sin(u)), -1)
    actual_inputs = [x.clone().requires_grad_() for x in (b, l, k, terminal)]
    refs = [x.clone().requires_grad_() for x in (b, l, k, terminal)]
    actual = diffusion_rollback_batch(actual_inputs[0], actual_inputs[1], actual_inputs[3],
        killing=actual_inputs[2], axes=axes, times=times, horizon=.2, time_steps=3, scheme=scheme, mode=mode)
    expected = torch.stack([dense_reference(*refs[:3], refs[3][:, j], axes, times, 3, scheme)
                            for j in range(3)], -1)
    seed = torch.cos(torch.arange(actual.numel(), dtype=DT)*.7).reshape_as(actual)
    gradients = torch.autograd.grad(actual, actual_inputs, seed)
    reference_gradients = torch.autograd.grad(expected, refs, seed)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(gradients, reference_gradients):
        torch.testing.assert_close(a, e, rtol=2e-10, atol=4e-12)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_batched_financial_prices_and_parameter_state_vjps_match_original(m, scheme, mode):
    axes, features, net, initial = joint_problem(m)
    strikes, types = [91., 100., 109.], ["call", "put", "call"]
    curves = dict(discount_curve=[(date(2025, 1, 1), 1.), (date(2025, 7, 1), .98), (date(2026, 1, 1), .95)],
                  dividend_curve=[(date(2025, 1, 1), 1.), (date(2026, 1, 1), .98)])
    plan = LatentBatch(strikes, axes=axes, times=[0., .4, 1.], option_types=types,
                       time_steps=3, scheme=scheme, **curves)
    b, l = neural_fields(net, features, m)
    parameters = (*net.parameters(), initial)
    values = plan.price(b, l, initial, mode=mode)
    seed = torch.tensor([.7, -.2, 1.1], dtype=DT)
    grads = torch.autograd.grad(values, parameters, seed)
    b, l = neural_fields(net, features, m)
    expected = torch.stack([fd_latent(b, l, initial, axes=axes, times=[0., .4, 1.],
        strike=strike, option_type=kind, time_steps=3, scheme=scheme, mode=mode, **curves)
        for strike, kind in zip(strikes, types)])
    expected_grads = torch.autograd.grad(expected, parameters, seed)
    torch.testing.assert_close(values, expected, rtol=4e-12, atol=5e-12)
    for a, e in zip(grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=3e-10, atol=4e-10)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_strided_fields_initial_and_terminal_ownership_and_finite_differences(mode):
    axes, times, fields = problem(2)
    b, l, k, u = fields
    inputs = [b, l, k, torch.stack((u, u.square()), -1)]
    leaves = [torch.stack((x, x), -1).requires_grad_() for x in inputs]
    views = [x[..., 0] for x in leaves]
    original = [x.detach().clone() for x in leaves]
    def loss(v):
        return diffusion_rollback_batch(v[0], v[1], v[3], killing=v[2], axes=axes,
            times=times, horizon=.2, time_steps=3, mode=mode).square().sum()
    gradients = torch.autograd.grad(loss(views), views)
    directions = [torch.cos(torch.arange(x.numel(), dtype=DT)).reshape_as(x)*.2 for x in inputs]
    with torch.no_grad():
        h = 1e-5
        difference = (loss([x+h*v for x, v in zip(inputs, directions)])-
                      loss([x-h*v for x, v in zip(inputs, directions)]))/(2*h)
    torch.testing.assert_close(sum((g*v).sum() for g, v in zip(gradients, directions)), difference,
                               rtol=2e-7, atol=2e-8)
    for before, after in zip(original, leaves):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_full_packed_curvature_matches_independent_dense_equations():
    axes, times, (b, l, k, u) = problem(2)
    derivatives = []
    for packed in (True, False):
        p = torch.tensor(.8, dtype=DT, requires_grad=True)
        if packed:
            value = diffusion_rollback_batch(b*p, l*(p+.2), torch.stack((u, u*p), -1),
                killing=k, axes=axes, times=times, horizon=.2, time_steps=2, mode="full").square().sum()
        else:
            value = sum(dense_reference(b*p, l*(p+.2), k, terminal, axes, times, 2,
                        "hundsdorfer").square().sum() for terminal in (u, u*p))
        first, = torch.autograd.grad(value, p, create_graph=True)
        second, = torch.autograd.grad(first, p)
        derivatives.append((value, first, second))
    for a, b in zip(*derivatives):
        torch.testing.assert_close(a, b, rtol=3e-11, atol=2e-11)


def test_equation_adjoint_rejects_double_backward():
    axes, times, (b, l, k, u) = problem(2)
    l.requires_grad_()
    result = diffusion_rollback_batch(b, l, u[:, None], killing=k, axes=axes,
        times=times, horizon=.2, time_steps=2, mode="adjoint")
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(result.sum(), l, create_graph=True)


def test_plan_does_not_retain_model_fields_or_price_graphs():
    import gc
    import weakref
    p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
    plan = LatentBatch([95., 105.], axes=AXES, time_steps=3)
    refs = []
    for _ in range(3):
        b, l = heston_fields(p)
        values = plan.price(b, l, p[4:])
        refs.extend(weakref.ref(v) for v in (b, l, values))
        values.sum().backward()
        del b, l, values
        gc.collect()
        assert all(r() is None for r in refs)
        p.grad = None


@pytest.mark.parametrize("bad", ["drift_shape", "factor_shape", "state_shape", "dtype", "nan", "overflow", "mode"])
def test_native_plan_input_validation_and_recovery(bad):
    p = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(p)
    z, mode = p[4:], "adjoint"
    plan = LatentBatch([100.], axes=AXES, time_steps=2)
    if bad == "drift_shape": b = b[..., :0]
    elif bad == "factor_shape": l = l[..., :0]
    elif bad == "state_shape": z = z[None]
    elif bad == "dtype": l = l.float()
    elif bad == "nan": b = b*float('nan')
    elif bad == "overflow": l = l*1e300
    elif bad == "mode": mode = "invalid"
    with pytest.raises((RuntimeError, ValueError)):
        plan.price(b, l, z, mode=mode)
    assert torch.isfinite(plan.price(*heston_fields(p), p[4:])).all()


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_batched_heston_against_isolated_stock_quantlib_and_independent_bumps(mode):
    p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
    plan = LatentBatch([100., 110.], axes=AXES, time_steps=5, option_types=["call", "put"])
    values = plan.price(*heston_fields(p), p[4:], mode=mode)
    grads, = torch.autograd.grad(values[0], p)
    oracle = stock(dict(axes=AXES, parameters=PARAMS, time_steps=5, scheme="hundsdorfer", bump=1e-5))
    torch.testing.assert_close(values[0], torch.tensor(oracle["price"], dtype=DT), rtol=2e-12, atol=5e-12)
    torch.testing.assert_close(grads, torch.tensor(oracle["gradients"], dtype=DT), rtol=4e-5, atol=4e-6)


@pytest.mark.parametrize("field,value", [
    ("axes", [[1., 2., 3., 4.], [-1., -.2, .4, 1.]]),
    ("scheme", "douglas"), ("maturity_date", date(2025, 9, 1)),
    ("time_steps", 7), ("times", [0., .5, 1.]), ("rate", .07),
    ("exercise", "american"), ("exercise_dates", [date(2025, 6, 1)]),
    ("cash_dividends", [(date(2025, 6, 1), 1.)]),
    ("boundary_conditions", ["dirichlet"]), ("damping_steps", 2),
    ("discount_curve", [(date(2025, 1, 1), 1.), (date(2026, 1, 1), .96)])])
def test_incompatible_contract_configurations_are_rejected(field, value):
    shared = dict(axes=AXES, time_steps=3)
    with pytest.raises(ValueError, match="incompatible"):
        LatentBatch.from_contracts([dict(strike=100.), dict(strike=110., **{field: value})], **shared)


@pytest.mark.parametrize("kwargs", [dict(exercise="invalid"), dict(exercise="bermudan"),
    dict(exercise_dates=[date(2025, 6, 1)]), dict(cash_dividends=[(date(2025, 6, 1), -1.)]),
    dict(boundary_conditions=["neumann"]), dict(damping_steps=1), dict(option_types=["call", "put"]),
    dict(scheme="implicit"), dict(times=[.1, 1.]), dict(maturity_date=date(2024, 1, 1))])
def test_unsupported_or_malformed_shared_contracts_fail(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LatentBatch([100.], axes=AXES, **kwargs)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_plan_reuse_fresh_graphs_ordering_state_gradients_and_input_copies(mode):
    axes = [list(a) for a in AXES]
    plan = LatentBatch.from_contracts([dict(strike=110., option_type="put"), dict(strike=100.)],
                                     axes=axes, time_steps=3)
    axes[0][0] = -999.
    assert plan.axes[0][0] == AXES[0][0]
    for bump in [0., .01, -.02, 0.]:
        p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
        p = p+torch.tensor([0., 0., bump, 0., 0., 0.], dtype=DT)
        result = plan.price(*heston_fields(p), p[4:], mode=mode)
        seed = torch.tensor([.2, .7], dtype=DT)
        grad, = torch.autograd.grad(result, p, seed)
        h = 1e-5
        direction = torch.tensor([.1, .02, .1, -.1, .01, .01], dtype=DT)
        with torch.no_grad():
            plus, minus = p+h*direction, p-h*direction
            diff = ((plan.price(*heston_fields(plus), plus[4:], mode=mode)-
                     plan.price(*heston_fields(minus), minus[4:], mode=mode))*seed).sum()/(2*h)
        torch.testing.assert_close((grad*direction).sum(), diff, rtol=2e-6, atol=3e-7)
    with pytest.raises(RuntimeError, match="strictly inside"):
        plan.price(*heston_fields(p), p[4:]+10, mode=mode)
    assert torch.isfinite(plan.price(*heston_fields(p), p[4:], mode=mode)).all()


def calibration(execution):
    # Interleaved maturities exercise grouping and restoration of quote order.
    return LatentCalibration([
        OptionQuote(95., date(2025, 4, 1), 8.1, .7),
        OptionQuote(105., date(2026, 1, 1), 4.2, 1.3),
        OptionQuote(105., date(2025, 4, 1), 1.8, .8, "put"),
        OptionQuote(100., date(2026, 1, 1), 8.8, .9)],
        axes=[[math.log(s) for s in [40., 70., 95., 110., 140., 200.]], [-1., -.3, .3, 1.]],
        times=[0., .3, 1.], steps_per_year=3, minimum_steps=2, execution=execution)


def test_mutating_a_calibration_configuration_cannot_reuse_a_stale_operator():
    problem, model = calibration("batched"), TinyModel()
    problem.market['rate'] = .07
    with pytest.raises(ValueError, match="configuration changed"):
        problem.prices(model, model.initial)
    with pytest.raises(ValueError, match="configuration changed"):
        problem.backward(model, model.initial)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_batched_calibration_retained_streamed_gradients_and_parameter_bumps(mode):
    model = TinyModel()
    reference = calibration("serial")
    packed = calibration("batched")
    expected_loss, expected_prices = reference.loss(model, model.initial, mode=mode)
    expected_gradients = torch.autograd.grad(expected_loss, (model.p, model.z))
    loss, prices = packed.loss(model, model.initial, mode=mode)
    actual_gradients = torch.autograd.grad(loss, (model.p, model.z))
    contexts = []
    from contextlib import nullcontext
    def context(indices):
        contexts.append(indices)
        return nullcontext()
    loss2, prices2 = packed.backward(model, model.initial, mode=mode, contract_context=context)
    assert contexts == [(0, 2), (1, 3)]
    torch.testing.assert_close(loss, expected_loss, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(prices, expected_prices, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(loss2, loss, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(prices2, prices, rtol=0, atol=0)
    for a, e, streamed in zip(actual_gradients, expected_gradients, (model.p.grad, model.z.grad)):
        torch.testing.assert_close(a, e, rtol=3e-10, atol=3e-10)
        torch.testing.assert_close(streamed, e, rtol=3e-10, atol=3e-10)
    packed.backward(model, model.initial, mode=mode)
    torch.testing.assert_close(model.p.grad, 2*actual_gradients[0], rtol=3e-10, atol=3e-10)
    if mode == "full":
        curvature = []
        for problem in (reference, packed):
            value = problem.loss(model, model.initial, mode="full")[0]
            first, = torch.autograd.grad(value, model.z, create_graph=True)
            curvature.append(torch.autograd.grad(first, model.z)[0])
        torch.testing.assert_close(*curvature, rtol=3e-10, atol=3e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_cuda_prototype_prices_shared_gradients_and_state_transfers(m, scheme, mode):
    axes, features, net, initial = joint_problem(m)
    values, gradients = [], []
    for device in ['cpu', 'cuda']:
        plan = LatentBatch([95., 105.], axes=axes, times=[0., .4, 1.], time_steps=3,
                           option_types=['call', 'put'], scheme=scheme, device=device)
        b, l = neural_fields(net, features, m)
        result = plan.price(b.to(device), l.to(device), initial.to(device), mode=mode)
        assert result.device.type == device
        g = torch.autograd.grad(result, (*net.parameters(), initial),
                                torch.tensor([.7, -.3], dtype=DT, device=device))
        values.append(result.detach().cpu()); gradients.append(g)
    torch.testing.assert_close(*values, rtol=2e-11, atol=2e-11)
    for cpu, gpu in zip(*gradients):
        torch.testing.assert_close(cpu, gpu, rtol=3e-9, atol=3e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_full_curvature_and_equation_adjoint_domain():
    derivatives = []
    for device in ['cpu', 'cuda']:
        p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
        plan = LatentBatch([100., 110.], axes=AXES, time_steps=3, device=device)
        b, l = heston_fields(p)
        values = plan.price(b.to(device), l.to(device), p[4:].to(device), mode='full')
        first, = torch.autograd.grad(values.square().sum(), p, create_graph=True)
        second, = torch.autograd.grad(first.square().sum(), p)
        derivatives.append((first, second))
    for a, b in zip(*derivatives):
        torch.testing.assert_close(a, b, rtol=3e-9, atol=1e-6)
    bad = LatentBatch([100.], axes=AXES, time_steps=1, dividend_yield=-3., device='cuda')
    p = torch.tensor(PARAMS, dtype=DT, device='cuda')
    b, l = heston_fields(p.cpu())
    with pytest.raises(RuntimeError, match="diagonally-dominant"):
        bad.price(b.cuda(), l.cuda(), p[4:])


@pytest.mark.parametrize("field", ['drift', 'diffusion', 'initial'])
def test_packed_adjoint_explicitly_rejects_forward_tangents(field):
    p = torch.tensor(PARAMS, dtype=DT)
    plan = LatentBatch([100.], axes=AXES, time_steps=2)
    b, l = heston_fields(p)
    with torch.autograd.forward_ad.dual_level():
        args = [b, l, p[4:]]
        i = ['drift', 'diffusion', 'initial'].index(field)
        args[i] = torch.autograd.forward_ad.make_dual(args[i], torch.ones_like(args[i]))
        with pytest.raises(RuntimeError, match="forward tangents"):
            plan.price(*args, mode='adjoint')
