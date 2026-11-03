"""Elementary step preparation retains every output VJP and solver-state edge."""
from contextlib import nullcontext
import os

import pytest
import torch
from torch.autograd import forward_ad

from quantlibtorch import fd_vanilla
from test_backward_duals import compare_incoming_duals
from test_full_tensor_step import inputs, step
from test_thomas_arithmetic import state_reference

DTYPE = torch.float64


def prepare(bands, state, dt=.04, theta=.6, kind=0, native=True, permuted=False):
    return tuple(torch.ops.quantlibtorch._step_preparation(bands, state, dt, theta, kind, native, permuted))


def equation(bands, state, dt=.04, theta=.6, kind=0, permuted=False):
    n = state.numel()
    rows = torch.arange(n, device=state.device)
    left = (3*rows+1) % n if permuted else torch.cat((rows[1:2], rows[:-1]))
    right = (5*rows+2) % n if permuted else torch.cat((rows[1:], rows[-2:-1]))
    lower, diagonal, upper = bands.unbind(0)
    rhs = state
    if kind == 0:
        applied = (state[left]*lower+state*diagonal)+state[right]*upper
        rhs = (state+dt*applied)-(theta*dt)*applied
    a = (-theta)*dt
    return a*lower[1:], a*diagonal+1., a*upper[:-1], rhs


def enabled():
    return (os.environ.get('QLT_STEP_PREPARATION', '1') == '1'
            and os.environ.get('QLT_ARITHMETIC_NATIVE_FORWARD', '1') == '1')


def loss(outputs):
    return sum((t.sin()*torch.cos(torch.arange(t.numel(), dtype=t.dtype, device=t.device)+.2)).sum()
               for t in outputs)


def grads(value, variables, graph=False):
    result = torch.autograd.grad(value, variables, create_graph=graph, allow_unused=True)
    return tuple(torch.zeros_like(v) if g is None else g for g, v in zip(result, variables))


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('theta', [-.3, 0., .5, 1., 1.3])
@pytest.mark.parametrize('n', [2, 9])
def test_all_prepared_outputs_keep_original_forward_bits_and_vjps(kind, theta, n):
    xs = inputs(n)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = prepare(*xs, kind=kind, theta=theta)
    expected = equation(*xs, kind=kind, theta=theta)
    for a, e in zip(actual, expected): assert torch.equal(a.view(torch.int64), e.view(torch.int64))
    assert any(e.key == 'qlt::step_preparation_forward' for e in profile.key_averages()) == enabled()
    torch.testing.assert_close(grads(loss(actual), xs), grads(loss(expected), xs), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('selected', [0, 1, 2, 3, 'all'])
def test_each_output_weight_including_unused_outputs(kind, selected):
    xs = inputs()
    actual, expected = prepare(*xs, kind=kind), equation(*xs, kind=kind)
    indices = [i for i in range(4) if selected in (i, 'all')]
    la, le = loss([actual[i] for i in indices]), loss([expected[i] for i in indices])
    for order in range(3):
        ga, ge = grads(la, xs, order < 2), grads(le, xs, order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-11, atol=2e-12)
        la = sum((g*torch.cos(x)).sum() for g, x in zip(ga, xs))
        le = sum((g*torch.cos(x)).sum() for g, x in zip(ge, xs))


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'shared', 'expanded'])
@pytest.mark.parametrize('permuted', [False, True])
def test_all_third_derivatives_views_aliases_and_actual_neighbours(kind, view, permuted):
    leaves = inputs(18 if view == 'strided' else 9)
    xs = leaves
    if view == 'strided': xs = (leaves[0][:, ::2], leaves[1][::2])
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in leaves)
    elif view == 'shared':
        leaves = (torch.linspace(.01, .1, 9, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0].expand(3, 9), leaves[0])
    elif view == 'expanded':
        leaves = (torch.linspace(.01, .1, 4, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0][:3, None].expand(3, 9), leaves[0][3:].expand(9))
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    actual, reference = prepare(*xs, kind=kind, permuted=permuted), equation(*xs, kind=kind, permuted=permuted)
    la, le = loss(actual), loss(reference)
    for order in range(3):
        ga, ge = grads(la, leaves, order < 2), grads(le, leaves, order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-11, atol=2e-12)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, leaves))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, leaves))
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('kind', [0, 1])
def test_first_and_second_differences_of_all_outputs(kind):
    f = lambda b, s: prepare(b, s, kind=kind)
    assert torch.autograd.gradcheck(f, inputs(5), fast_mode=True)
    assert torch.autograd.gradgradcheck(f, inputs(5), fast_mode=True)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('selected', [0, 1, 2, 3, 'all'])
@pytest.mark.parametrize('create_graph', [False, True])
def test_incoming_duals_on_every_output(kind, selected, create_graph):
    xs = inputs()
    compare_incoming_duals(xs, prepare(*xs, kind=kind), equation(*xs, kind=kind), selected=selected, create_graph=create_graph)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('no_grad', [False, True])
def test_forward_duals_keep_tensor_expression(kind, no_grad):
    xs = inputs()
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, torch.cos(t)) for t in xs)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = prepare(*duals, kind=kind)
            expected = equation(*duals, kind=kind)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(forward_ad.unpack_dual(a), forward_ad.unpack_dual(e), rtol=0, atol=0)
            assert not any(e.key == 'qlt::step_preparation_forward' for e in profile.key_averages())


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('transform', ['grad', 'jvp', 'functionalize', 'vmap', 'fake'])
def test_function_transforms_use_tensor_fallback(kind, transform):
    xs = inputs()
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: loss(f(*args)), argnums=(0, 1))(*xs)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.cos(t) for t in xs))
    elif transform == 'functionalize':
        run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap':
        run = lambda f: torch.vmap(f)(*(torch.stack((t, t*1.1)) for t in xs))
    else:
        from torch.fx.experimental.proxy_tensor import make_fx
        run = lambda f: make_fx(f, tracing_mode='fake')(*xs)(*xs)
    torch.testing.assert_close(run(lambda b, s: prepare(b, s, kind=kind)),
                               run(lambda b, s: equation(b, s, kind=kind)), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('active', [0, 1])
def test_one_active_input(kind, active):
    xs = tuple(t if i == active else t.detach() for i, t in enumerate(inputs()))
    actual, expected = prepare(*xs, kind=kind), equation(*xs, kind=kind)
    torch.testing.assert_close(grads(loss(actual), (xs[active],)), grads(loss(expected), (xs[active],)), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_other_dtype_and_device_fallback(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    xs = tuple(t.detach().to(device=device, dtype=dtype).requires_grad_() for t in inputs())
    actual, expected = prepare(*xs), equation(*xs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(grads(loss(actual), xs), grads(loss(expected), xs), rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_probe_no_grad_and_inference(context):
    xs = inputs()
    with context():
        actual = prepare(*xs)
        torch.testing.assert_close(actual, equation(*xs), rtol=0, atol=0)
        assert all(not t.requires_grad for t in actual)


@pytest.mark.parametrize('kind', [0, 1])
def test_preparation_and_solver_keep_all_elimination_state_third_derivatives(kind):
    xs = inputs(5)
    prepared = prepare(*xs, kind=kind)
    reference = equation(*xs, kind=kind)
    actual = torch.ops.quantlibtorch._thomas_arithmetic_state(*prepared[:3], prepared[3][:, None], False, True)
    expected = state_reference(*reference[:3], reference[3][:, None])
    torch.testing.assert_close(tuple(actual), expected, rtol=2e-12, atol=2e-12)
    la = sum(t.sin().sum() for t in actual)
    le = sum(t.sin().sum() for t in expected)
    for order in range(3):
        ga, ge = grads(la, xs, order < 2), grads(le, xs, order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, xs))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, xs))


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_complete_step_preparation_dispatch_and_unchanged_solver_checks(kind):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        step(*inputs(), kind=kind)
    counts = {e.key: e.count for e in profile.key_averages()}
    full = os.environ.get('QLT_FULL_TENSOR_STEP', '1') == '1' and os.environ.get('QLT_OPERATOR_ARITHMETIC', '1') == '1'
    assert counts.get('qlt::step_preparation_forward', 0) == int(enabled() and full and kind != 2)
    if full and kind != 2:
        assert counts.get('qlt::auto_domain_native', 0) == 1
        assert counts.get('qlt::auto_residual_native', 0) == 1
