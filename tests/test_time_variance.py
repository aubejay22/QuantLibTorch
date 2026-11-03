"""Fused time interpolation preserves the original square and all AD paths."""
from contextlib import nullcontext
import math
import os

import pytest
import torch
from torch.autograd import forward_ad

from quantlibtorch import fd_vanilla
from test_backward_duals import compare_incoming_duals

DTYPE = torch.float64


def variance(low, high, weight=.37, native=True):
    return torch.ops.quantlibtorch._time_variance(low, high, weight, native)


def equation(low, high, weight=.37):
    sigma = (1.-weight)*low + weight*high
    return sigma*sigma


def enabled():
    return os.environ.get('QLT_TIME_VARIANCE_ARITHMETIC', '1') == '1'


def count(profile, name):
    return sum(e.count for e in profile.key_averages() if e.key == name)


def inputs(view='contiguous', n=17):
    a = torch.sin(torch.arange(n*2, dtype=DTYPE)*.3).requires_grad_()
    b = torch.cos(torch.arange(n*2, dtype=DTYPE)*.7).requires_grad_()
    leaves = a, b
    if view == 'strided': return leaves, (a[::2], b[1::2])
    if view == 'negative': return leaves, (torch._neg_view(a), torch._neg_view(b))
    if view == 'shared': return (a,), (a, a)
    if view == 'expanded': return leaves, (a[:1].expand(n), b[:1].expand(n))
    return leaves, leaves


@pytest.mark.parametrize('n', [0, 1, 17, 257])
@pytest.mark.parametrize('weight', [-.3, 0., .13, .5, 1., 1.3])
def test_original_forward_bits_and_all_vjps(n, weight):
    low = torch.sin(torch.arange(n, dtype=DTYPE)).requires_grad_()
    high = torch.cos(torch.arange(n, dtype=DTYPE)).requires_grad_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = variance(low, high, weight)
    reference = equation(low, high, weight)
    assert torch.equal(actual.view(torch.int64), reference.view(torch.int64))
    assert count(profile, 'qlt::time_variance_forward') == int(enabled())
    g = torch.sin(torch.arange(n, dtype=DTYPE)+.3)
    torch.testing.assert_close(torch.autograd.grad(actual, (low, high), g),
                               torch.autograd.grad(reference, (low, high), g), rtol=0, atol=0)
    torch.testing.assert_close(actual, variance(low, high, weight, native=False), rtol=0, atol=0)


@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'shared', 'expanded'])
def test_views_aliases_and_derivatives_through_third_order(view):
    leaves, xs = inputs(view)
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    actual, reference = variance(*xs).sin().sum(), equation(*xs).sin().sum()
    for order in range(3):
        ga = torch.autograd.grad(actual, leaves, create_graph=order < 2)
        ge = torch.autograd.grad(reference, leaves, create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-12, atol=2e-12)
        actual = sum((g*torch.cos(t)).sum() for g, t in zip(ga, leaves))
        reference = sum((g*torch.cos(t)).sum() for g, t in zip(ge, leaves))
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('selected', [0, 1])
def test_single_active_input(selected):
    _, xs = inputs(n=5)
    xs = tuple(t if i == selected else t.detach() for i, t in enumerate(xs))
    actual, reference = variance(*xs).sum(), equation(*xs).sum()
    torch.testing.assert_close(torch.autograd.grad(actual, xs[selected]),
                               torch.autograd.grad(reference, xs[selected]), rtol=0, atol=0)


def test_first_and_second_finite_differences():
    _, xs = inputs(n=4)
    assert torch.autograd.gradcheck(variance, xs, fast_mode=True)
    assert torch.autograd.gradgradcheck(variance, xs, fast_mode=True)


@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'shared', 'expanded'])
@pytest.mark.parametrize('create_graph', [False, True])
def test_incoming_dual_weights(view, create_graph):
    leaves, xs = inputs(view)
    compare_incoming_duals(leaves, variance(*xs), equation(*xs), create_graph=create_graph)


@pytest.mark.parametrize('selected', [0, 1, 'both'])
@pytest.mark.parametrize('no_grad', [False, True])
def test_forward_duals_use_original_expression(selected, no_grad):
    _, xs = inputs(n=5)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, torch.cos(t)) if selected in (i, 'both') else t
                          for i, t in enumerate(xs))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = forward_ad.unpack_dual(variance(*duals))
            reference = forward_ad.unpack_dual(equation(*duals))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert count(profile, 'qlt::time_variance_forward') == 0


@pytest.mark.parametrize('transform', ['grad', 'jvp', 'functionalize', 'vmap', 'fake'])
def test_tensor_transforms_fall_back(transform):
    _, xs = inputs(n=5)
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: f(*args).sin().sum(), argnums=(0, 1))(*xs)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.cos(t) for t in xs))
    elif transform == 'functionalize':
        run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap':
        run = lambda f: torch.vmap(f)(*(torch.stack((t, t*1.1)) for t in xs))
    else:
        from torch.fx.experimental.proxy_tensor import make_fx
        run = lambda f: make_fx(lambda low, high: f(low, high), tracing_mode='fake')(*xs)(*xs)
    torch.testing.assert_close(run(variance), run(equation), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_other_dtype_and_device_fallback(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    _, xs = inputs(n=5)
    xs = tuple(t.detach().to(device=device, dtype=dtype).requires_grad_() for t in xs)
    actual, reference = variance(*xs), equation(*xs)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), xs),
                               torch.autograd.grad(reference.sum(), xs), rtol=0, atol=0)


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_no_grad_and_inference(context):
    _, xs = inputs(n=5)
    with context():
        actual = variance(*xs)
        torch.testing.assert_close(actual, equation(*xs), rtol=0, atol=0)
        assert not actual.requires_grad


def test_saved_input_version_is_checked():
    _, xs = inputs(n=5)
    value = variance(*xs)
    with torch.no_grad(): xs[0].add_(.1)
    if enabled():
        with pytest.raises(RuntimeError, match='modified by an inplace operation'):
            torch.autograd.grad(value.sum(), xs)
    # The original expression saves sigma, so it need not reject this mutation.


@pytest.mark.parametrize('scale', [1e-150, 1e150])
def test_extreme_finite_inputs(scale):
    _, xs = inputs(n=5)
    xs = tuple((t.detach()*scale).requires_grad_() for t in xs)
    actual, reference = variance(*xs), equation(*xs)
    assert torch.equal(actual.view(torch.int64), reference.view(torch.int64))
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), xs),
                               torch.autograd.grad(reference.sum(), xs), rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0., .37, 1.])
def test_nonfinite_inputs_preserve_original_operations(weight):
    low = torch.tensor([float('inf'), float('-inf'), float('nan'), -0., 0.], dtype=DTYPE, requires_grad=True)
    high = torch.tensor([.2, .3, .4, 0., -0.], dtype=DTYPE, requires_grad=True)
    actual, reference = variance(low, high, weight), equation(low, high, weight)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), (low, high)),
                               torch.autograd.grad(reference.sum(), (low, high)), rtol=0, atol=0, equal_nan=True)


def test_live_financial_graphs_keep_original_surface_edges_after_owner_destruction():
    fields = [torch.tensor([[.21, .24], [.26, .22]], dtype=DTYPE, requires_grad=True) for _ in range(3)]
    options = dict(times=[0., .7], log_spots=[math.log(70.), math.log(145.)], time_steps=3, grid_points=11)
    prices = [fd_vanilla(field, spot=95.+i*4, **options) for i, field in enumerate(fields)]
    for i in (2, 0, 1):
        reference = fd_vanilla(fields[i], spot=95.+i*4, kernels='scalar', **options)
        torch.testing.assert_close(prices[i], reference, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(torch.autograd.grad(prices[i], fields[i]), torch.autograd.grad(reference, fields[i]),
                                   rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize('nt', [1, 3])
@pytest.mark.parametrize('mode', ['full', 'adjoint', 'inference', 'inference_mode'])
def test_fresh_financial_graphs_time_edges_and_constant_controls(nt, mode):
    recording = mode in ('full', 'adjoint')
    context = nullcontext if recording else (torch.no_grad if mode == 'inference' else torch.inference_mode)
    for nodes, spot, shift in [(11, 103., 0.), (17, 94., .02), (11, 103., 0.)]:
        field = (.22 + shift + .01*torch.sin(torch.arange(nt*8, dtype=DTYPE))).reshape(nt, 8).requires_grad_()
        options = dict(times=[.2] if nt == 1 else [.1, .43, .7],
                       log_spots=[math.log(s) for s in (55., 85., 115., 180.)],
                       time_steps=4, grid_points=nodes, spot=spot, damping_steps=1,
                       mode=mode if recording else 'full')
        with context():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = fd_vanilla(field[:, ::2], **options)
            reference = fd_vanilla(field[:, ::2], kernels='scalar', **options)
        assert bool(count(profile, 'qlt::time_variance_forward')) == (enabled() and nt > 1)
        torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)
        if recording:
            ga, = torch.autograd.grad(actual, field, create_graph=mode == 'full')
            ge, = torch.autograd.grad(reference, field, create_graph=mode == 'full')
            torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-9)
            assert torch.count_nonzero(ga[:, 1::2]) == 0
            if mode == 'full':
                torch.testing.assert_close(torch.autograd.grad(ga, field, torch.cos(field)),
                                           torch.autograd.grad(ge, field, torch.cos(field)),
                                           rtol=2e-8, atol=2e-8)
        else:
            assert not actual.requires_grad
