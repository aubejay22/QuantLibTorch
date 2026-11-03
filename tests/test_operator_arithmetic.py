"""Full-AD stencil arithmetic against an independent dense reflected operator."""
from contextlib import nullcontext
import os

import pytest
import torch

from quantlibtorch import fd_vanilla

DTYPE = torch.float64


def arguments(n=3):
    gen = torch.Generator().manual_seed(811)
    return (torch.randn(3, n, dtype=DTYPE, generator=gen).requires_grad_(),
            torch.randn(n, dtype=DTYPE, generator=gen).requires_grad_())


def apply(bands, rhs, mode=4):
    return torch.ops.quantlibtorch._hook_apply(bands, rhs, mode, False)


def dense(bands, rhs):
    n = rhs.numel()
    rows = torch.arange(n)
    left = torch.cat((torch.tensor([1]), torch.arange(n-1)))
    right = torch.cat((torch.arange(1, n), torch.tensor([n-2])))
    matrix = torch.diag(bands[1])
    matrix = matrix.index_put((rows, left), bands[0], accumulate=True)
    matrix = matrix.index_put((rows, right), bands[2], accumulate=True)
    return matrix @ rhs


@pytest.mark.parametrize('n', [2, 7])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'expanded'])
@pytest.mark.parametrize('mode', [1, 4])
def test_full_application_values_all_vjps_and_input_preservation(n, view, mode):
    base = arguments(2*n if view == 'strided' else 1 if view == 'expanded' else n)
    if view == 'strided': xs = (base[0][:, ::2], base[1][::2])
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in base)
    elif view == 'expanded': xs = (base[0].expand(3, n), base[1].expand(n))
    else: xs = base
    original, versions = [t.clone() for t in base], [t._version for t in base]
    actual, expected = apply(*xs, mode), dense(*xs)
    weights = torch._neg_view(torch.linspace(-.2, .8, n, dtype=DTYPE))
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual, base, weights),
                               torch.autograd.grad(expected, base, weights), rtol=2e-12, atol=2e-12)
    for value, before, version in zip(base, original, versions):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('active', [(0,), (1,), (2,), (3,), (0, 1, 2, 3)])
def test_full_application_finite_first_and_second_differences(n, active):
    bands, rhs = arguments(n)
    xs = tuple(t.detach().requires_grad_(i in active) for i, t in enumerate((*bands.unbind(), rhs)))
    function = lambda lo, di, up, r: apply(torch.stack((lo, di, up)), r)
    assert torch.autograd.gradcheck(function, xs)
    assert torch.autograd.gradgradcheck(function, xs)


@pytest.mark.parametrize('n', [2, 7])
@pytest.mark.parametrize('alias', [0, 1, 2])
def test_full_application_alias_and_incoming_weight_third_derivatives(n, alias):
    bands, _ = arguments(n)
    rhs = torch._neg_view(bands[alias])
    weights = torch.cos(torch.arange(n, dtype=DTYPE)).requires_grad_()
    la = (apply(bands, rhs).sin()*weights).sum()
    le = (dense(bands, rhs).sin()*weights).sum()
    directions = (torch.cos(bands.detach()), .2*weights.detach())
    for order in range(3):
        ga = torch.autograd.grad(la, (bands, weights), create_graph=order < 2)
        ge = torch.autograd.grad(le, (bands, weights), create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-10)
        la = sum((g*d).sum() for g, d in zip(ga, directions))
        le = sum((g*d).sum() for g, d in zip(ge, directions))
    with torch.no_grad():
        inference = apply(bands, rhs)
    assert not inference.requires_grad
    torch.testing.assert_close(inference, dense(bands, rhs), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('active', [0, 1, 'all'])
@pytest.mark.parametrize('no_grad', [False, True])
def test_full_application_duals_use_original_tensor_path(active, no_grad):
    from torch.autograd import forward_ad
    xs = arguments()
    directions = tuple(torch.full_like(t, .2) if active in (i, 'all') else torch.zeros_like(t)
                       for i, t in enumerate(xs))
    p, d = torch.autograd.functional.jvp(dense, xs, directions)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, v) if active in (i, 'all') else t
                          for i, (t, v) in enumerate(zip(xs, directions)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                value, tangent = forward_ad.unpack_dual(apply(*duals))
    assert not any(e.key == 'qlt::apply_arithmetic_forward' for e in profile.key_averages())
    torch.testing.assert_close(value, p, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(tangent, d, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('transform', ['grad', 'jacrev', 'jvp', 'functionalize', 'vmap', 'unmapped'])
def test_full_application_function_transform_falls_back(transform):
    xs = arguments()
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda b, r: f(b, r).square().sum(), argnums=(0, 1))(*xs)
    elif transform == 'jacrev':
        run = lambda f: torch.func.jacrev(f, argnums=(0, 1))(*xs)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.full_like(t, .2) for t in xs))
    elif transform == 'functionalize':
        run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap':
        run = lambda f: torch.vmap(f)(*(torch.stack((t, t*1.1)) for t in xs))
    else:
        run = lambda f: torch.vmap(lambda w: f(*xs)*w)(torch.linspace(.5, 1., 2, dtype=DTYPE))
    torch.testing.assert_close(run(apply), run(dense), rtol=2e-12, atol=2e-12)


def test_full_application_fake_tensor_trace_preserves_real_values_and_vjps():
    from torch.fx.experimental.proxy_tensor import make_fx
    xs = arguments()
    traced = make_fx(lambda b, r: apply(b, r), tracing_mode='fake')(*xs)
    actual, expected = traced(*xs), dense(*xs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), xs),
                               torch.autograd.grad(expected.square().sum(), xs), rtol=2e-12, atol=2e-12)


def test_full_application_forward_dispatch():
    xs = arguments()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        value = apply(*xs)
        torch.autograd.grad(value.sum(), xs)
    counts = {e.key: e.count for e in profile.key_averages()}
    enabled = os.environ.get('QLT_OPERATOR_ARITHMETIC', '1') == '1'
    assert ('qlt::apply_arithmetic_forward' in counts) == enabled
    assert ('qlt::apply_arithmetic_backward' in counts) == enabled
    native = enabled and os.environ.get('QLT_ARITHMETIC_NATIVE_FORWARD', '1') == '1'
    assert ('qlt::apply_arithmetic_native_forward' in counts) == native
    assert counts['qlt::band_apply'] == (2 if enabled and not native else 1)


def test_full_application_pricing_dispatch_and_higher_gradients():
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        value = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='auto', mode='full')
        gradient, = torch.autograd.grad(value, sigma, create_graph=True)
        hessian, = torch.autograd.grad(gradient, sigma)
    counts = {e.key: e.count for e in profile.key_averages()}
    enabled = os.environ.get('QLT_OPERATOR_ARITHMETIC', '1') == '1'
    prepared = (enabled and os.environ.get('QLT_FULL_TENSOR_STEP', '1') == '1'
                and os.environ.get('QLT_STEP_PREPARATION', '1') == '1'
                and os.environ.get('QLT_ARITHMETIC_NATIVE_FORWARD', '1') == '1')
    assert ('qlt::apply_arithmetic_forward' in counts) == (enabled and not prepared)
    assert ('qlt::apply_arithmetic_backward_graph' in counts) == (enabled and not prepared)
    assert ('qlt::step_preparation_forward' in counts) == prepared
    assert ('qlt::step_preparation_backward_graph' in counts) == prepared
    original = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='scalar', mode='full')
    g, = torch.autograd.grad(original, sigma, create_graph=True)
    h, = torch.autograd.grad(g, sigma)
    torch.testing.assert_close((value, gradient, hessian), (original, g, h), rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize('n', [2, 7, 61])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_full_application_preserves_upstream_forward_arithmetic(n, scale):
    xs = tuple((t.detach()*scale).requires_grad_() for t in arguments(n))
    actual = apply(*xs)
    upstream = torch.ops.quantlibtorch._hook_apply(*xs, 0, False)
    torch.testing.assert_close(actual, upstream, rtol=0, atol=0)
    weights = torch.cos(torch.arange(n, dtype=DTYPE))
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(upstream, xs, weights), rtol=2e-12, atol=1e-160)
