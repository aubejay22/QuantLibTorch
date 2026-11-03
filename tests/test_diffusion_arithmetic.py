"""Elementary assembly derivatives through third order and transform fallback."""
from contextlib import nullcontext
import os

import pytest
import torch

from quantlibtorch import fd_vanilla

DTYPE = torch.float64


def arguments(n=3):
    gen = torch.Generator().manual_seed(619)
    return tuple(t.requires_grad_() for t in (
        .04+.01*torch.rand(n, dtype=DTYPE, generator=gen),
        .2*torch.randn(3, n, dtype=DTYPE, generator=gen),
        .3*torch.randn(3, n, dtype=DTYPE, generator=gen),
        torch.tensor(.03, dtype=DTYPE), torch.tensor(.01, dtype=DTYPE)))


def assembly(*xs, alias=0):
    return torch.ops.quantlibtorch._hook_diffusion(*xs, False, alias)


def equation(v, dx, dxx, rate, yield_rate):
    discount = torch.stack((torch.zeros_like(rate), -rate, torch.zeros_like(rate)))[:, None]
    return (rate-yield_rate-.5*v)*dx + .5*v*dxx + discount


@pytest.mark.parametrize('n', [2, 7])
@pytest.mark.parametrize('alias', [0, 1, 2])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative'])
def test_full_assembly_values_all_vjps_aliases_and_input_preservation(n, alias, view):
    base = arguments(2*n if view == 'strided' else n)
    if view == 'strided': xs = (base[0][::2], base[1][:, ::2], base[2][:, ::2], *base[3:])
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in base)
    else: xs = base
    initial, versions = [t.clone() for t in base], [t._version for t in base]
    actual, expected = assembly(*xs, alias=alias), equation(*xs)
    weights = torch._neg_view(torch.cos(torch.arange(3*n, dtype=DTYPE)).reshape(3, n))
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(torch.autograd.grad(actual, base, weights), torch.autograd.grad(expected, base, weights)):
        torch.testing.assert_close(a, e, rtol=2e-12, atol=2e-13)
    for value, before, version in zip(base, initial, versions):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize('active', [(0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)])
def test_full_assembly_finite_first_and_second_differences(active):
    xs = tuple(t.detach().requires_grad_(i in active) for i, t in enumerate(arguments()))
    assert torch.autograd.gradcheck(assembly, xs)
    assert torch.autograd.gradgradcheck(assembly, xs)


@pytest.mark.parametrize('alias', [0, 1, 2])
def test_full_assembly_shared_parameters_and_incoming_weight_third_derivatives(alias):
    raw = (.3*torch.cos(torch.arange(9, dtype=DTYPE))).reshape(3, 3).requires_grad_()
    xs = (raw[0], raw, torch._neg_view(raw), raw[1, 0], raw[2, 0])
    weights = torch.sin(torch.arange(9, dtype=DTYPE)).reshape(3, 3).requires_grad_()
    actual, expected = assembly(*xs, alias=alias), equation(*xs)
    la, le = (actual.sin()*weights).sum(), (expected.sin()*weights).sum()
    directions = (torch.cos(raw.detach()), torch.cos(weights.detach())*.2)
    for order in range(3):
        ga = torch.autograd.grad(la, (raw, weights), create_graph=order < 2)
        ge = torch.autograd.grad(le, (raw, weights), create_graph=order < 2)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-11)
        la = sum((g*d).sum() for g, d in zip(ga, directions))
        le = sum((g*d).sum() for g, d in zip(ge, directions))


@pytest.mark.parametrize('active', [0, 1, 2, 3, 4, 'all'])
@pytest.mark.parametrize('no_grad', [False, True])
def test_full_assembly_duals_use_original_tensor_path(active, no_grad):
    from torch.autograd import forward_ad
    xs = arguments()
    directions = tuple(torch.full_like(t, .2) if active in (i, 'all') else torch.zeros_like(t)
                       for i, t in enumerate(xs))
    p, d = torch.autograd.functional.jvp(equation, xs, directions)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, v) if active in (i, 'all') else t
                          for i, (t, v) in enumerate(zip(xs, directions)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                value, tangent = forward_ad.unpack_dual(assembly(*duals))
    assert not any(e.key == 'qlt::diffusion_arithmetic_forward' for e in profile.key_averages())
    torch.testing.assert_close(value, p, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(tangent, d, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('transform', ['grad', 'jacrev', 'jvp', 'functionalize', 'vmap', 'unmapped'])
def test_full_assembly_function_transform_falls_back(transform):
    xs = arguments()
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: f(*args).square().sum(), argnums=(0, 1, 2, 3, 4))(*xs)
    elif transform == 'jacrev':
        run = lambda f: torch.func.jacrev(f, argnums=(0, 1, 2, 3, 4))(*xs)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.full_like(t, .2) for t in xs))
    elif transform == 'functionalize':
        run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap':
        run = lambda f: torch.vmap(f)(*(torch.stack((t, t*1.1)) for t in xs))
    else:
        run = lambda f: torch.vmap(lambda w: f(*xs)*w)(torch.linspace(.5, 1., 2, dtype=DTYPE))
    torch.testing.assert_close(run(assembly), run(equation), rtol=2e-12, atol=2e-13)


def test_full_assembly_fake_tensor_trace_preserves_real_values_and_vjps():
    from torch.fx.experimental.proxy_tensor import make_fx
    xs = arguments()
    traced = make_fx(assembly, tracing_mode='fake')(*xs)
    actual, expected = traced(*xs), equation(*xs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), xs),
                               torch.autograd.grad(expected.square().sum(), xs), rtol=2e-12, atol=2e-13)


def test_full_assembly_pricing_dispatch_and_higher_gradients():
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        value = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='auto', mode='full')
        gradient, = torch.autograd.grad(value, sigma, create_graph=True)
        hessian, = torch.autograd.grad(gradient, sigma)
    counts = {e.key: e.count for e in profile.key_averages()}
    enabled = os.environ.get('QLT_ASSEMBLY_ARITHMETIC', '1') == '1'
    assert ('qlt::diffusion_arithmetic_forward' in counts) == enabled
    assert ('qlt::diffusion_arithmetic_backward_graph' in counts) == enabled
    original = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='scalar', mode='full')
    g, = torch.autograd.grad(original, sigma, create_graph=True)
    h, = torch.autograd.grad(g, sigma)
    torch.testing.assert_close((value, gradient, hessian), (original, g, h), rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize('n', [2, 7, 61])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_full_assembly_preserves_upstream_forward_arithmetic(n, scale):
    xs = tuple((t.detach()*scale).requires_grad_() for t in arguments(n))
    actual = assembly(*xs)
    # Force the original QuantLib mult/axpyb, independent of adjoint switches.
    upstream = torch.ops.quantlibtorch._hook_diffusion(*xs, False, 0, 0)
    torch.testing.assert_close(actual, upstream, rtol=0, atol=0)
    weights = torch.cos(torch.arange(3*n, dtype=DTYPE)).reshape(3, n)
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(upstream, xs, weights), rtol=2e-12, atol=1e-160)
