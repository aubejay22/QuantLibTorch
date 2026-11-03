"""Immutable spatial bases may be reused without retaining pricing graphs."""
from contextlib import nullcontext
import math
import os

import pytest
import torch
from torch.autograd import forward_ad

from quantlibtorch import fd_vanilla
from test_diffusion_arithmetic import arguments, equation
from test_backward_duals import compare_incoming_duals

DTYPE = torch.float64


def counts(profile):
    return {event.key: event.count for event in profile.key_averages()}


def cache_enabled():
    return os.environ.get('QLT_BASIS_CACHE', '1') == '1' and os.environ.get('QLT_PASSIVE_CONSTANTS', '1') == '1'


@pytest.mark.parametrize('mode', ['full', 'adjoint', 'inference', 'inference_mode'])
def test_each_price_owns_basis_and_fresh_surface_graph(mode):
    recording = mode in ('full', 'adjoint')
    context = nullcontext if recording else (torch.no_grad if mode == 'inference' else torch.inference_mode)
    for nodes, spot, mesh, shift in [(7, 94., .2, 0.), (17, 108., .35, .02), (7, 101., .27, 0.)]:
        # Equal-valued fresh leaves recur after a different mesh and market.
        raw = (torch.tensor([[.21, .24, .22, .23], [.23, .25, .22, .21]], dtype=DTYPE)+shift).requires_grad_()
        field = raw[:, ::2]
        before, version = raw.clone(), raw._version
        options = dict(times=[0., .7], log_spots=[math.log(70.), math.log(145.)],
                       time_steps=4, grid_points=nodes, damping_steps=1, spot=spot,
                       mesh_volatility=mesh, mode=mode if recording else 'full')
        with context():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = fd_vanilla(field, **options)
            expected = fd_vanilla(field, kernels='scalar', **options)
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        events = counts(profile)
        if cache_enabled():
            assert events['qlt::diffusion_basis_pack'] == 1
            assert events['qlt::diffusion_basis_reuse'] >= 3
        else:
            assert events.get('qlt::diffusion_basis_reuse', 0) == 0
        if recording:
            ga, = torch.autograd.grad(actual, raw, create_graph=mode == 'full')
            ge, = torch.autograd.grad(expected, raw, create_graph=mode == 'full')
            torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-9)
            assert torch.count_nonzero(ga[:, 1::2]) == 0
            if mode == 'full':
                direction = torch.cos(raw)
                ha, = torch.autograd.grad(ga, raw, direction)
                he, = torch.autograd.grad(ge, raw, direction)
                torch.testing.assert_close(ha, he, rtol=2e-8, atol=2e-8)
        else:
            assert not actual.requires_grad
        torch.testing.assert_close(raw, before, rtol=0, atol=0)
        assert raw._version == version


def test_live_price_graphs_outlive_operators_and_backward_in_reverse_order():
    leaves = [torch.tensor(value, dtype=DTYPE, requires_grad=True) for value in (.21, .24, .21)]
    outputs = [fd_vanilla(sigma, spot=95.+i*4, time_steps=3, grid_points=11)
               for i, sigma in enumerate(leaves)]
    for i in (2, 0, 1):
        reference = fd_vanilla(leaves[i], spot=95.+i*4, time_steps=3, grid_points=11, kernels='scalar')
        torch.testing.assert_close(outputs[i], reference, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(torch.autograd.grad(outputs[i], leaves[i]),
                                   torch.autograd.grad(reference, leaves[i]), rtol=2e-9, atol=2e-9)


def test_engine_exception_releases_basis_before_next_price():
    previous_date = torch.ops.quantlibtorch._evaluation_date()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with pytest.raises(RuntimeError):
            fd_vanilla(torch.tensor(1e300, dtype=DTYPE, requires_grad=True), time_steps=3, grid_points=11)
    assert counts(profile).get('qlt::diffusion_basis_pack', 0) >= 1
    assert torch.ops.quantlibtorch._evaluation_date() == previous_date
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    actual = fd_vanilla(sigma, time_steps=3, grid_points=11)
    expected = fd_vanilla(sigma, time_steps=3, grid_points=11, kernels='scalar')
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual, sigma), torch.autograd.grad(expected, sigma),
                               rtol=2e-9, atol=2e-9)


def series(*xs, adjoint=False):
    return torch.ops.quantlibtorch._diffusion_basis_series(*xs, adjoint)


def series_reference(variances, dx, dxx, rate, yield_rate):
    return torch.stack([equation(v, dx, dxx, rate, yield_rate) for v in variances])


def series_inputs(view='contiguous'):
    v, dx, dxx, rate, yield_rate = arguments(8 if view == 'strided' else 4)
    variances = torch.stack((v, v*1.1, v*.9)).detach().requires_grad_()
    leaves = (variances, dx, dxx, rate, yield_rate)
    if view == 'strided': return leaves, (variances[:, ::2], dx[:, ::2], dxx[:, ::2], rate, yield_rate)
    if view == 'negative': return leaves, tuple(torch._neg_view(t) for t in leaves)
    if view == 'shared': return leaves[:2]+leaves[3:], (variances, dx, -dx, rate, yield_rate)
    return leaves, leaves


@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'shared'])
def test_active_immutable_bases_keep_all_vjps_and_higher_derivatives(adjoint, view):
    leaves, xs = series_inputs(view)
    originals, versions = [t.clone() for t in leaves], [t._version for t in leaves]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = series(*xs, adjoint=adjoint)
    reference = series_reference(*xs)
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)
    assert counts(profile).get('qlt::diffusion_basis_reuse', 0) == 0
    la, le = actual.sin().sum(), reference.sin().sum()
    for order in range(1 if adjoint else 3):
        ga = torch.autograd.grad(la, leaves, create_graph=not adjoint and order < 2)
        ge = torch.autograd.grad(le, leaves, create_graph=not adjoint and order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-10)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, leaves))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, leaves))
    for leaf, before, version in zip(leaves, originals, versions):
        torch.testing.assert_close(leaf, before, rtol=0, atol=0)
        assert leaf._version == version


@pytest.mark.parametrize('create_graph', [False, True])
def test_series_incoming_dual_weights(create_graph):
    leaves, xs = series_inputs()
    compare_incoming_duals(leaves, series(*xs), series_reference(*xs), create_graph=create_graph)


@pytest.mark.parametrize('transform', ['grad', 'jvp', 'functionalize', 'vmap', 'fake'])
def test_series_transform_fallback(transform):
    _, xs = series_inputs()
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: f(*args).square().sum(), argnums=(0, 1, 2, 3, 4))(*xs)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.full_like(t, .2) for t in xs))
    elif transform == 'functionalize':
        run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap':
        run = lambda f: torch.vmap(f)(*(torch.stack((t, t*1.1)) for t in xs))
    else:
        from torch.fx.experimental.proxy_tensor import make_fx
        run = lambda f: make_fx(f, tracing_mode='fake')(*xs)(*xs)
    torch.testing.assert_close(run(series), run(series_reference), rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize('no_grad', [False, True])
def test_basis_pricing_preserves_forward_duals(no_grad):
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    options = dict(time_steps=3, grid_points=11)
    reference = fd_vanilla(sigma, kernels='scalar', **options)
    gradient, = torch.autograd.grad(reference, sigma)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(sigma, torch.tensor(.7, dtype=DTYPE))
            actual, tangent = forward_ad.unpack_dual(fd_vanilla(dual, **options))
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(tangent, gradient*.7, rtol=2e-9, atol=2e-9)
