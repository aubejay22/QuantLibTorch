"""Exact predicate parity, including decision boundaries and native read views."""
import math
import os

import pytest
import torch

from quantlibtorch import fd_vanilla

DTYPE = torch.float64


def arguments(n=8):
    return (torch.zeros(n-1, dtype=DTYPE), torch.ones(n, dtype=DTYPE),
            torch.zeros(n-1, dtype=DTYPE), torch.ones(n, dtype=DTYPE))


def domain(xs, native=True):
    return torch.ops.quantlibtorch._auto_domain_check(*xs, native)


def residual(xs, x, native=True):
    return torch.ops.quantlibtorch._auto_residual_check(*xs, x, native)


@pytest.mark.parametrize('which,bound', [('small', 1e-12), ('large', 1e12), ('ratio', 1e6),
                                         ('rhs', 1e50), ('dominance', 1.-1e-7)])
@pytest.mark.parametrize('direction', [-1, 0, 1])
@pytest.mark.parametrize('negative', [False, True])
def test_domain_exact_and_adjacent_float_thresholds(which, bound, direction, negative):
    xs = list(arguments())
    value = bound if direction == 0 else math.nextafter(bound, -math.inf if direction < 0 else math.inf)
    if which in ('small', 'large'):
        xs[1].fill_(value)
    elif which == 'ratio':
        xs[1][-1] = value
    elif which == 'rhs':
        xs[3][-1] = value
    else:
        xs[0][2] = value
    if negative: xs = [torch._neg_view(t) for t in xs]
    expected = direction >= 0 if which == 'small' else direction <= 0
    assert domain(xs, False) == expected
    assert domain(xs) == expected


@pytest.mark.parametrize('n', [1, 2, 7, 8, 61])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'expanded', 'negative'])
def test_domain_views_size_and_input_preservation(n, view):
    xs = arguments(n)
    if view == 'strided': xs = tuple(torch.stack((t, t+17), dim=-1).flatten()[::2] for t in xs)
    elif view == 'expanded': xs = tuple(t[:1].expand_as(t) if t.numel() else t for t in xs)
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in xs)
    xs = tuple(t.requires_grad_() for t in xs)
    initial, versions = [t.clone() for t in xs], [t._version for t in xs]
    assert domain(xs) == domain(xs, False) == (n >= 8)
    x = torch.ones_like(xs[1])
    assert residual(xs, x) == residual(xs, x, False) == True
    for value, before, version in zip(xs, initial, versions):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize('argument', range(4))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_domain_nonfinite_rejections(argument, bad):
    xs = list(arguments())
    xs[argument][0] = bad
    assert not domain(xs, False)
    assert not domain(xs)


@pytest.mark.parametrize('n', [2, 8, 61, 257])
@pytest.mark.parametrize('scale', [1e-310, 1e-170, 1., 1e170, 1e300])
def test_residual_nextafter_decisions_match_separately_rounded_tensor_operations(n, scale):
    xs = list(arguments(n))
    xs[1].fill_(scale)
    xs[3].fill_(scale)
    x = torch.ones(n, dtype=DTYPE)
    tolerance = 64*n*torch.finfo(DTYPE).eps
    boundary = (scale*(1+tolerance)+1e-300)/(1-tolerance)
    outcomes = set()
    for displacement in range(-4, 5):
        value = boundary
        for _ in range(abs(displacement)):
            value = math.nextafter(value, -math.inf if displacement < 0 else math.inf)
        xs[3][0] = value
        expected = residual(xs, x, False)
        assert residual(xs, x) == expected
        outcomes.add(expected)
    assert outcomes == {False, True}


@pytest.mark.parametrize('argument', range(5))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_residual_nonfinite_arithmetic_retains_existing_comparisons(argument, bad):
    xs = [*arguments(), torch.ones(8, dtype=DTYPE)]
    xs[argument][0] = bad
    assert residual(xs[:4], xs[4]) == residual(xs[:4], xs[4], False)


def test_residual_random_scales_cancellation_and_readonly_strides():
    generator = torch.Generator().manual_seed(917)
    for _ in range(100):
        n = int(torch.randint(2, 20, (), generator=generator))
        scale = 10.**float(torch.empty((), dtype=DTYPE).uniform_(-300, 300, generator=generator))
        data = torch.randn(5, 2*n, dtype=DTYPE, generator=generator)
        lo, di, up, x = data[0, 2::2]*scale, data[1, ::2]*scale, data[2, :-2:2]*scale, data[3, ::2]
        zero = torch.zeros(1, dtype=DTYPE)
        rhs = di*x+torch.cat((zero, lo*x[:-1]))+torch.cat((up*x[1:], zero))
        rhs = rhs*(1+float(torch.empty((), dtype=DTYPE).uniform_(-1e-12, 1e-12, generator=generator)))
        xs = tuple(torch._neg_view(torch.stack((t, t+17), dim=-1).flatten()[::2]).requires_grad_()
                   for t in (lo, di, up, rhs))
        initial, versions = [t.clone() for t in xs], [t._version for t in xs]
        assert residual(xs, x) == residual(xs, x, False)
        for value, before, version in zip(xs, initial, versions):
            torch.testing.assert_close(value, before, rtol=0, atol=0)
            assert value._version == version


def test_predicate_reads_preserve_duals_and_pricing_vjps():
    from torch.autograd import forward_ad
    xs = tuple(t.requires_grad_() for t in arguments())
    directions = tuple(torch.full_like(t, .2) for t in xs)
    with forward_ad.dual_level():
        duals = tuple(forward_ad.make_dual(t, v) for t, v in zip(xs, directions))
        assert domain(duals) == domain(duals, False)
        x = forward_ad.make_dual(xs[3], directions[3])
        assert residual(duals, x) == residual(duals, x, False)
        for dual, tangent in zip(duals, directions):
            torch.testing.assert_close(forward_ad.unpack_dual(dual).tangent, tangent, rtol=0, atol=0)
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='auto')
        ga, = torch.autograd.grad(actual, sigma, create_graph=True)
        ha, = torch.autograd.grad(ga, sigma)
    expected = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='vectorized')
    ge, = torch.autograd.grad(expected, sigma, create_graph=True)
    he, = torch.autograd.grad(ge, sigma)
    torch.testing.assert_close((actual, ga, ha), (expected, ge, he), rtol=2e-9, atol=2e-9)
    names = {e.key for e in profile.key_averages()}
    enabled = os.environ.get('QLT_NATIVE_SOLVE_CHECKS', '1') == '1'
    assert ('qlt::auto_domain_native' in names) == enabled
    assert ('qlt::auto_residual_native' in names) == enabled


def test_cuda_predicates_retain_tensor_path():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    xs = tuple(t.cuda() for t in arguments())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        assert domain(xs) == domain(xs, False)
        assert residual(xs, xs[3]) == residual(xs, xs[3], False)
    assert not any(e.key in ('qlt::auto_domain_native', 'qlt::auto_residual_native') for e in profile.key_averages())


def test_predicates_under_function_gradient_transform_use_tensor_path():
    xs = arguments()
    def loss(di):
        values = (xs[0], di, xs[2], xs[3])
        assert domain(values)
        assert residual(values, xs[3])
        return di.square().sum()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        gradient = torch.func.grad(loss)(xs[1])
    torch.testing.assert_close(gradient, 2*xs[1])
    assert not any(e.key in ('qlt::auto_domain_native', 'qlt::auto_residual_native') for e in profile.key_averages())
