"""Repeated spatial interpolation retains every coefficient dependency."""
from contextlib import nullcontext
import math

import pytest
import torch

from quantlibtorch import fd_vanilla

DTYPE = torch.float64


@pytest.mark.parametrize('nt,nx', [(1, 1), (1, 5), (3, 1), (3, 5)])
@pytest.mark.parametrize('mode', ['full', 'adjoint'])
def test_cached_sampling_strided_surface_and_time_edges_match_scalar(nt, nx, mode):
    base = (.21+.02*torch.sin(torch.arange(nt*nx*2, dtype=DTYPE))).reshape(nt, nx*2).requires_grad_()
    field = base[:, ::2]
    args = dict(times=[.2] if nt == 1 else [.05, .42, .7],
                log_spots=[math.log(100.)] if nx == 1 else [math.log(s) for s in (40., 70., 100., 150., 250.)],
                time_steps=8, grid_points=21, spot=103., mode=mode)
    actual = fd_vanilla(field, kernels='auto', **args)
    expected = fd_vanilla(field, kernels='scalar', **args)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    ga, = torch.autograd.grad(actual, base)
    ge, = torch.autograd.grad(expected, base)
    torch.testing.assert_close(ga, ge, rtol=2e-9, atol=1e-9)
    assert torch.count_nonzero(ga[:, 1::2]) == 0
    # A later call with different coefficients must construct its own cache.
    with torch.no_grad():
        changed = field+.01
        next_value = fd_vanilla(changed, **args)
        reference = fd_vanilla(changed, kernels='scalar', **args)
    assert not next_value.requires_grad
    torch.testing.assert_close(next_value, reference, rtol=2e-12, atol=2e-12)
    assert abs((next_value-actual).item()) > 1e-4


@pytest.mark.parametrize('no_grad', [False, True])
def test_cached_nonuniform_forward_ad_and_second_derivatives(no_grad):
    from torch.autograd import forward_ad
    field = torch.tensor([[.19, .22], [.24, .21]], dtype=DTYPE, requires_grad=True)
    direction = torch.tensor([[.3, -.2], [.1, .4]], dtype=DTYPE)
    def price(sigma):
        return fd_vanilla(sigma, times=[.1, .65], log_spots=[math.log(75.), math.log(145.)],
                          time_steps=4, grid_points=11, spot=103., damping_steps=1)
    value = price(field)
    gradient, = torch.autograd.grad(value, field)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(field, direction)
            primal, tangent = forward_ad.unpack_dual(price(dual))
    torch.testing.assert_close(primal, value, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(tangent, (gradient*direction).sum(), rtol=2e-10, atol=2e-10)
    if not no_grad:
        assert torch.autograd.gradgradcheck(price, (field,), fast_mode=True)


def test_pricing_reuses_each_spatial_row_within_its_graph():
    field = (.21+.02*torch.sin(torch.arange(15, dtype=DTYPE))).reshape(3, 5).requires_grad_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        value = fd_vanilla(field, times=[0., .37, 1.],
            log_spots=[math.log(s) for s in (40., 70., 100., 150., 250.)],
            time_steps=12, grid_points=21, mode='adjoint')
    counts = {item.key: item.count for item in profile.key_averages()}
    assert counts['qlt::sample_grid'] > 12
    assert counts['qlt::spatial_row'] == 3
    gradient, = torch.autograd.grad(value, field)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient.sum(1)) == 3
