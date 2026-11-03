"""Native first-order forwards against forced original QuantLib operations."""
from contextlib import nullcontext
import os

import pytest
import torch

from quantlibtorch import fd_vanilla

DTYPE = torch.float64


@pytest.mark.parametrize('kind', ['assembly', 'apply'])
@pytest.mark.parametrize('n', [2, 7, 61])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_adjoint_forward_preserves_upstream_arithmetic_and_vjps(kind, n, scale):
    gen = torch.Generator().manual_seed(922)
    if kind == 'assembly':
        values = (torch.rand(n, dtype=DTYPE, generator=gen),
                  torch.randn(3, n, dtype=DTYPE, generator=gen),
                  torch.randn(3, n, dtype=DTYPE, generator=gen),
                  torch.tensor(.03, dtype=DTYPE), torch.tensor(.01, dtype=DTYPE))
        xs = tuple((t*scale).requires_grad_() for t in values)
        actual = torch.ops.quantlibtorch._hook_diffusion(*xs, True, 0)
        expected = torch.ops.quantlibtorch._hook_diffusion(*xs, False, 0, 0)
    else:
        values = (torch.randn(3, n, dtype=DTYPE, generator=gen),
                  torch.randn(n, dtype=DTYPE, generator=gen))
        xs = tuple((t*scale).requires_grad_() for t in values)
        actual = torch.ops.quantlibtorch._hook_apply(*xs, 4, True)
        expected = torch.ops.quantlibtorch._hook_apply(*xs, 0, False)
    assert torch.equal(actual.view(torch.int64), expected.view(torch.int64))
    weights = torch.cos(torch.arange(actual.numel(), dtype=DTYPE)).reshape_as(actual)
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(expected, xs, weights), rtol=2e-12, atol=1e-160)


@pytest.mark.parametrize('inference', [False, True])
def test_adjoint_arithmetic_financial_dispatch_and_original_reference(inference):
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    with torch.no_grad() if inference else nullcontext():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            actual = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='auto',
                                mode='full' if inference else 'adjoint')
            if not inference:
                ga, = torch.autograd.grad(actual, sigma)
        expected = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='scalar', mode='full')
        if not inference:
            ge, = torch.autograd.grad(expected, sigma)
            torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-9)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    names = {event.key for event in prof.key_averages()}
    assert 'qlt::diffusion_adjoint_forward' in names
    enabled = os.environ.get('QLT_ADJOINT_ARITHMETIC_FORWARD', '1') == '1'
    assert ('qlt::diffusion_arithmetic_native_forward' in names) == enabled
    if inference:
        assert not actual.requires_grad
