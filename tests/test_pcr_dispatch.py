"""Preserve tensor transforms and avoid expanding shared single-RHS matrices."""
import os

import pytest
import torch

from quantlibtorch import tridiagonal_solve_pcr

DTYPE = torch.float64


def inputs(batch=3, n=5, nrhs=1):
    return (torch.full((batch, n-1), .1, dtype=DTYPE),
            torch.linspace(2., 3., batch*n, dtype=DTYPE).reshape(batch, n),
            torch.full((batch, n-1), -.2, dtype=DTYPE),
            torch.cos(torch.arange(batch*n*nrhs, dtype=DTYPE)).reshape(batch, n, nrhs))


def solve(*xs):
    return tridiagonal_solve_pcr(*xs, check_pivots=False)


def dense(*xs):
    return torch.linalg.solve(torch.diag_embed(xs[0], offset=-1)+torch.diag_embed(xs[1])
                              +torch.diag_embed(xs[2], offset=1), xs[3])


@pytest.mark.parametrize('transform', ['vmap', 'unmapped', 'grad', 'jacrev', 'jvp', 'functionalize'])
def test_pcr_preserves_function_transforms(transform):
    xs = inputs()
    single = tuple(t[0] for t in xs)
    if transform == 'vmap':
        run = lambda f: torch.vmap(f)(*xs)
    elif transform == 'unmapped':
        run = lambda f: torch.vmap(lambda w: f(*single)*w)(torch.arange(3, dtype=DTYPE))
    elif transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: f(*args).square().sum(), argnums=(0, 1, 2, 3))(*xs)
    elif transform == 'jacrev':
        run = lambda f: torch.func.jacrev(f, argnums=1)(*single)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.ones_like(t)*.2 for t in xs))
    else:
        run = lambda f: torch.func.functionalize(f)(*xs)
    torch.testing.assert_close(run(solve), run(dense), rtol=2e-11, atol=2e-12)
    # Returning from a transform must not disable the ordinary optimized path.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        solve(*single)
    used = any(e.key == 'qlt::pcr_stage_forward' for e in profile.key_averages())
    assert used == (os.environ.get('QLT_PCR_NATIVE_STAGE', '1') == '1')


def test_pcr_fake_tensor_trace_preserves_real_values_and_all_vjps():
    from torch.fx.experimental.proxy_tensor import make_fx
    xs = tuple(t[0].clone().requires_grad_() for t in inputs(nrhs=2))
    traced = make_fx(solve, tracing_mode='fake')(*xs)
    actual, expected = traced(*xs), dense(*xs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(torch.autograd.grad(actual.square().sum(), xs),
                    torch.autograd.grad(expected.square().sum(), xs)):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('pattern,nrhs,native', [
    ('shared', 1, False), ('shared', 2, True), ('one_batch', 1, True),
    ('independent', 1, True), ('cross', 1, True), ('rhs_cross', 1, False),
    ('large_independent', 1, False), ('limit_independent', 1, True), ('large_multiple', 2, True)])
def test_single_rhs_dispatch_preserves_values_and_shared_gradients(pattern, nrhs, native):
    if pattern in ('large_independent', 'large_multiple'):
        xs = inputs(batch=8193, n=4, nrhs=nrhs)
    elif pattern == 'limit_independent':
        xs = inputs(batch=8192, n=4, nrhs=nrhs)
    else:
        xs = inputs(batch=3, n=5, nrhs=nrhs)
    if pattern == 'shared':
        xs = (*[t[0] for t in xs[:3]], xs[3])
    elif pattern == 'one_batch':
        xs = (*[t[0] for t in xs[:3]], xs[3][:1])
    elif pattern == 'cross':
        xs = (xs[0][:2, None, :], xs[1][None, :, :], xs[2][0], xs[3].expand(2, *xs[3].shape))
    elif pattern == 'rhs_cross':
        xs = (*[t[:2, None, :] for t in xs[:3]], xs[3][None, ...])
    xs = tuple(t.detach().requires_grad_() for t in xs)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = solve(*xs)
        ga = torch.autograd.grad(actual.square().sum(), xs)
    expected = dense(*xs)
    ge = torch.autograd.grad(expected.square().sum(), xs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=2e-12)
    used = any(e.key == 'qlt::pcr_stage_forward' for e in profile.key_averages())
    assert used == (native and os.environ.get('QLT_PCR_NATIVE_STAGE', '1') == '1')
