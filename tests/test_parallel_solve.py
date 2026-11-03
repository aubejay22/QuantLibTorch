"""Independent dense/finite-difference checks for the optional PCR kernel."""
from contextlib import nullcontext
from datetime import date
import math
import os

import pytest
import torch

from quantlibtorch import tridiagonal_solve_pcr, fd_vanilla
from test_fd_engine import stock, MARKETS

DTYPE = torch.float64


def dense(lower, diagonal, upper):
    return torch.diag_embed(lower, offset=-1)+torch.diag_embed(diagonal)+torch.diag_embed(upper, offset=1)


@pytest.mark.parametrize("n", [2, 3, 8, 17])
@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("shared", [False, True])
def test_pcr_batched_values_and_all_band_rhs_vjps(n, r, shared):
    g = torch.Generator().manual_seed(157)
    shape = () if shared else (3,)
    args = tuple(t.requires_grad_() for t in (
        .1*torch.randn(*shape, n-1, dtype=DTYPE, generator=g),
        2+torch.rand(*shape, n, dtype=DTYPE, generator=g),
        -.2*torch.randn(*shape, n-1, dtype=DTYPE, generator=g),
        torch.randn(3, n, r, dtype=DTYPE, generator=g)))
    before = [t.detach().clone() for t in args]
    actual = tridiagonal_solve_pcr(*args)
    expected = torch.linalg.solve(dense(*args[:3]), args[3])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    w = torch.randn(actual.shape, dtype=DTYPE, generator=g)
    for a, b in zip(torch.autograd.grad(actual, args, w), torch.autograd.grad(expected, args, w)):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=1e-12)
    for t, original in zip(args, before):
        torch.testing.assert_close(t, original, rtol=0, atol=0)


def test_pcr_cross_broadcast_strided_and_expanded_views():
    g = torch.Generator().manual_seed(811)
    n = 7
    args = tuple(t.requires_grad_() for t in (
        (.1*torch.rand(2, 1, 2*(n-1), dtype=DTYPE, generator=g))[..., ::2],
        (2+torch.rand(1, 3, 2*n, dtype=DTYPE, generator=g))[..., ::2],
        torch.tensor(-.15, dtype=DTYPE).expand(n-1),
        torch.randn(2, 3, 2*n, 4, dtype=DTYPE, generator=g)[..., ::2, ::2]))
    actual = tridiagonal_solve_pcr(*args)
    expected = torch.linalg.solve(dense(*args[:3]), args[3])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    w = torch.randn(actual.shape, dtype=DTYPE, generator=g)
    for a, b in zip(torch.autograd.grad(actual, args, w), torch.autograd.grad(expected, args, w)):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)


def test_pcr_first_second_differences_and_checked_unchecked_agreement():
    args = tuple(t.requires_grad_() for t in (
        torch.tensor([.1, -.2], dtype=DTYPE), torch.tensor([2., 3., 2.5], dtype=DTYPE),
        torch.tensor([-.15, .12], dtype=DTYPE), torch.tensor([[1., .2], [-.3, .7], [.1, -.8]], dtype=DTYPE)))
    assert torch.autograd.gradcheck(tridiagonal_solve_pcr, args)
    assert torch.autograd.gradgradcheck(tridiagonal_solve_pcr, args)
    torch.testing.assert_close(tridiagonal_solve_pcr(*args), tridiagonal_solve_pcr(*args, check_pivots=False), rtol=0, atol=0)


def test_pcr_pivot_policy_differs_from_thomas():
    off = torch.tensor([1.], dtype=DTYPE)
    diag = torch.tensor([1., 0.], dtype=DTYPE)
    rhs = torch.tensor([[.2], [.5]], dtype=DTYPE)
    # Invertible and Thomas-admissible, but PCR eliminates both rows together.
    assert torch.isfinite(torch.linalg.solve(dense(off, diag, off), rhs)).all()
    with pytest.raises(RuntimeError, match="PCR.*pivots"):
        tridiagonal_solve_pcr(off, diag, off, rhs)
    assert not torch.isfinite(tridiagonal_solve_pcr(off, diag, off, rhs, check_pivots=False)).all()
    with pytest.raises(RuntimeError, match="PCR.*pivots"):
        tridiagonal_solve_pcr(off, torch.ones_like(diag), off, rhs)


@pytest.mark.parametrize("kind", ["dtype", "shape", "batch", "nonfinite", "rhs"])
def test_pcr_input_validation(kind):
    lo, di, up, rhs = torch.ones(3, dtype=DTYPE)*.1, torch.ones(4, dtype=DTYPE)*2, torch.ones(3, dtype=DTYPE)*-.1, torch.ones(4, 1, dtype=DTYPE)
    if kind == "dtype": lo = lo.float()
    if kind == "shape": up = up[:2]
    if kind == "batch": lo, di = lo.expand(2, 3), di.expand(3, 4)
    if kind == "nonfinite": rhs[1] = float('nan')
    if kind == "rhs": rhs = rhs[:, 0]
    with pytest.raises(RuntimeError):
        tridiagonal_solve_pcr(lo, di, up, rhs)


def test_pcr_scaled_quantlib_hook_values_vjps_and_higher_derivatives():
    args = tuple(t.requires_grad_() for t in (
        torch.tensor([.1, -.2], dtype=DTYPE), torch.tensor([2., 3., 2.5], dtype=DTYPE),
        torch.tensor([-.15, .12], dtype=DTYPE), torch.tensor([1., -.3, .1], dtype=DTYPE),
        torch.tensor(.2, dtype=DTYPE), torch.tensor(1.3, dtype=DTYPE)))
    f = lambda *xs: torch.ops.quantlibtorch._band_solve(*xs, False, 3)
    matrix = args[4]*dense(*args[:3])+args[5]*torch.eye(3, dtype=DTYPE)
    ref = torch.linalg.solve(matrix, args[3])
    value = f(*args)
    torch.testing.assert_close(value, ref, rtol=1e-13, atol=1e-13)
    w = torch.tensor([.4, -.1, .3], dtype=DTYPE)
    for a, b in zip(torch.autograd.grad(value, args, w), torch.autograd.grad(ref, args, w)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradgradcheck(f, args)


@pytest.mark.parametrize("market", MARKETS)
@pytest.mark.parametrize("scheme", ["douglas", "implicit_euler", "crank_nicolson"])
def test_pcr_engine_prices_and_vega_against_stock(market, scheme):
    args = dict(time_steps=8, grid_points=21, damping_steps=2, scheme=scheme, **market)
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    value = fd_vanilla(sigma, mode="full", kernels="pcr", **args)
    gradient, = torch.autograd.grad(value, sigma)
    reference = stock(volatility=.23, bump=1e-5, **args)
    assert value.item() == pytest.approx(reference['price'], abs=2e-10)
    assert gradient.item() == pytest.approx(reference['parallel_vol_shift_derivative'], rel=3e-6, abs=2e-6)


@pytest.mark.parametrize("exercise", ["european", "american", "bermudan"])
def test_pcr_surface_event_vjp_and_directional_difference(exercise):
    field = (.21+.025*torch.sin(torch.arange(15, dtype=DTYPE))).reshape(3, 5).requires_grad_()
    mid, end = date(2025, 7, 1), date(2026, 1, 1)
    args = dict(times=[0., .37, 1.], log_spots=[math.log(s) for s in (40., 70., 100., 140., 250.)],
                exercise=exercise, option_type="put", spot=96., cash_dividends=[(mid, 1.7)], time_steps=6, grid_points=21)
    if exercise == 'bermudan': args['exercise_dates'] = [mid, end]
    actual = fd_vanilla(field, kernels='pcr', **args)
    expected = fd_vanilla(field, kernels='vectorized', **args)
    ga, = torch.autograd.grad(actual, field)
    ge, = torch.autograd.grad(expected, field)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ga, ge, rtol=2e-9, atol=1e-9)
    direction = torch.cos(torch.arange(15, dtype=DTYPE)).reshape_as(field)
    with torch.no_grad():
        difference = (fd_vanilla(field+1e-5*direction, kernels='pcr', **args)
                      - fd_vanilla(field-1e-5*direction, kernels='pcr', **args))/(2e-5)
    torch.testing.assert_close((ga*direction).sum(), difference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("no_grad", [False, True])
@pytest.mark.parametrize("kernels", ["pcr", "auto"])
def test_pcr_engine_forward_ad_and_full_second_derivatives(no_grad, kernels):
    from torch.autograd import forward_ad
    sigma = torch.tensor(.21, dtype=DTYPE, requires_grad=True)
    f = lambda s: fd_vanilla(s, spot=103., time_steps=2, grid_points=11, damping_steps=0, kernels=kernels)
    value = f(sigma)
    gradient, = torch.autograd.grad(value, sigma)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(sigma, torch.ones_like(sigma)*.4)
            primal, tangent = forward_ad.unpack_dual(f(dual))
    torch.testing.assert_close(primal, value)
    torch.testing.assert_close(tangent, .4*gradient, rtol=2e-10, atol=1e-10)
    if not no_grad:
        assert torch.autograd.gradgradcheck(f, (sigma,))
    with pytest.raises(ValueError, match="mode='full'"):
        fd_vanilla(sigma, kernels='pcr', mode='adjoint')


@pytest.mark.parametrize("kind", ["dominant", "small", "zero_diagonal", "tiny", "large", "row_scale", "weak_dominance"])
@pytest.mark.parametrize("hook", ["split", "tridiagonal_alias"])
def test_auto_selection_and_upstream_fallback_preserve_all_vjps(kind, hook):
    n = 5 if kind == 'small' else 11
    lo = torch.full((n-1,), .1, dtype=DTYPE)
    up = torch.full((n-1,), -.2, dtype=DTYPE)
    di = torch.linspace(2., 3., n, dtype=DTYPE)
    if kind == 'zero_diagonal': di[-1] = 0.  # Thomas-admissible, PCR is not.
    if kind == 'tiny': lo, di, up = lo*1e-14, di*1e-14, up*1e-14
    if kind == 'large': lo, di, up = lo*1e14, di*1e14, up*1e14
    if kind == 'row_scale': di[-1] = 1e8
    if kind == 'weak_dominance': di = torch.full_like(di, .3)
    args = tuple(t.requires_grad_() for t in (
        lo, di, up, torch.cos(torch.arange(n, dtype=DTYPE)),
        torch.tensor(1., dtype=DTYPE), torch.tensor(0., dtype=DTYPE)))
    if hook == 'split':
        run = lambda mode: torch.ops.quantlibtorch._band_solve(*args, False, mode)
        inputs = args
    else:
        run = lambda mode: torch.ops.quantlibtorch._tridiagonal_hook(*args[:4], False, True, mode)
        inputs = args[:4]
    # Verify which algorithm actually executes as well as its result. Timing
    # measurements live in separate workers and never use the profiler.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = run(4)
    thomas = os.environ.get('QLT_THOMAS_ARITHMETIC', '1') == '1'
    names = {e.key for e in profile.key_averages()}
    assert ('qlt::pcr_solve' in names) == (kind == 'dominant' and not thomas)
    assert ('qlt::thomas_arithmetic_forward' in names) == (kind == 'dominant' and thomas)
    expected = run(2)
    reference = torch.linalg.solve(dense(lo, di, up), args[3])
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=1e-13)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=1e-13)
    weights = torch.linspace(-.5, .4, n, dtype=DTYPE)
    for a, b in zip(torch.autograd.grad(actual, inputs, weights), torch.autograd.grad(expected, inputs, weights)):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=1e-12)


def test_auto_split_second_derivatives_and_invalid_upstream_pivot():
    n = 8
    args = tuple(t.requires_grad_() for t in (
        torch.full((n-1,), .1, dtype=DTYPE), torch.linspace(2., 3., n, dtype=DTYPE),
        torch.full((n-1,), -.2, dtype=DTYPE), torch.cos(torch.arange(n, dtype=DTYPE)),
        torch.tensor(.3, dtype=DTYPE), torch.tensor(1.2, dtype=DTYPE)))
    f = lambda *xs: torch.ops.quantlibtorch._band_solve(*xs, False, 4)
    assert torch.autograd.gradgradcheck(f, args, fast_mode=True)
    bad = list(t.detach().clone() for t in args)
    bad[1][0] = 0.
    bad[4], bad[5] = torch.tensor(1., dtype=DTYPE), torch.tensor(0., dtype=DTYPE)
    for mode in (2, 4):
        with pytest.raises(RuntimeError):
            torch.ops.quantlibtorch._band_solve(*bad, False, mode)


def test_cuda_pcr_matches_independent_cpu_dense_vjps():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    g = torch.Generator().manual_seed(51)
    args = tuple(t.requires_grad_() for t in (
        torch.rand(17-1, dtype=DTYPE, generator=g)*.1,
        2+torch.rand(3, 17, dtype=DTYPE, generator=g),
        torch.rand(17-1, dtype=DTYPE, generator=g)*-.2,
        torch.randn(3, 17, 2, dtype=DTYPE, generator=g)))
    device_args = tuple(t.detach().cuda().requires_grad_() for t in args)
    actual = tridiagonal_solve_pcr(*device_args)
    expected = torch.linalg.solve(dense(*args[:3]), args[3])
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-12, atol=2e-13)
    w = torch.randn(expected.shape, dtype=DTYPE, generator=g)
    for a, b in zip(torch.autograd.grad(actual, device_args, w.cuda()), torch.autograd.grad(expected, args, w)):
        torch.testing.assert_close(a.cpu(), b, rtol=2e-11, atol=2e-12)
