"""Independent dense operators and differences for batched numerical kernels."""
from datetime import date
import math

import pytest
import torch

from quantlibtorch import band_axpyb, fd_vanilla, tridiagonal_apply

DTYPE = torch.float64


def random(shape, seed, device="cpu"):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=DTYPE).to(device)


def dense(lower, diagonal, upper):
    return torch.diag_embed(diagonal) + torch.diag_embed(lower, offset=-1) + torch.diag_embed(upper, offset=1)


@pytest.mark.parametrize("n,r", [(2, 1), (7, 4)])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
@pytest.mark.parametrize("shared", [False, True])
def test_batched_apply_dense_vjps(n, r, batch, shared):
    bands_batch = () if shared else batch
    shapes = [bands_batch+(n-1,), bands_batch+(n,), bands_batch+(n-1,), batch+(n, r)]
    args = tuple(random(s, i).requires_grad_() for i, s in enumerate(shapes))
    before = [x.detach().clone() for x in args]
    result = tridiagonal_apply(*args)
    expected = dense(*args[:3]) @ args[3]
    weights = random(result.shape, 73)
    torch.testing.assert_close(result, expected, atol=2e-14, rtol=2e-14)
    for g, ref in zip(torch.autograd.grad(result, args, weights), torch.autograd.grad(expected, args, weights)):
        torch.testing.assert_close(g, ref, atol=2e-14, rtol=2e-14)
    for x, original in zip(args, before):
        torch.testing.assert_close(x, original, atol=0, rtol=0)


def test_noncontiguous_cross_broadcast_and_expanded_coefficients():
    n, r = 5, 3
    # Two independent leading axes broadcast, with strided bands/RHS.
    bases = [random((2, 1, 2*(n-1)), 1).requires_grad_(),
             random((n,), 2).requires_grad_(),
             random((1, 3, 2*(n-1)), 3).requires_grad_(),
             random((3, r, n), 4).requires_grad_()]
    lo, di, up, rhs = bases[0][..., ::2], bases[1].expand(2, 1, n), bases[2][..., ::2], bases[3].transpose(-1, -2)
    out = tridiagonal_apply(lo, di, up, rhs)
    ref = dense(lo, di, up) @ rhs
    w = random(out.shape, 83)
    torch.testing.assert_close(out, ref)
    for g, expected in zip(torch.autograd.grad(out, bases, w), torch.autograd.grad(ref, bases, w)):
        torch.testing.assert_close(g, expected)


def test_batched_apply_first_and_second_differences():
    args = tuple(random(s, i).requires_grad_() for i, s in enumerate([(2,), (3,), (2,), (2, 3, 2)]))
    assert torch.autograd.gradcheck(tridiagonal_apply, args)
    assert torch.autograd.gradgradcheck(tridiagonal_apply, args)


def test_batched_assembly_dense_vjp_and_differences():
    n = 4
    shapes = [(2, 1, 1), (1, 3, 3, n), (2, 1, 3, n), (n,)]
    args = tuple(random(s, i).requires_grad_() for i, s in enumerate(shapes))
    a, x, y, b = args
    out = band_axpyb(*args)
    # Map row-packed storage to an independent dense matrix; endpoint entries
    # do not participate in a compact tridiagonal operator.
    result = dense(out[..., 0, 1:], out[..., 1, :], out[..., 2, :-1])
    dx = dense(x[..., 0, 1:], x[..., 1, :], x[..., 2, :-1])
    dy = dense(y[..., 0, 1:], y[..., 1, :], y[..., 2, :-1])
    expected = dy + a.unsqueeze(-1)*dx + torch.diag_embed(b)
    w = random(result.shape, 88)
    torch.testing.assert_close(result, expected)
    for g, ref in zip(torch.autograd.grad(result, args, w), torch.autograd.grad(expected, args, w)):
        torch.testing.assert_close(g, ref)
    small = tuple(random(s, i).requires_grad_() for i, s in enumerate([(1,), (3, 2), (3, 2), (2,)]))
    assert torch.autograd.gradcheck(band_axpyb, small)
    assert torch.autograd.gradgradcheck(band_axpyb, small)


@pytest.mark.parametrize("n", [2, 9])
def test_upstream_application_with_nonzero_reflected_endpoints(n):
    bands, rhs = random((3, n), 12).requires_grad_(), random((n,), 13).requires_grad_()
    rows = torch.arange(n)
    left = torch.cat((torch.tensor([1]), torch.arange(n-1)))
    right = torch.cat((torch.arange(1, n), torch.tensor([n-2])))
    matrix = torch.zeros(n, n, dtype=DTYPE).index_put((rows, left), bands[0], accumulate=True)
    matrix = matrix.index_put((rows, rows), bands[1], accumulate=True)
    matrix = matrix.index_put((rows, right), bands[2], accumulate=True)
    expected = matrix @ rhs
    w = random((n,), 14)
    reference = torch.autograd.grad(expected, (bands, rhs), w)
    for mode in (0, 1, 2):
        result = torch.ops.quantlibtorch._hook_apply(bands, rhs, mode)
        torch.testing.assert_close(result, expected)
        for g, ref in zip(torch.autograd.grad(result, (bands, rhs), w), reference):
            torch.testing.assert_close(g, ref)


@pytest.mark.parametrize("na,nb", [(0, 0), (0, 1), (0, 5), (1, 0), (5, 0), (1, 1), (5, 5)])
@pytest.mark.parametrize("alias", [0, 1, 2])
def test_upstream_assembly_scales_and_aliases(na, nb, alias):
    args = tuple(random(s, i).requires_grad_() for i, s in enumerate([(na,), (3, 5), (3, 5), (nb,), (5,)]))
    snapshots = [v.detach().clone() for v in args]
    scalar = torch.ops.quantlibtorch._hook_assembly(*args, 0, alias)
    vector = torch.ops.quantlibtorch._hook_assembly(*args, 2, alias)
    torch.testing.assert_close(vector, scalar, atol=1e-14, rtol=1e-14)
    w = random(vector.shape, 44)
    for g, ref in zip(torch.autograd.grad(vector, args, w, allow_unused=True),
                      torch.autograd.grad(scalar, args, w, allow_unused=True)):
        if ref is None:
            assert g is None
        else:
            torch.testing.assert_close(g, ref)
    for value, original in zip(args, snapshots):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
@pytest.mark.parametrize("exercise,scheme", [("european", "douglas"), ("american", "implicit_euler"),
                                             ("bermudan", "crank_nicolson")])
def test_engine_scalar_vectorized_event_surface_vjp(mode, exercise, scheme):
    surface = (.21+.025*random((3, 5), 21)).requires_grad_()
    mid, expiry = date(2025, 7, 1), date(2026, 1, 1)
    kwargs = dict(times=[0., .37, 1.], log_spots=[math.log(s) for s in (40., 70., 100., 140., 250.)],
                  exercise=exercise, option_type="put", spot=96., cash_dividends=[(mid, 1.7)],
                  time_steps=6, grid_points=21, mode=mode, scheme=scheme)
    if exercise == "bermudan":
        kwargs["exercise_dates"] = [mid, expiry]
    results = [fd_vanilla(surface, kernels=k, **kwargs) for k in ("scalar", "apply", "vectorized")]
    gradients = [torch.autograd.grad(p, surface)[0] for p in results]
    for p, g in zip(results[1:], gradients[1:]):
        torch.testing.assert_close(p, results[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(g, gradients[0], rtol=2e-9, atol=1e-9)
    direction = random(surface.shape, 99)
    h = 1e-5
    with torch.no_grad():
        fd = (fd_vanilla(surface+h*direction, **kwargs) - fd_vanilla(surface-h*direction, **kwargs))/(2*h)
    torch.testing.assert_close((gradients[-1]*direction).sum(), fd, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("case", ["dtype", "rank", "nodes", "batch", "rhs", "sparse"])
def test_invalid_apply_inputs(case):
    lo, di, up, rhs = torch.ones(3, dtype=DTYPE), torch.ones(4, dtype=DTYPE), torch.ones(3, dtype=DTYPE), torch.ones(4, 2, dtype=DTYPE)
    if case == "dtype": lo = lo.float()
    if case == "rank": rhs = rhs[:, 0]
    if case == "nodes": up = torch.ones(4, dtype=DTYPE)
    if case == "batch": lo, rhs = lo.expand(2, 3), rhs.expand(3, 4, 2)
    if case == "rhs": rhs = rhs[:, :0]
    if case == "sparse": lo = lo.to_sparse()
    with pytest.raises(RuntimeError):
        tridiagonal_apply(lo, di, up, rhs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batched_kernels_match_dense_cpu_values_and_vjps():
    cpu = tuple(random(s, i).requires_grad_() for i, s in enumerate([(4, 1), (3, 9), (4, 3, 9), (9,), (4, 9, 3)]))
    gpu = tuple(t.detach().cuda().requires_grad_() for t in cpu)
    a, x, y, b, rhs = cpu
    dx = dense(x[0, 1:], x[1], x[2, :-1])
    dy = dense(y[:, 0, 1:], y[:, 1], y[:, 2, :-1])
    expected = (dy + a.unsqueeze(-1)*dx + torch.diag_embed(b)) @ rhs
    bands = band_axpyb(*gpu[:4])
    actual = tridiagonal_apply(bands[:, 0, 1:], bands[:, 1], bands[:, 2, :-1], gpu[-1])
    w = random(expected.shape, 121)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-12, atol=1e-12)
    for g, ref in zip(torch.autograd.grad(actual, gpu, w.cuda()), torch.autograd.grad(expected, cpu, w)):
        torch.testing.assert_close(g.cpu(), ref, rtol=1e-12, atol=1e-12)
    with pytest.raises(RuntimeError, match="same CPU or CUDA device"):
        tridiagonal_apply(gpu[1][0, 1:], gpu[1][1], gpu[1][2, :-1], cpu[-1])
