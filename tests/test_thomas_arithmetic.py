"""Native elimination recurrence: dense/upstream references and full AD."""
from contextlib import nullcontext
import os

import pytest
import torch

from quantlibtorch import fd_vanilla, tridiagonal_solve

DTYPE = torch.float64


def inputs(n=3, nrhs=2, batch=(), shared=False):
    gen = torch.Generator().manual_seed(607)
    bands = () if shared else batch
    return tuple(t.requires_grad_() for t in (
        .1*torch.randn(*bands, n-1, dtype=DTYPE, generator=gen),
        2.+torch.rand(*bands, n, dtype=DTYPE, generator=gen),
        -.2*torch.randn(*bands, n-1, dtype=DTYPE, generator=gen),
        torch.randn(*batch, n, nrhs, dtype=DTYPE, generator=gen)))


def solve(*xs, native=True, checked=False):
    return torch.ops.quantlibtorch._thomas_arithmetic(*xs, checked, native)


def dense(lo, di, up, r):
    return torch.linalg.solve(torch.diag_embed(lo, offset=-1)+torch.diag_embed(di)
                              +torch.diag_embed(up, offset=1), r)


@pytest.mark.parametrize('n', [2, 3, 8, 61])
@pytest.mark.parametrize('nrhs', [1, 3])
@pytest.mark.parametrize('shared', [False, True])
def test_thomas_values_and_all_batched_band_rhs_vjps(n, nrhs, shared):
    xs = inputs(n, nrhs, (3,), shared)
    before, versions = [t.clone() for t in xs], [t._version for t in xs]
    actual, expected = solve(*xs, checked=True), dense(*xs)
    w = torch._neg_view(torch.cos(torch.arange(actual.numel(), dtype=DTYPE)).reshape_as(actual))
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(torch.autograd.grad(actual, xs, w),
                               torch.autograd.grad(expected, xs, w), rtol=2e-11, atol=2e-12)
    for t, b, version in zip(xs, before, versions):
        torch.testing.assert_close(t, b, rtol=0, atol=0)
        assert t._version == version


@pytest.mark.parametrize('view', ['strided', 'negative', 'expanded', 'cross', 'empty_batch'])
def test_thomas_views_and_broadcast_reductions(view):
    n = 5
    if view == 'strided':
        xs = tuple(torch.stack((t, t+.3), -1).flatten(-2)[..., ::2] for t in inputs(n))
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in inputs(n))
    elif view == 'expanded':
        xs = (torch.tensor(.1, dtype=DTYPE).expand(n-1), torch.tensor(2., dtype=DTYPE).expand(n),
              torch.tensor(-.2, dtype=DTYPE).expand(n-1), torch.tensor(.7, dtype=DTYPE).expand(3, n, 2))
    elif view == 'empty_batch': xs = inputs(n, 2, (0,))
    else:
        base = inputs(n, 2, (2, 3))
        xs = (base[0][:, :1], base[1][:1], base[2][0, 0], base[3])
    xs = tuple(t.detach().requires_grad_() for t in xs)
    actual, expected = solve(*xs), dense(*xs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), xs),
                               torch.autograd.grad(expected.square().sum(), xs), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('active', [(0,), (1,), (2,), (3,), (0, 1, 2, 3)])
@pytest.mark.parametrize('native', [False, True])
def test_thomas_first_and_second_differences_partial_inputs(active, native):
    xs = tuple(t.detach().requires_grad_(i in active) for i, t in enumerate(inputs()))
    run = lambda *args: solve(*args, native=native)
    assert torch.autograd.gradcheck(run, xs)
    assert torch.autograd.gradgradcheck(run, xs)


@pytest.mark.parametrize('nrhs', [1, 3])
def test_thomas_shared_aliases_and_incoming_weight_third_derivatives(nrhs):
    base = (.1*torch.cos(torch.arange(3, dtype=DTYPE))).requires_grad_()
    xs = (base[:-1], 2.+base, torch._neg_view(base[1:]), base[:, None].expand(2, 3, nrhs))
    weights = torch.sin(torch.arange(6*nrhs, dtype=DTYPE)).reshape(2, 3, nrhs).requires_grad_()
    la, le = (solve(*xs).sin()*weights).sum(), (dense(*xs).sin()*weights).sum()
    direction = (torch.cos(base.detach()), .2*torch.cos(weights.detach()))
    for order in range(3):
        ga = torch.autograd.grad(la, (base, weights), create_graph=order < 2)
        ge = torch.autograd.grad(le, (base, weights), create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)
        la = sum((g*d).sum() for g, d in zip(ga, direction))
        le = sum((g*d).sum() for g, d in zip(ge, direction))


@pytest.mark.parametrize('scale', [1e-300, 1e-170, 1e170, 1e300])
def test_thomas_extreme_scales_keep_finite_first_derivatives(scale):
    xs = tuple((t.detach()*scale).requires_grad_() for t in inputs())
    actual, expected = solve(*xs), dense(*xs)
    weights = torch.cos(torch.arange(6, dtype=DTYPE)).reshape(3, 2)
    ga, ge = torch.autograd.grad(actual, xs, weights), torch.autograd.grad(expected, xs, weights)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(ga, ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=2e-11, atol=1e-310)


@pytest.mark.parametrize('n', [2, 7, 61])
@pytest.mark.parametrize('scale', [1e-8, 1., 1e150])
def test_thomas_forward_matches_original_quantlib_bitwise(n, scale):
    xs = tuple((t.detach()*scale).requires_grad_() for t in inputs(n, 1))
    actual = solve(*xs).squeeze(-1)
    upstream = tridiagonal_solve(*xs[:3], xs[3].squeeze(-1), mode='full')
    torch.testing.assert_close(actual, upstream, rtol=0, atol=0)
    weights = torch.sin(torch.arange(n, dtype=DTYPE))
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(upstream, xs, weights), rtol=2e-11, atol=1e-160)


@pytest.mark.parametrize('active', [0, 1, 2, 3, 'all'])
@pytest.mark.parametrize('no_grad', [False, True])
def test_thomas_forward_duals_fall_back_without_losing_tangents(active, no_grad):
    from torch.autograd import forward_ad
    xs = inputs()
    directions = tuple(torch.full_like(t, .2) if active in (i, 'all') else torch.zeros_like(t)
                       for i, t in enumerate(xs))
    expected = torch.autograd.functional.jvp(dense, xs, directions)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, v) if active in (i, 'all') else t
                          for i, (t, v) in enumerate(zip(xs, directions)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                actual = forward_ad.unpack_dual(solve(*duals))
    assert not any(e.key == 'qlt::thomas_arithmetic_forward' for e in prof.key_averages())
    torch.testing.assert_close(tuple(actual), expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('transform', ['grad', 'jacrev', 'jvp', 'functionalize', 'vmap', 'unmapped'])
def test_thomas_function_transforms_use_tensor_recurrence(transform):
    xs = inputs()
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: f(*args).square().sum(), argnums=(0, 1, 2, 3))(*xs)
    elif transform == 'jacrev': run = lambda f: torch.func.jacrev(f, argnums=(0, 1, 2, 3))(*xs)
    elif transform == 'jvp': run = lambda f: torch.func.jvp(f, xs, tuple(torch.full_like(t, .2) for t in xs))
    elif transform == 'functionalize': run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap': run = lambda f: torch.vmap(f)(*(torch.stack((t, 1.1*t)) for t in xs))
    else: run = lambda f: torch.vmap(lambda w: f(*xs)*w)(torch.tensor([.4, .7], dtype=DTYPE))
    torch.testing.assert_close(run(solve), run(dense), rtol=2e-11, atol=2e-12)


def test_thomas_fake_trace_preserves_runtime_values_and_gradients():
    from torch.fx.experimental.proxy_tensor import make_fx
    xs = inputs()
    traced = make_fx(lambda *args: solve(*args), tracing_mode='fake')(*xs)
    runtime = tuple((t.detach()*1.1).requires_grad_() for t in xs)
    actual, expected = traced(*runtime), dense(*runtime)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), runtime),
                               torch.autograd.grad(expected.square().sum(), runtime), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'batch', 'nonfinite', 'first_pivot', 'later_pivot', 'overflow'])
def test_thomas_checked_probe_validation(bad):
    xs = list(inputs(2, 1))
    if bad == 'dtype': xs[0] = xs[0].float()
    elif bad == 'shape': xs[2] = xs[2][:-1]
    elif bad == 'batch': xs[0], xs[1] = xs[0].expand(2, 1), xs[1].expand(3, 2)
    elif bad == 'nonfinite': xs[3] = xs[3]*float('nan')
    elif bad == 'first_pivot': xs[1] = torch.tensor([0., 1.], dtype=DTYPE)
    elif bad == 'later_pivot': xs = [torch.ones_like(t) for t in xs]
    else: xs[1], xs[3] = torch.full_like(xs[1], .1), torch.full_like(xs[3], 1e308)
    with pytest.raises(RuntimeError): solve(*xs, checked=True)


def test_thomas_cuda_preserves_tensor_path():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    xs = tuple(t.detach().cuda().requires_grad_() for t in inputs())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        actual = solve(*xs, checked=True)
        ga = torch.autograd.grad(actual.square().sum(), xs)
    expected = dense(*xs)
    ge = torch.autograd.grad(expected.square().sum(), xs)
    assert not any(e.key == 'qlt::thomas_arithmetic_forward' for e in prof.key_averages())
    torch.testing.assert_close((actual, ga), (expected, ge), rtol=2e-11, atol=2e-12)


def test_thomas_pricing_dispatch_and_higher_derivatives():
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        actual = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='auto')
        ga, = torch.autograd.grad(actual, sigma, create_graph=True)
        ha, = torch.autograd.grad(ga, sigma)
    names = {e.key for e in prof.key_averages()}
    enabled = os.environ.get('QLT_THOMAS_ARITHMETIC', '1') == '1'
    assert ('qlt::thomas_arithmetic_forward' in names) == enabled
    backward = ('qlt::thomas_arithmetic_backward_vector' if os.environ.get('QLT_THOMAS_VECTOR_BACKWARD', '1') == '1'
                else 'qlt::thomas_arithmetic_backward_graph')
    assert (backward in names) == enabled
    expected = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='vectorized')
    ge, = torch.autograd.grad(expected, sigma, create_graph=True)
    he, = torch.autograd.grad(ge, sigma)
    torch.testing.assert_close((actual, ga, ha), (expected, ge, he), rtol=2e-9, atol=2e-9)


def state_reference(lo, di, up, rhs):
    n = di.shape[-1]
    p, t, y = [di[..., 0]], [torch.zeros_like(di[..., 0])], [rhs[..., 0, :]/di[..., 0, None]]
    for i in range(1, n):
        t.append(up[..., i-1]/p[-1])
        p.append(di[..., i]-lo[..., i-1]*t[-1])
        y.append((rhs[..., i, :]-lo[..., i-1, None]*y[-1])/p[-1][..., None])
    x = y.copy()
    for i in range(n-2, -1, -1): x[i] = y[i]-t[i+1][..., None]*x[i+1]
    return torch.stack(x, -2), torch.stack(p, -1), torch.stack(t, -1), torch.stack(y, -2)


@pytest.mark.parametrize('n', [2, 5])
@pytest.mark.parametrize('selected', [0, 1, 2, 3, 'all'])
def test_thomas_intermediate_outputs_and_all_incoming_vjps(n, selected):
    xs = inputs(n, 2, (2,))
    actual = torch.ops.quantlibtorch._thomas_arithmetic_state(*xs, False, True)
    expected = state_reference(*xs)
    torch.testing.assert_close(tuple(actual), expected, rtol=0, atol=0)
    active = tuple(i for i in range(4) if selected in (i, 'all'))
    la = sum((actual[i].sin()*torch.cos(actual[i].detach())).sum() for i in active)
    le = sum((expected[i].sin()*torch.cos(expected[i].detach())).sum() for i in active)
    ga = torch.autograd.grad(la, xs)
    ge = torch.autograd.grad(le, xs, allow_unused=True)
    ge = tuple(torch.zeros_like(x) if g is None else g for g, x in zip(ge, xs))
    torch.testing.assert_close(ga, ge, rtol=2e-11, atol=2e-12)


def test_thomas_all_state_output_second_differences_and_mixed_third_derivatives():
    xs = inputs()
    native = lambda *args: tuple(torch.ops.quantlibtorch._thomas_arithmetic_state(*args, False, True))
    assert torch.autograd.gradcheck(native, xs)
    assert torch.autograd.gradgradcheck(native, xs)
    actual, expected = native(*xs), state_reference(*xs)
    weights = tuple(torch.cos(torch.arange(t.numel(), dtype=DTYPE)).reshape_as(t).requires_grad_() for t in actual)
    la = sum((t.sin()*w).sum() for t, w in zip(actual, weights))
    le = sum((t.sin()*w).sum() for t, w in zip(expected, weights))
    variables = (*xs, *weights)
    directions = tuple(.2*torch.cos(t.detach()) for t in variables)
    for order in range(3):
        ga = torch.autograd.grad(la, variables, create_graph=order < 2)
        ge = torch.autograd.grad(le, variables, create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)
        la = sum((g*d).sum() for g, d in zip(ga, directions))
        le = sum((g*d).sum() for g, d in zip(ge, directions))


def test_thomas_forward_dual_incoming_vjp_is_preserved():
    from torch.autograd import forward_ad
    xs = inputs()
    actual, expected = solve(*xs), dense(*xs)
    weight = torch.cos(torch.arange(6, dtype=DTYPE)).reshape(3, 2)
    tangent = torch.sin(torch.arange(6, dtype=DTYPE)).reshape(3, 2)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(weight, tangent)
        ga = torch.autograd.grad(actual, xs, dual)
        ge = torch.autograd.grad(expected, xs, dual)
        for a, e in zip(ga, ge):
            pa, ta = forward_ad.unpack_dual(a)
            pe, te = forward_ad.unpack_dual(e)
            assert ta is not None and te is not None
            torch.testing.assert_close((pa, ta), (pe, te), rtol=2e-11, atol=2e-12)
