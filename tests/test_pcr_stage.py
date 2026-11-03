"""Independent recurrence, derivatives and dispatch checks for PCR stage fusion."""
from contextlib import nullcontext
import io
import os

import pytest
import torch

from quantlibtorch import fd_vanilla, tridiagonal_solve_pcr

DTYPE = torch.float64


def equations(lo, di, up, rhs, step):
    shape = torch.broadcast_shapes(lo.shape, di.shape, up.shape, rhs.shape[:-1])
    lo, di, up = (t.expand(shape) for t in (lo, di, up))
    rhs = rhs.expand(*shape, rhs.shape[-1])
    n = di.shape[-1]
    left, right = (torch.arange(n, device=di.device)-step) % n, (torch.arange(n, device=di.device)+step) % n
    a, b = -lo/di[..., left], -up/di[..., right]
    return (a*lo[..., left], di+a*up[..., left]+b*lo[..., right], b*up[..., right],
            rhs+a[..., None]*rhs[..., left, :]+b[..., None]*rhs[..., right, :])


def arguments(n=3, nrhs=2, width=None):
    width = n if width is None else width
    gen = torch.Generator().manual_seed(923)
    return tuple(t.requires_grad_() for t in (
        .1*torch.randn(width, dtype=DTYPE, generator=gen),
        2+torch.rand(width, dtype=DTYPE, generator=gen),
        -.2*torch.randn(width, dtype=DTYPE, generator=gen),
        torch.randn(width, nrhs, dtype=DTYPE, generator=gen)))


def stage(inputs, step, native=True):
    return tuple(torch.ops.quantlibtorch._pcr_stage(*inputs, step, native))


@pytest.mark.parametrize('n,step', [(2, 1), (3, 2), (8, 1), (8, 4), (17, 8), (17, 16)])
@pytest.mark.parametrize('nrhs', [1, 3])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative'])
def test_stage_values_every_vjp_and_input_preservation(n, step, nrhs, view):
    base = arguments(n, nrhs, 2*n if view == 'strided' else n)
    if view == 'strided': inputs = tuple(t[::2] for t in base)
    elif view == 'negative': inputs = tuple(torch._neg_view(t) for t in base)
    else: inputs = base
    initial = [t.detach().clone() for t in base]
    versions = [t._version for t in base]
    actual, expected = stage(inputs, step), equations(*inputs, step)
    weights = tuple(torch._neg_view(torch.cos(torch.arange(t.numel(), dtype=DTYPE)).reshape_as(t)) for t in actual)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-13, atol=2e-14)
    for ga, ge in zip(torch.autograd.grad(actual, base, weights), torch.autograd.grad(expected, base, weights)):
        torch.testing.assert_close(ga, ge, rtol=2e-12, atol=2e-13)
    for value, before, version in zip(base, initial, versions):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize('active', [(0,), (1,), (2,), (3,), (0, 1, 2, 3)])
def test_stage_partial_gradients_and_differences(active):
    inputs = tuple(t.detach().requires_grad_(i in active) for i, t in enumerate(arguments()))
    assert torch.autograd.gradcheck(lambda *args: stage(args, 1), inputs)
    assert torch.autograd.gradgradcheck(lambda *args: stage(args, 1), inputs)


@pytest.mark.parametrize('output', [0, 1, 2, 3])
def test_stage_unused_outputs_materialize_correct_zero_vjps(output):
    inputs = arguments()
    actual = stage(inputs, 1)[output]
    expected = equations(*inputs, 1)[output]
    ga = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    ge = torch.autograd.grad(expected.sum(), inputs, allow_unused=True)
    for a, e, original in zip(ga, ge, inputs):
        torch.testing.assert_close(torch.zeros_like(original) if a is None else a,
                                   torch.zeros_like(original) if e is None else e, rtol=2e-12, atol=2e-13)


def test_stage_shared_inputs_and_third_derivatives():
    raw = torch.cos(torch.arange(12, dtype=DTYPE)).reshape(3, 4).requires_grad_()
    inputs = (raw[:, 0], 3+raw[:, 1], raw[:, 2], raw[:, ::2])
    actual, expected = stage(inputs, 2), equations(*inputs, 2)
    la = sum((i+1)*value.square().sum() for i, value in enumerate(actual))
    le = sum((i+1)*value.square().sum() for i, value in enumerate(expected))
    direction = torch.sin(torch.arange(12, dtype=DTYPE)).reshape_as(raw)
    for order in range(3):
        ga, = torch.autograd.grad(la, raw, create_graph=order < 2)
        ge, = torch.autograd.grad(le, raw, create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-11, atol=2e-11)
        la, le = (ga*direction).sum(), (ge*direction).sum()


@pytest.mark.parametrize('scale', [1e-170, 1e170])
def test_stage_quotient_vjp_avoids_squared_denominator_overflow(scale):
    inputs = tuple((t.detach()*scale if i < 3 else t.detach()).requires_grad_()
                   for i, t in enumerate(arguments()))
    actual, expected = stage(inputs, 1), equations(*inputs, 1)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2e-12, atol=1e-190)
    for ga, ge in zip(torch.autograd.grad(sum(t.sum() for t in actual), inputs),
                      torch.autograd.grad(sum(t.sum() for t in expected), inputs)):
        torch.testing.assert_close(ga, ge, rtol=2e-12, atol=1e-190)


@pytest.mark.parametrize('active', [0, 1, 2, 3, 'all'])
@pytest.mark.parametrize('no_grad', [False, True])
def test_stage_forward_ad_falls_back_without_losing_tangents(active, no_grad):
    from torch.autograd import forward_ad
    inputs = arguments(4, 2)
    directions = tuple(torch.ones_like(t)*.2 if active in (i, 'all') else torch.zeros_like(t)
                       for i, t in enumerate(inputs))
    primal, tangent = torch.autograd.functional.jvp(lambda *args: equations(*args, 2), inputs, directions)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, d) if active in (i, 'all') else t
                          for i, (t, d) in enumerate(zip(inputs, directions)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                result = [forward_ad.unpack_dual(t) for t in stage(duals, 2)]
    assert not any(e.key == 'qlt::pcr_stage_forward' for e in profile.key_averages())
    for (value, derivative), p, d in zip(result, primal, tangent):
        torch.testing.assert_close(value, p, rtol=2e-12, atol=2e-13)
        torch.testing.assert_close(torch.zeros_like(p) if derivative is None else derivative, d, rtol=2e-12, atol=2e-13)


def test_stage_graph_backward_dispatch_and_engine_flag():
    inputs = arguments()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = stage(inputs, 1)
        gradient = torch.autograd.grad(sum(t.square().sum() for t in result), inputs, create_graph=True)
        torch.autograd.grad(sum(t.sum() for t in gradient), inputs)
    counts = {event.key: event.count for event in profile.key_averages()}
    assert counts['qlt::pcr_stage_forward'] == 1
    assert counts['qlt::pcr_stage_backward_graph'] == 1
    field = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    for mode, kernels in [('full', 'auto'), ('full', 'pcr'), ('full', 'vectorized'), ('adjoint', 'auto')]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            value = fd_vanilla(field, time_steps=2, grid_points=11, mode=mode, kernels=kernels)
            torch.autograd.grad(value, field)
        names = {e.key for e in profile.key_averages()}
        used = 'qlt::pcr_stage_forward' in names
        enabled = os.environ.get('QLT_PCR_NATIVE_STAGE', '1') == '1'
        thomas = os.environ.get('QLT_THOMAS_ARITHMETIC', '1') == '1'
        assert used == (enabled and mode == 'full' and (kernels == 'pcr' or (kernels == 'auto' and not thomas)))
        assert ('qlt::thomas_arithmetic_forward' in names) == (thomas and mode == 'full' and kernels == 'auto')


def test_stage_cuda_retains_tensor_equations():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    cpu = arguments(7, 2)
    device = tuple(t.detach().cuda().requires_grad_() for t in cpu)
    actual, expected = stage(device, 4), equations(*cpu, 4)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-12, atol=2e-13)
    ga = torch.autograd.grad(sum(t.square().sum() for t in actual), device)
    ge = torch.autograd.grad(sum(t.square().sum() for t in expected), cpu)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('pattern', ['independent', 'shared', 'cross', 'expanded'])
@pytest.mark.parametrize('negative', [False, True])
@pytest.mark.parametrize('nrhs', [1, 3])
def test_batched_stage_broadcast_strides_and_shared_parameter_gradients(pattern, negative, nrhs):
    n = 5
    gen = torch.Generator().manual_seed(492)
    shapes = {'independent': [(2, 3, 2*n)]*3,
              'shared': [(2*n,)]*3,
              'cross': [(2, 1, 2*n), (1, 3, 2*n), (2*n,)],
              'expanded': [(), (), ()]}[pattern]
    base = tuple(t.requires_grad_() for t in (
        .1*torch.randn(shapes[0], generator=gen, dtype=DTYPE),
        2+torch.rand(shapes[1], generator=gen, dtype=DTYPE),
        -.2*torch.randn(shapes[2], generator=gen, dtype=DTYPE),
        torch.randn(2, 3, 2*n, nrhs, generator=gen, dtype=DTYPE)))
    bands = tuple(t.expand(2, 3, n) if pattern == 'expanded' else t[..., ::2] for t in base[:3])
    inputs = (*bands, base[3][..., ::2, :])
    if negative: inputs = tuple(torch._neg_view(t) for t in inputs)
    initial = [t.detach().clone() for t in base]
    actual, expected = stage(inputs, 4), equations(*inputs, 4)
    weights = tuple(torch.sin(torch.arange(t.numel(), dtype=DTYPE)).reshape_as(t) for t in actual)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-12, atol=2e-13)
    for a, e in zip(torch.autograd.grad(actual, base, weights), torch.autograd.grad(expected, base, weights)):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=2e-12)
    for value, before in zip(base, initial):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


def test_batched_stage_second_and_third_derivatives_and_forward_ad():
    from torch.autograd import forward_ad
    raw = torch.cos(torch.arange(12, dtype=DTYPE)).reshape(3, 4).requires_grad_()
    inputs = (raw[:, 0], 3+raw[:, 1], raw[:, 2], raw[:, ::2].expand(2, 3, 2))
    actual, expected = stage(inputs, 2), equations(*inputs, 2)
    la = sum((i+1)*value.square().sum() for i, value in enumerate(actual))
    le = sum((i+1)*value.square().sum() for i, value in enumerate(expected))
    direction = torch.sin(torch.arange(12, dtype=DTYPE)).reshape_as(raw)
    for order in range(3):
        ga, = torch.autograd.grad(la, raw, create_graph=order < 2)
        ge, = torch.autograd.grad(le, raw, create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-11, atol=2e-11)
        la, le = (ga*direction).sum(), (ge*direction).sum()
    inputs = tuple(t.detach().requires_grad_() for t in (
        torch.full((2, 1, 3), .1, dtype=DTYPE), torch.full((1, 2, 3), 2., dtype=DTYPE),
        torch.full((3,), -.2, dtype=DTYPE), torch.sin(torch.arange(24, dtype=DTYPE)).reshape(2, 2, 3, 2)))
    assert torch.autograd.gradcheck(lambda *xs: stage(xs, 1), inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda *xs: stage(xs, 1), inputs, fast_mode=True)
    directions = tuple(torch.ones_like(t)*.3 for t in inputs)
    p, d = torch.autograd.functional.jvp(lambda *xs: equations(*xs, 1), inputs, directions)
    with forward_ad.dual_level():
        dual = tuple(forward_ad.make_dual(t, v) for t, v in zip(inputs, directions))
        result = [forward_ad.unpack_dual(t) for t in stage(dual, 1)]
    for (value, derivative), primal, tangent in zip(result, p, d):
        torch.testing.assert_close(value, primal, rtol=2e-12, atol=2e-13)
        torch.testing.assert_close(derivative, tangent, rtol=2e-12, atol=2e-13)


def test_pcr_trace_roundtrip_preserves_runtime_shapes_values_and_vjps():
    args = arguments(3, 1)
    compact = tuple(t.detach().requires_grad_() for t in (args[0][1:], args[1], args[2][:-1], args[3]))
    traced = torch.jit.trace(tridiagonal_solve_pcr, compact)
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    buffer.seek(0)
    restored = torch.jit.load(buffer)
    source = arguments(5, 2)
    inputs = (source[0][1:], source[1], source[2][:-1], source[3].expand(2, 5, 2))
    actual = restored(*inputs)
    matrix = torch.diag(source[1])+torch.diag(source[0][1:], -1)+torch.diag(source[2][:-1], 1)
    expected = torch.linalg.solve(matrix, inputs[3])
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(torch.autograd.grad(actual.square().sum(), source),
                    torch.autograd.grad(expected.square().sum(), source)):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('checked', [False, True])
def test_complete_batched_pcr_mixed_higher_derivatives_and_forward_ad(checked):
    from torch.autograd import forward_ad
    n = 5
    inputs = tuple(t.requires_grad_() for t in (
        torch.full((2, 1, n-1), .1, dtype=DTYPE),
        torch.linspace(2., 3., 3*n, dtype=DTYPE).reshape(1, 3, n),
        torch.full((n-1,), -.2, dtype=DTYPE),
        torch.sin(torch.arange(2*3*n*2, dtype=DTYPE)).reshape(2, 3, n, 2)))
    directions = tuple(torch.cos(torch.arange(t.numel(), dtype=DTYPE)).reshape_as(t)*.2 for t in inputs)
    def reference(*xs):
        matrix = torch.diag_embed(xs[0], offset=-1)+torch.diag_embed(xs[1])+torch.diag_embed(xs[2], offset=1)
        return torch.linalg.solve(matrix, xs[3])
    def solve(*xs):
        return tridiagonal_solve_pcr(*xs, check_pivots=checked)
    actual, expected = solve(*inputs), reference(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    la, le = actual.square().sum(), expected.square().sum()
    for order in range(3):
        ga = torch.autograd.grad(la, inputs, create_graph=order < 2)
        ge = torch.autograd.grad(le, inputs, create_graph=order < 2)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-11)
        la, le = sum((g*d).sum() for g, d in zip(ga, directions)), sum((g*d).sum() for g, d in zip(ge, directions))
    p, d = torch.autograd.functional.jvp(reference, inputs, directions)
    with torch.no_grad(), forward_ad.dual_level():
        duals = tuple(forward_ad.make_dual(t, v) for t, v in zip(inputs, directions))
        primal, tangent = forward_ad.unpack_dual(solve(*duals))
    torch.testing.assert_close(primal, p, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(tangent, d, rtol=2e-11, atol=2e-12)
