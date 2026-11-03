"""Original matrix ownership preserves every preparation edge and its lifetime."""
from contextlib import nullcontext
import os

import pytest
import torch
from torch.autograd import forward_ad

from test_backward_duals import compare_incoming_duals
from test_full_tensor_step import inputs
from test_step_preparation import equation, grads, loss
from test_thomas_arithmetic import state_reference

DTYPE = torch.float64


def prepare(bands, state, dt=.04, theta=.6, kind=0, permuted=False):
    return torch.ops.quantlibtorch._step_preparation(bands, state, dt, theta, kind, True, permuted, True)


def enabled():
    return (os.environ.get('QLT_ASSEMBLED_BANDS', '1') == '1'
            and os.environ.get('QLT_STEP_PREPARATION', '1') == '1'
            and os.environ.get('QLT_ARITHMETIC_NATIVE_FORWARD', '1') == '1')


def owner_enabled():
    return enabled() and all(os.environ.get(flag, '1') == '1' for flag in (
        'QLT_FULL_TENSOR_STEP', 'QLT_OPERATOR_ARITHMETIC', 'QLT_ASSEMBLY_ARITHMETIC'))


def counts(profile):
    return {event.key: event.count for event in profile.key_averages()}


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('theta', [-.3, 0., .5, 1., 1.3])
@pytest.mark.parametrize('n', [2, 9])
def test_matrix_input_retains_exact_forward_bits_and_all_vjps(kind, theta, n):
    xs = inputs(n)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = prepare(*xs, kind=kind, theta=theta)
    expected = equation(*xs, kind=kind, theta=theta)
    for a, e in zip(actual, expected):
        assert torch.equal(a.view(torch.int64), e.view(torch.int64))
    assert bool(counts(profile).get('qlt::assembled_preparation_forward', 0)) == enabled()
    torch.testing.assert_close(grads(loss(actual), xs), grads(loss(expected), xs), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'shared', 'expanded'])
@pytest.mark.parametrize('permuted', [False, True])
def test_original_matrix_keeps_third_derivatives_aliases_and_neighbours(kind, view, permuted):
    leaves = inputs(18 if view == 'strided' else 9)
    xs = leaves
    if view == 'strided': xs = (leaves[0][:, ::2], leaves[1][::2])
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in leaves)
    elif view == 'shared':
        leaves = (torch.linspace(.01, .1, 9, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0].expand(3, 9), leaves[0])
    elif view == 'expanded':
        leaves = (torch.linspace(.01, .1, 4, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0][:3, None].expand(3, 9), leaves[0][3:].expand(9))
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    la, le = loss(prepare(*xs, kind=kind, permuted=permuted)), loss(equation(*xs, kind=kind, permuted=permuted))
    for order in range(3):
        ga, ge = grads(la, leaves, order < 2), grads(le, leaves, order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-11, atol=2e-12)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, leaves))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, leaves))
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('selected', [0, 1, 2, 3, 'all'])
@pytest.mark.parametrize('create_graph', [False, True])
def test_each_output_incoming_dual_and_undefined_other_weights(kind, selected, create_graph):
    xs = inputs()
    compare_incoming_duals(xs, prepare(*xs, kind=kind), equation(*xs, kind=kind),
                           selected=selected, create_graph=create_graph)


@pytest.mark.parametrize('kind', [0, 1])
def test_matrix_input_first_and_second_differences(kind):
    f = lambda b, s: prepare(b, s, kind=kind)
    assert torch.autograd.gradcheck(f, inputs(5), fast_mode=True)
    assert torch.autograd.gradgradcheck(f, inputs(5), fast_mode=True)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('no_grad', [False, True])
def test_matrix_input_forward_dual_fallback(kind, no_grad):
    xs = inputs()
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, torch.cos(t)) for t in xs)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = prepare(*duals, kind=kind)
            expected = equation(*duals, kind=kind)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(forward_ad.unpack_dual(a), forward_ad.unpack_dual(e), rtol=0, atol=0)
            assert counts(profile).get('qlt::assembled_preparation_forward', 0) == 0


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('transform', ['grad', 'jvp', 'functionalize', 'vmap', 'fake'])
def test_matrix_input_function_transform_fallback(kind, transform):
    xs = inputs()
    actual = lambda *args: prepare(*args, kind=kind)
    expected = lambda *args: equation(*args, kind=kind)
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: loss(f(*args)), argnums=(0, 1))(*xs)
    elif transform == 'jvp':
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.cos(t) for t in xs))
    elif transform == 'functionalize':
        run = lambda f: torch.func.functionalize(f)(*xs)
    elif transform == 'vmap':
        run = lambda f: torch.vmap(f)(*(torch.stack((t, t*1.1)) for t in xs))
    else:
        from torch.fx.experimental.proxy_tensor import make_fx
        run = lambda f: make_fx(f, tracing_mode='fake')(*xs)(*xs)
    torch.testing.assert_close(run(actual), run(expected), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('active', [0, 1])
def test_partial_input_activity(kind, active):
    xs = tuple(t.detach().requires_grad_(i == active) for i, t in enumerate(inputs()))
    actual, expected = prepare(*xs, kind=kind), equation(*xs, kind=kind)
    torch.testing.assert_close(torch.autograd.grad(loss(actual), xs[active]),
                               torch.autograd.grad(loss(expected), xs[active]), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float32), ('cuda', torch.float32), ('cuda', torch.float64)])
def test_other_dtype_and_device_fallbacks(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    xs = tuple(t.detach().to(device=device, dtype=dtype).requires_grad_() for t in inputs())
    actual, expected = prepare(*xs), equation(*xs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(grads(loss(actual), xs), grads(loss(expected), xs), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('kind', [0, 1])
def test_prepared_matrix_keeps_all_elimination_output_dependencies(kind):
    xs = inputs()
    prepared = prepare(*xs, kind=kind)
    original = equation(*xs, kind=kind)
    actual = torch.ops.quantlibtorch._thomas_arithmetic_state(*prepared[:3], prepared[3][:, None], False, True)
    expected = state_reference(*original[:3], original[3][:, None])
    la, le = loss(tuple(t.flatten() for t in actual)), loss(tuple(t.flatten() for t in expected))
    for order in range(3):
        ga, ge = grads(la, xs, order < 2), grads(le, xs, order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, xs))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, xs))


@pytest.mark.parametrize('transition', [0, 1, 2], ids=['exception', 'other_kernel', 'no_grad'])
def test_operator_rebinding_copy_and_invalidation_keep_original_graphs(transition):
    leaves = (torch.tensor(.21, dtype=DTYPE, requires_grad=True),
              torch.tensor(.27, dtype=DTYPE, requires_grad=True),
              torch.linspace(.5, 1.3, 11, dtype=DTYPE, requires_grad=True))
    versions = tuple(t._version for t in leaves)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, True, transition)
    expected = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, False, transition)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    events = counts(profile)
    assert events.get('qlt::assembled_bands_step', 0) == (3 if owner_enabled() else 0)
    # A fourth step after exception, fallback assembly, or NoGrad assembly must
    # use the Array coefficients; an old active tensor would leak dependencies.
    assert events.get('qlt::assembled_bands_store', 0) == (2 if enabled() and
        os.environ.get('QLT_ASSEMBLY_ARITHMETIC', '1') == '1' else 0)
    torch.testing.assert_close(actual[2], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], actual[1], rtol=2e-12, atol=2e-12)
    la, le = loss(actual), loss(expected)
    for order in range(3):
        ga, ge = grads(la, leaves, order < 2), grads(le, leaves, order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-10)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, leaves))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, leaves))
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('transition', [0, 1, 2])
def test_copied_operator_outputs_survive_owner_destruction_and_reversed_backwards(transition):
    leaves = (torch.tensor(.21, dtype=DTYPE, requires_grad=True),
              torch.tensor(.27, dtype=DTYPE, requires_grad=True),
              torch.linspace(.5, 1.3, 11, dtype=DTYPE, requires_grad=True))
    actual = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, True, transition)
    expected = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, False, transition)
    for i in (3, 2, 0, 1):
        ga = torch.autograd.grad(actual[i].sin().sum(), leaves, allow_unused=True, retain_graph=True)
        ge = torch.autograd.grad(expected[i].sin().sum(), leaves, allow_unused=True, retain_graph=True)
        ga = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(ga, leaves))
        ge = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(ge, leaves))
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize('pattern', ['large_rhs', 'weak_dominance'])
def test_owned_preparation_rejection_preserves_original_step_and_inputs(pattern):
    scale = 1e60 if pattern == 'large_rhs' else 1.
    sigma = .21 if pattern == 'large_rhs' else 3000.
    leaves = (torch.tensor(sigma, dtype=DTYPE, requires_grad=True),
              torch.tensor(sigma*1.1, dtype=DTYPE, requires_grad=True),
              (torch.linspace(.5, 1.3, 11, dtype=DTYPE)*scale).requires_grad_())
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, True, 0)
    expected = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, False, 0)
    events = counts(profile)
    assert events.get('qlt::auto_domain_native', 0) > events.get('qlt::auto_residual_native', 0)
    assert events.get('qlt::assembled_bands_step', 0) == (3 if owner_enabled() else 0)
    torch.testing.assert_close(tuple(t/scale for t in actual), tuple(t/scale for t in expected),
                               rtol=2e-10, atol=2e-11)
    ga = grads(loss(tuple(t/scale for t in actual)), leaves)
    ge = grads(loss(tuple(t/scale for t in expected)), leaves)
    # Normalize the state VJP as well, so a huge RHS cannot hide its error.
    torch.testing.assert_close(ga[:2]+(ga[2]*scale,), ge[:2]+(ge[2]*scale,), rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions
