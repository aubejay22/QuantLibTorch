"""Independent equations for directional solves and indexed stencil pullbacks."""
import math

import pytest
import torch

import quantlibtorch  # register native operations


DTYPE = torch.float64
DIRECTIONS = [(dims, direction) for dims in [(2,), (3, 4), (4, 2), (2, 3, 4)]
              for direction in range(len(dims))]


def elimination_order(dimensions, direction):
    # QuantLib dimension zero is fastest; move the chosen tensor axis last.
    n = math.prod(dimensions)
    return torch.arange(n).reshape(tuple(reversed(dimensions))).transpose(
        -1, -1-direction).reshape(-1)


def dense_directional_matrix(bands, scales, dimensions, direction):
    order = elimination_order(dimensions, direction)
    # Build in the ORIGINAL layout independently of the native solve. Deliberately
    # retain arbitrary nonzero bands across adjacent directional-line endpoints.
    matrix = torch.diag(scales[0]*bands[1]+scales[1])
    matrix = matrix.index_put((order[1:], order[:-1]),
                              scales[0]*bands[0, order[1:]], accumulate=True)
    matrix = matrix.index_put((order[:-1], order[1:]),
                              scales[0]*bands[2, order[:-1]], accumulate=True)
    return matrix


def dense_directional_equation(bands, rhs, scales, dimensions, direction):
    return torch.linalg.solve(dense_directional_matrix(bands, scales, dimensions, direction), rhs)


def directional_inputs(n, negative=False):
    bands = torch.stack((.13+.03*torch.sin(torch.arange(2*n, dtype=DTYPE)),
                         1.7+.1*torch.cos(torch.arange(2*n, dtype=DTYPE)),
                         -.17+.02*torch.cos(torch.arange(2*n, dtype=DTYPE)))).requires_grad_()
    rhs = (.3+torch.sin(torch.arange(2*n, dtype=DTYPE))).requires_grad_()
    scales = torch.tensor([.17, .8, 1.1, .9], dtype=DTYPE, requires_grad=True)
    leaves = (bands, rhs, scales)
    args = (bands[:, ::2], rhs[::2], scales[::2])
    if negative:
        args = tuple(torch._neg_view(t) for t in args)
    return leaves, args


@pytest.mark.parametrize("dimensions,direction", DIRECTIONS)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_directional_solve_original_layout_dense_vjps(dimensions, direction, adjoint, negative):
    n = math.prod(dimensions)
    leaves, args = directional_inputs(n, negative)
    before = tuple(t.detach().clone() for t in leaves)
    actual = torch.ops.quantlibtorch._directional_solve_probe(*args, dimensions, direction, adjoint)
    expected = dense_directional_equation(*args, dimensions, direction)
    weights = torch._neg_view(torch.linspace(-.7, .9, 2*n, dtype=DTYPE)[::2])
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    actual_vjp = torch.autograd.grad(actual, leaves, weights)
    expected_vjp = torch.autograd.grad(expected, leaves, weights)
    for gradient, reference in zip(actual_vjp, expected_vjp):
        torch.testing.assert_close(gradient, reference, rtol=2e-11, atol=2e-12)
    for value, original in zip(leaves, before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    # Both unused GLOBAL endpoints have zero derivatives. Cross-line bands
    # elsewhere must remain active, even though PDE operators normally zero them.
    order = elimination_order(dimensions, direction)
    band_gradient = actual_vjp[0][:, ::2]
    assert band_gradient[0, order[0]].item() == 0
    assert band_gradient[2, order[-1]].item() == 0
    if n > dimensions[direction]:
        assert band_gradient[0, order[dimensions[direction]]].abs().item() > 1e-8


@pytest.mark.parametrize("adjoint", [False, True])
def test_directional_solve_parameter_finite_differences(adjoint):
    _, args = directional_inputs(6)
    assert torch.autograd.gradcheck(
        lambda *xs: torch.ops.quantlibtorch._directional_solve_probe(*xs, [2, 3], 1, adjoint),
        args, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("values", [[0., 1.3], [.7, 0.]])
def test_directional_zero_scale_or_shift_keeps_scale_dependencies(values):
    leaves, args = directional_inputs(12)
    scales = torch.tensor(values, dtype=DTYPE, requires_grad=True)
    args = (args[0], args[1], scales)
    inputs = (leaves[0], leaves[1], scales)
    actual = torch.ops.quantlibtorch._directional_solve_probe(*args, [3, 4], 1)
    expected = dense_directional_equation(*args, [3, 4], 1)
    weights = torch.cos(torch.arange(12, dtype=DTYPE))
    actual_gradients = torch.autograd.grad(actual, inputs, weights)
    for gradient, reference in zip(actual_gradients, torch.autograd.grad(expected, inputs, weights)):
        torch.testing.assert_close(gradient, reference, rtol=2e-11, atol=2e-12)
    if values[0] == 0:
        torch.testing.assert_close(actual_gradients[0], torch.zeros_like(leaves[0]), rtol=0, atol=0)
        assert actual_gradients[2][0].abs().item() > 1e-5


def test_directional_full_mode_retains_second_derivatives():
    _, args = directional_inputs(6)
    assert torch.autograd.gradgradcheck(
        lambda *xs: torch.ops.quantlibtorch._directional_solve_probe(*xs, [2, 3], 1, False), args)


def test_directional_context_outlives_operator_and_other_calls():
    leaves, args = directional_inputs(12)
    actual = torch.ops.quantlibtorch._directional_solve_probe(*args, [3, 4], 1)
    # Each native probe's local QuantLib operator has already been destroyed.
    # Further calls must not replace the saved bands, permutation, or solution.
    other_leaves, other_args = directional_inputs(24, True)
    other = torch.ops.quantlibtorch._directional_solve_probe(*other_args, [2, 3, 4], 2)
    torch.autograd.grad(other.sum(), other_leaves)
    expected = dense_directional_equation(*args, [3, 4], 1)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), leaves),
                                   torch.autograd.grad(expected.sum(), leaves)):
        torch.testing.assert_close(gradient, reference, rtol=2e-11, atol=2e-12)


def test_directional_normalized_primal_and_adjoint_residuals():
    _, (bands, rhs, scales) = directional_inputs(12)
    value = torch.ops.quantlibtorch._directional_solve_probe(bands, rhs, scales, [3, 4], 1)
    seed = torch.cos(torch.arange(12, dtype=DTYPE))
    adjoint, = torch.autograd.grad(value, rhs, seed)
    with torch.no_grad():
        matrix = dense_directional_matrix(bands, scales, [3, 4], 1)
        assert (matrix-matrix.T).abs().max().item() > 1e-3
        # Backward errors use the induced infinity norm in each equation:
        # ||Ax-b|| / (||A|| ||x|| + ||b||), separately for A and A.T.
        primal_scale = matrix.abs().sum(1).max()*value.abs().max()+rhs.abs().max()
        adjoint_scale = matrix.T.abs().sum(1).max()*adjoint.abs().max()+seed.abs().max()
        primal_error = (matrix @ value-rhs).abs().max()/primal_scale
        adjoint_error = (matrix.T @ adjoint-seed).abs().max()/adjoint_scale
    assert primal_error.item() < 2e-14
    assert adjoint_error.item() < 2e-14


def test_directional_equation_adjoint_rejects_higher_and_forward_derivatives():
    leaves, args = directional_inputs(6)
    value = torch.ops.quantlibtorch._directional_solve_probe(*args, [2, 3], 1)
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(value.sum(), leaves, create_graph=True)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(args[1], torch.ones_like(args[1]))
        with pytest.raises(RuntimeError, match="forward-mode tangents"):
            torch.ops.quantlibtorch._directional_solve_probe(args[0], dual, args[2], [2, 3], 1)


def test_directional_adjoint_rejects_forward_dual_backward_seed():
    leaves, args = directional_inputs(6)
    value = torch.ops.quantlibtorch._directional_solve_probe(*args, [2, 3], 1)
    with torch.autograd.forward_ad.dual_level():
        seed = torch.autograd.forward_ad.make_dual(
            torch.cos(torch.arange(6, dtype=DTYPE)), torch.linspace(-.4, .8, 6, dtype=DTYPE))
        with pytest.raises(RuntimeError, match="forward-mode tangents"):
            torch.autograd.grad(value, leaves, seed, create_graph=False)


@pytest.mark.parametrize("dimensions,direction", [([], 0), ([1, 3], 0), ([2, 3], -1), ([2, 3], 2)])
def test_directional_invalid_layout_rejected(dimensions, direction):
    _, args = directional_inputs(6)
    with pytest.raises(RuntimeError, match="direction|dimensions"):
        torch.ops.quantlibtorch._directional_solve_probe(*args, dimensions, direction)


@pytest.mark.parametrize("diagonal", [[0., 1.], [1., 1.]])
def test_directional_failed_pivots_rejected(diagonal):
    bands = torch.tensor([[1., 1.], diagonal, [1., 1.]], dtype=DTYPE, requires_grad=True)
    with pytest.raises(RuntimeError, match="division by zero|non-finite"):
        torch.ops.quantlibtorch._directional_solve_probe(
            bands, torch.ones(2, dtype=DTYPE), torch.tensor([1., 0.], dtype=DTYPE), [2], 0)


def test_directional_transpose_keeps_tiny_nonzero_pivot_rejection():
    # A 2x2 block in direction-one elimination order [0,2,1,3]. The scaled
    # upstream forward remains finite; the compact transpose's second pivot
    # falls inside QuantLib's close(pivot,0) threshold, as in the 1D boundary.
    bands = torch.tensor([[0., 0., 90.90909090909092, 0.],
                          [0., 0., 90.90909090909089, 0.],
                          [2.0185873175002846e-14, 0., 0., 0.]],
                         dtype=DTYPE, requires_grad=True)
    rhs = torch.ones(4, dtype=DTYPE, requires_grad=True)
    value = torch.ops.quantlibtorch._directional_solve_probe(
        bands, rhs, torch.tensor([-.011, 1.], dtype=DTYPE), [2, 2], 1)
    assert torch.isfinite(value).all()
    with pytest.raises(RuntimeError, match="division by zero"):
        torch.autograd.grad(value.sum(), (bands, rhs))


def stencil_data(n=7, k=9):
    # Repeated destinations include folded edges and deliberate duplicate
    # stencil entries. The implementation must accumulate all contributions.
    raw_indices = (torch.arange(2*n).unsqueeze(0)+torch.arange(k).unsqueeze(1)) % n
    raw_indices[0] = 0
    indices = raw_indices[:, ::2]
    coefficients = (.2*torch.cos(torch.arange(2*k*n, dtype=DTYPE))).reshape(k, 2*n).requires_grad_()
    state = torch.sin(torch.arange(2*n, dtype=DTYPE)).requires_grad_()
    seed = (.4+torch.cos(torch.arange(2*n, dtype=DTYPE))).requires_grad_()
    return (coefficients, state, seed), (coefficients[:, ::2], indices, state[::2], seed[::2])


def dense_stencil(coefficients, indices):
    n = coefficients.shape[1]
    # One-hot row entries are independent of the implementation's gather/scatter.
    basis = torch.nn.functional.one_hot(indices, n).to(DTYPE)
    return (coefficients.unsqueeze(-1)*basis).sum(0)


@pytest.mark.parametrize("k", [1, 3, 9])
@pytest.mark.parametrize("negative", [False, True])
def test_stencil_application_transpose_and_coefficient_vjp(k, negative):
    leaves, (coefficients, indices, state, seed) = stencil_data(k=k)
    originals = tuple(t.detach().clone() for t in leaves)
    if negative:
        coefficients, state, seed = (torch._neg_view(t) for t in (coefficients, state, seed))
    expected_matrix = dense_stencil(coefficients, indices)
    output = torch.ops.quantlibtorch._stencil_apply(coefficients, indices, state)
    transposed = torch.ops.quantlibtorch._stencil_transpose(coefficients, indices, seed)
    gc, gu = torch.ops.quantlibtorch._stencil_vjp(coefficients, indices, state, seed)
    torch.testing.assert_close(output, expected_matrix @ state, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(transposed, expected_matrix.T @ seed, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(output @ seed, state @ transposed, rtol=2e-13, atol=2e-14)
    reference_gradients = torch.autograd.grad(expected_matrix @ state, (coefficients, state), seed)
    actual_gradients = torch.autograd.grad(output, (coefficients, state), seed)
    for actual, explicit, reference in zip(actual_gradients, (gc, gu), reference_gradients):
        torch.testing.assert_close(actual, reference, rtol=2e-13, atol=2e-14)
        torch.testing.assert_close(explicit, reference, rtol=2e-13, atol=2e-14)
    for value, original in zip(leaves, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_stencil_vjp_keeps_coefficient_state_and_seed_cross_derivatives():
    _, (coefficients, indices, state, seed) = stencil_data(n=3, k=3)
    assert torch.autograd.gradcheck(
        lambda c, u: torch.ops.quantlibtorch._stencil_apply(c, indices, u), (coefficients, state))
    assert torch.autograd.gradgradcheck(
        lambda c, u: torch.ops.quantlibtorch._stencil_apply(c, indices, u), (coefficients, state))
    assert torch.autograd.gradcheck(
        lambda c, u, w: torch.ops.quantlibtorch._stencil_vjp(c, indices, u, w),
        (coefficients, state, seed))
    assert torch.autograd.gradgradcheck(
        lambda c, w: torch.ops.quantlibtorch._stencil_transpose(c, indices, w), (coefficients, seed))


@pytest.mark.parametrize("mode", ["apply", "transpose"])
def test_stencil_forward_ad(mode):
    _, (coefficients, indices, state, _) = stencil_data(n=3, k=3)
    operation = getattr(torch.ops.quantlibtorch, "_stencil_"+mode)
    dc, du = torch.sin(coefficients), torch.cos(state)
    with torch.autograd.forward_ad.dual_level():
        c = torch.autograd.forward_ad.make_dual(coefficients, dc)
        u = torch.autograd.forward_ad.make_dual(state, du)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(operation(c, indices, u))
    torch.testing.assert_close(primal, operation(coefficients, indices, state))
    torch.testing.assert_close(tangent, operation(dc, indices, state)+operation(coefficients, indices, du))


@pytest.mark.parametrize("problem", ["dtype", "shape", "index"])
def test_stencil_invalid_inputs_rejected(problem):
    _, (coefficients, indices, state, _) = stencil_data(n=3, k=3)
    if problem == "dtype":
        coefficients = coefficients.float()
    elif problem == "shape":
        indices = indices[:2]
    else:
        indices = indices.clone()
        indices[0, 0] = 3
    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.quantlibtorch._stencil_apply(coefficients, indices, state)
