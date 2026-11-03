"""Independent discrete references for real QuantLib two-dimensional rollback.

Dense matrices are deliberately confined to these small validation problems.
Spatial weights come from polynomial moment fitting, not native band exports.
"""
import math

import pytest
import torch

import quantlibtorch  # Register the native dispatcher library.


DTYPE = torch.float64
SCHEMES = [0, 1, 2, 3]  # Douglas, Hundsdorfer, modified Craig-Sneyd, Craig-Sneyd.
GRIDS = [
    ([-1.1, -.45, .2, 1.3], [-.7, .15, .9]),
    ([-.9, .8], [-.6, -.05, .5, 1.2]),
]


def derivative_matrix(axis, order):
    """Differentiate the local interpolating polynomial at each grid point.

    QuantLib uses two-point first derivatives at boundaries and zero boundary
    rows for second derivatives. Interior formulas differentiate a quadratic.
    Solving the moment equations is independent of its hard-coded stencil.
    """
    points = torch.tensor(axis, dtype=DTYPE)
    matrix = torch.zeros((len(axis), len(axis)), dtype=DTYPE)
    for row in range(len(axis)):
        if order == 2 and row in (0, len(axis)-1):
            continue
        indices = ([0, 1] if row == 0 else
                   [len(axis)-2, len(axis)-1] if row == len(axis)-1 else
                   [row-1, row, row+1])
        offsets = points[indices]-points[row]
        moments = torch.stack([offsets**power for power in range(len(indices))])
        rhs = torch.zeros(len(indices), dtype=DTYPE)
        rhs[order] = math.factorial(order)
        matrix[row, indices] = torch.linalg.solve(moments, rhs)
    return matrix


def spatial_matrices(x, y):
    ix, iy = torch.eye(len(x), dtype=DTYPE), torch.eye(len(y), dtype=DTYPE)
    dx, dy = derivative_matrix(x, 1), derivative_matrix(y, 1)
    # QL axis zero is fastest, hence a [ny,nx] tensor has the native ordering.
    return (torch.kron(iy, dx), torch.kron(dy, ix),
            torch.kron(iy, derivative_matrix(x, 2)),
            torch.kron(derivative_matrix(y, 2), ix),
            torch.kron(dy, dx))


def dense_operators(coefficients, x, y):
    dx, dy, dxx, dyy, dxy = spatial_matrices(x, y)
    bx, by, ax, ay, mixed, killing = coefficients.reshape(6, -1)
    identity = torch.eye(len(x)*len(y), dtype=DTYPE)
    first = bx[:, None]*dx+ax[:, None]*dxx-.5*killing[:, None]*identity
    second = by[:, None]*dy+ay[:, None]*dyy-.5*killing[:, None]*identity
    return mixed[:, None]*dxy, first, second


def dense_rollback(coefficients, terminal, x, y, horizon, steps, scheme):
    """Dense mathematical stage equations, using independent global solves."""
    mixed, first, second = dense_operators(coefficients, x, y)
    total, directions = mixed+first+second, (first, second)
    theta = .5+math.sqrt(3)/6 if scheme == 1 else 1/3 if scheme == 2 else .5
    mu = 1/3 if scheme == 2 else .5
    identity = torch.eye(terminal.numel(), dtype=DTYPE)
    dt = horizon/steps
    matrices = [identity-theta*dt*operator for operator in directions]

    def sweep(start, anchor):
        result = start
        for matrix, operator in zip(matrices, directions):
            result = torch.linalg.solve(matrix, result-theta*dt*(operator @ anchor))
        return result

    state = terminal.reshape(-1)
    for _ in range(steps):
        explicit = state+dt*(total @ state)
        predicted = sweep(explicit, state)
        if scheme == 0:
            state = predicted
        elif scheme == 1:
            corrected = explicit+mu*dt*(total @ (predicted-state))
            state = sweep(corrected, predicted)
        else:
            corrected = explicit+mu*dt*(mixed @ (predicted-state))
            if scheme == 2:
                corrected = corrected+(.5-mu)*dt*(total @ (predicted-state))
            state = sweep(corrected, state)
    return state.reshape_as(terminal)


def problem(x, y):
    yy, xx = torch.meshgrid(torch.tensor(y, dtype=DTYPE),
                            torch.tensor(x, dtype=DTYPE), indexing="ij")
    coefficients = torch.stack((
        .025*torch.sin(.7*xx+yy),
        -.035+.006*torch.sin(2*yy-xx),
        .09+.01*torch.cos(xx-.3*yy),
        .07+.008*torch.cos(2*xx+yy),
        .025+.006*torch.sin(xx*yy+yy),
        .03+.004*torch.cos(.4*xx+yy),
    ))
    terminal = torch.cos(.6*xx-.4*yy)+.2*xx*yy+.1*xx.square()*yy.square()
    return coefficients, terminal


def weights_like(state):
    return (torch.cos(torch.arange(state.numel(), dtype=DTYPE)+.4)
            + torch.linspace(-.2, .3, state.numel(), dtype=DTYPE)).reshape_as(state)


def native(coefficients, terminal, x, y, *, horizon=.13, steps=3, scheme=0, adjoint=True):
    return torch.ops.quantlibtorch._pde2d_probe(
        coefficients, terminal, x, y, horizon, steps, scheme, adjoint)


@pytest.mark.parametrize("x,y", GRIDS)
def test_independent_mixed_reference_including_edges_and_corners(x, y):
    dx, dy, dxx, dyy, dxy = spatial_matrices(x, y)
    yy, xx = torch.meshgrid(torch.tensor(y, dtype=DTYPE),
                            torch.tensor(x, dtype=DTYPE), indexing="ij")
    one = torch.ones(xx.numel(), dtype=DTYPE)
    for operator in (dx, dy, dxx, dyy, dxy):
        torch.testing.assert_close(operator @ one, torch.zeros_like(one), rtol=0, atol=2e-14)
    torch.testing.assert_close(dxy @ (xx*yy).reshape(-1), one, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(dx @ xx.reshape(-1), one, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(dy @ yy.reshape(-1), one, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("x,y", GRIDS)
def test_quantlib_2d_states_and_all_coefficient_vjps_match_dense(scheme, adjoint, steps, x, y):
    coefficients, terminal = (value.requires_grad_() for value in problem(x, y))
    inputs = (coefficients, terminal)
    before = tuple(value.detach().clone() for value in inputs)
    actual = native(*inputs, x, y, steps=steps, scheme=scheme, adjoint=adjoint)
    expected = dense_rollback(*inputs, x, y, .13, steps, scheme)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    seeds = weights_like(terminal)
    gradients = torch.autograd.grad(actual, inputs, seeds)
    references = torch.autograd.grad(expected, inputs, seeds)
    for gradient, reference in zip(gradients, references):
        torch.testing.assert_close(gradient, reference, rtol=2e-10, atol=2e-12)
    assert references[0][4].abs().max() > 1e-5
    for value, original in zip(inputs, before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_dense_validation_problem_has_noncommuting_directions():
    x, y = GRIDS[0]
    coefficients, _ = problem(x, y)
    _, first, second = dense_operators(coefficients, x, y)
    assert (first @ second-second @ first).abs().max() > 1e-3


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("adjoint", [False, True])
def test_quantlib_mixed_polynomial_exact_solution_and_sensitivity(scheme, adjoint):
    x, y = GRIDS[0]
    yy, xx = torch.meshgrid(torch.tensor(y, dtype=DTYPE),
                            torch.tensor(x, dtype=DTYPE), indexing="ij")
    amplitude = torch.tensor(.05, dtype=DTYPE, requires_grad=True)
    zero = torch.zeros_like(xx)
    coefficients = torch.stack((zero, zero, zero+.12, zero+.08,
                                amplitude.expand_as(xx), zero))
    output = native(coefficients, xx*yy, x, y, horizon=.19, steps=4,
                    scheme=scheme, adjoint=adjoint)
    torch.testing.assert_close(output, xx*yy+.19*amplitude, rtol=2e-12, atol=2e-12)
    seeds = weights_like(xx)
    gradient, = torch.autograd.grad(output, amplitude, seeds)
    torch.testing.assert_close(gradient, .19*seeds.sum(), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("adjoint", [False, True])
def test_quantlib_2d_strided_negative_views_and_saved_ownership(adjoint):
    x, y = GRIDS[0]
    values = problem(x, y)
    raw = tuple(torch.repeat_interleave(-value, 2, dim=-1).requires_grad_() for value in values)
    inputs = tuple(torch._neg_view(value[..., ::2]) for value in raw)
    before = tuple(value.detach().clone() for value in raw)
    actual = native(*inputs, x, y, scheme=1, adjoint=adjoint)
    # Constructing a second operator must not rebind saved state from the first.
    other = native(inputs[0]*1.1, inputs[1]+.2, x, y, scheme=2, adjoint=adjoint)
    assert not torch.allclose(actual, other)
    expected = dense_rollback(*inputs, x, y, .13, 3, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    seeds = torch._neg_view(-weights_like(inputs[1]))
    for result, reference in zip(torch.autograd.grad(actual, raw, seeds),
                                 torch.autograd.grad(expected, raw, seeds)):
        torch.testing.assert_close(result, reference, rtol=2e-10, atol=2e-12)
    for value, original in zip(raw, before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_quantlib_2d_adjoint_gradcheck():
    x, y = [-.8, .05, 1.1], [-.6, .25, .9]
    inputs = tuple(value.requires_grad_() for value in problem(x, y))
    assert torch.autograd.gradcheck(
        lambda c, state: native(c, state, x, y, steps=1, scheme=2),
        inputs, eps=1e-6, atol=2e-6, rtol=2e-5, fast_mode=True)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_quantlib_2d_adjoint_explicit_first_order_contract(scheme):
    x, y = GRIDS[0]
    coefficients, terminal = (value.requires_grad_() for value in problem(x, y))
    output = native(coefficients, terminal, x, y, scheme=scheme, steps=1)
    with pytest.raises(RuntimeError, match="first derivatives only|first-order"):
        torch.autograd.grad(output.sum(), (coefficients, terminal), create_graph=True)


def test_quantlib_2d_full_hvp_matches_dense_and_gradient_differences():
    x, y = [-.8, .05, 1.1], [-.6, .25, .9]
    inputs = tuple(value.requires_grad_() for value in problem(x, y))
    directions = tuple(torch.sin(torch.arange(value.numel(), dtype=DTYPE)+.3).reshape_as(value)
                       for value in inputs)
    seeds = weights_like(inputs[1])
    actual = native(*inputs, x, y, steps=2, scheme=2, adjoint=False)
    expected = dense_rollback(*inputs, x, y, .13, 2, 2)

    def product(output):
        gradient = torch.autograd.grad(output, inputs, seeds, create_graph=True)
        contraction = sum((g*direction).sum() for g, direction in zip(gradient, directions))
        return torch.autograd.grad(contraction, inputs)

    hvp, reference = product(actual), product(expected)
    for value, target in zip(hvp, reference):
        torch.testing.assert_close(value, target, rtol=5e-9, atol=2e-11)
    h = 1e-5
    differences = []
    for sign in (-1, 1):
        shifted = tuple((value.detach()+sign*h*direction).requires_grad_()
                        for value, direction in zip(inputs, directions))
        output = native(*shifted, x, y, steps=2, scheme=2, adjoint=False)
        differences.append(torch.autograd.grad(output, shifted, seeds))
    for value, low, high in zip(hvp, *differences):
        torch.testing.assert_close(value, (high-low)/(2*h), rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("scheme", [1, 2])
def test_neural_coefficient_parameter_vjp_matches_directional_differences(adjoint, scheme):
    x, y = GRIDS[0]
    base, terminal = problem(x, y)
    yy, xx = torch.meshgrid(torch.tensor(y, dtype=DTYPE),
                            torch.tensor(x, dtype=DTYPE), indexing="ij")
    coordinates = torch.stack((xx, yy), dim=-1)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(),
                                torch.nn.Linear(4, 6)).double()
    parameters = tuple(model.parameters())
    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            parameter.copy_(torch.linspace(-.2+.02*index, .25, parameter.numel()).reshape_as(parameter))
    directions = tuple(torch.cos(torch.arange(value.numel(), dtype=DTYPE)+index).reshape_as(value)
                       for index, value in enumerate(parameters))
    seeds = weights_like(terminal)

    def loss():
        coefficients = base+.01*torch.tanh(model(coordinates)).movedim(-1, 0)
        output = native(coefficients, terminal, x, y, scheme=scheme, adjoint=adjoint)
        return ((output*seeds).sum()-.3).square()

    gradients = torch.autograd.grad(loss(), parameters)
    predicted = sum((gradient*direction).sum() for gradient, direction in zip(gradients, directions))
    assert predicted.abs() > 1e-5
    original = tuple(value.detach().clone() for value in parameters)
    estimates = []
    with torch.no_grad():
        for h in (1e-4, 3e-5, 1e-5):
            for value, saved, direction in zip(parameters, original, directions):
                value.copy_(saved+h*direction)
            high = loss()
            for value, saved, direction in zip(parameters, original, directions):
                value.copy_(saved-h*direction)
            low = loss()
            estimates.append((high-low)/(2*h))
        for value, saved in zip(parameters, original):
            value.copy_(saved)
    for estimate in estimates:
        torch.testing.assert_close(predicted, estimate, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("invalid", [
    "dtype", "coefficient_shape", "terminal_shape", "axis_shape",
    "duplicate_axis", "nonfinite_axis", "nonfinite_coefficients",
    "horizon_zero", "horizon_nonfinite", "steps_zero", "scheme",
])
def test_quantlib_2d_invalid_inputs_are_rejected(invalid):
    x, y = (list(axis) for axis in GRIDS[0])
    coefficients, terminal = problem(x, y)
    kwargs = {}
    if invalid == "dtype":
        coefficients = coefficients.float()
    elif invalid == "coefficient_shape":
        coefficients = coefficients[:5]
    elif invalid == "terminal_shape":
        terminal = terminal.reshape(-1)
    elif invalid == "axis_shape":
        x = x[:-1]
    elif invalid == "duplicate_axis":
        x[1] = x[0]
    elif invalid == "nonfinite_axis":
        y[1] = float("nan")
    elif invalid == "nonfinite_coefficients":
        coefficients[4, 1, 1] = float("nan")
    elif invalid == "horizon_zero":
        kwargs["horizon"] = 0.
    elif invalid == "horizon_nonfinite":
        kwargs["horizon"] = float("inf")
    elif invalid == "steps_zero":
        kwargs["steps"] = 0
    else:
        kwargs["scheme"] = 4
    with pytest.raises(RuntimeError, match="PDE|nodes"):
        native(coefficients, terminal, x, y, **kwargs)
