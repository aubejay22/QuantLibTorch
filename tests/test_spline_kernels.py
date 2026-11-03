"""Spline values, public coefficients/primitives, and independent VJPs."""
import pytest
import torch

import quantlibtorch

DTYPE = torch.float64


def probe(x, y, q, monotonic=True, kernels=2, adjoint=False):
    return torch.ops.quantlibtorch._spline_probe(x, y, q, monotonic, kernels, adjoint)


@pytest.mark.parametrize("n", [2, 3, 4, 11])
@pytest.mark.parametrize("field", ["flat", "linear", "spiky", "monotone"])
@pytest.mark.parametrize("kernels,adjoint", [(2, False), (2, True), (4, False)])
def test_spline_matches_original_values_flags_and_vjps(n, field, kernels, adjoint):
    x = torch.cumsum(.2+torch.arange(n, dtype=DTYPE)*.06, 0).requires_grad_()
    if field == "flat":
        y = torch.ones(n, dtype=DTYPE)
    elif field == "linear":
        y = 1.3+2*x.detach()
    elif field == "spiky":
        y = torch.cos(torch.arange(n, dtype=DTYPE)*2.7)
    else:
        y = torch.cumsum(.01+torch.sin(torch.arange(n, dtype=DTYPE)*2.1).square(), 0)
    y.requires_grad_()
    q = torch.linspace(x[0].item()-.15, x[-1].item()+.13, 7, dtype=DTYPE).requires_grad_()
    before = [v.detach().clone() for v in (x, y, q)]
    expected = probe(x, y, q, kernels=0, adjoint=False)
    actual = probe(x, y, q, kernels=kernels, adjoint=adjoint)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=2e-12, atol=2e-12)
    assert torch.equal(actual[3], expected[3])
    weights = [torch.cos(torch.arange(v.numel(), dtype=DTYPE)).reshape_as(v)/v.numel()
               for v in actual[:3]]
    def vjp(outputs):
        pairs = [(v, w) for v, w in zip(outputs[:3], weights) if v.requires_grad]
        return torch.autograd.grad([v for v, _ in pairs], (x, y, q), [w for _, w in pairs])
    for g, ref in zip(vjp(actual), vjp(expected)):
        torch.testing.assert_close(g, ref, rtol=2e-10, atol=2e-10)
    for v, original in zip((x, y, q), before):
        torch.testing.assert_close(v, original, rtol=0, atol=0)


def dense_natural(x, y, q):
    h = x[1:]-x[:-1]
    s = (y[1:]-y[:-1])/h
    one = x.new_ones(1)
    matrix = torch.diag(torch.cat((2*one, 2*(h[:-1]+h[1:]), 2*one)))
    matrix = matrix + torch.diag(torch.cat((h[1:], one)), -1)
    matrix = matrix + torch.diag(torch.cat((one, h[:-1])), 1)
    rhs = torch.cat((3*s[:1], 3*(h[1:]*s[:-1]+h[:-1]*s[1:]), 3*s[-1:]))
    d = torch.linalg.solve(matrix, rhs)
    a = d[:-1]
    b = (3*s-d[1:]-2*d[:-1])/h
    c = (d[1:]+d[:-1]-2*s)/(h*h)
    integrals = h*(y[:-1]+h*(a/2+h*(b/3+h*c/4)))
    constants = [x.new_zeros(())]
    for increment in integrals[:-1]:
        constants.append(constants[-1]+increment)
    constants = torch.stack(constants)
    j = (torch.searchsorted(x.detach(), q.detach(), right=True)-1).clamp(0, x.numel()-2)
    z = q-x[j]
    values = y[j]+z*(a[j]+z*(b[j]+z*c[j]))
    deriv = a[j]+(2*b[j]+3*c[j]*z)*z
    second = 2*b[j]+6*c[j]*z
    primitive = constants[j]+z*(y[j]+z*(a[j]/2+z*(b[j]/3+z*c[j]/4)))
    return torch.stack((values, deriv, second, primitive))


@pytest.mark.parametrize("adjoint", [False, True])
def test_natural_spline_dense_reference_and_coordinate_vjps(adjoint):
    x = torch.tensor([.1, .3, .65, 1.2, 1.8, 2.], dtype=DTYPE, requires_grad=True)
    y = torch.tensor([.2, .5, -.2, .9, 1.1, -.1], dtype=DTYPE, requires_grad=True)
    q = torch.tensor([-.2, .21, .5, 1.6, 2.1], dtype=DTYPE, requires_grad=True)
    actual = probe(x, y, q, monotonic=False, adjoint=adjoint)[0]
    expected = dense_natural(x, y, q)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    weights = torch.linspace(-.4, .8, actual.numel(), dtype=DTYPE).reshape_as(actual)
    for g, ref in zip(torch.autograd.grad(actual, (x, y, q), weights),
                      torch.autograd.grad(expected, (x, y, q), weights)):
        torch.testing.assert_close(g, ref, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("adjoint", [False, True])
def test_spline_finite_differences(adjoint):
    args = tuple(v.requires_grad_() for v in (
        torch.tensor([.1, .4, .9, 1.4], dtype=DTYPE),
        torch.tensor([.2, .5, -.2, 1.1], dtype=DTYPE),
        torch.tensor([.23, 1.1], dtype=DTYPE)))
    function = lambda *vs: probe(*vs, adjoint=adjoint)[0]
    assert torch.autograd.gradcheck(function, args)
    if not adjoint:
        assert torch.autograd.gradgradcheck(function, args)


@pytest.mark.parametrize("n", [2, 7, 11])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_internal_tridiagonal_hook_against_dense(n, alias, adjoint):
    generator = torch.Generator().manual_seed(193)
    args = tuple(v.requires_grad_() for v in (
        (.1*torch.rand(2*(n-1), dtype=DTYPE, generator=generator))[::2],
        (2+torch.rand(2*n, dtype=DTYPE, generator=generator))[::2],
        (-.2*torch.rand(2*(n-1), dtype=DTYPE, generator=generator))[::2],
        torch.randn(2*n, dtype=DTYPE, generator=generator)[::2]))
    lower, diagonal, upper, rhs = args
    before = [v.detach().clone() for v in args]
    actual = torch.ops.quantlibtorch._tridiagonal_hook(*args, adjoint, alias)
    matrix = torch.diag(lower, -1)+torch.diag(diagonal)+torch.diag(upper, 1)
    expected = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    weights = torch.linspace(-.5, 1.1, 2*n, dtype=DTYPE)[::2]
    for g, ref in zip(torch.autograd.grad(actual, args, weights), torch.autograd.grad(expected, args, weights)):
        torch.testing.assert_close(g, ref, rtol=2e-12, atol=2e-13)
    for v, original in zip(args, before):
        torch.testing.assert_close(v, original, rtol=0, atol=0)


def test_spline_adjoint_rejects_second_derivatives():
    x = torch.tensor([0., .3, 1.], dtype=DTYPE)
    y = torch.tensor([.2, .1, .5], dtype=DTYPE, requires_grad=True)
    q = torch.tensor([.4], dtype=DTYPE)
    value = probe(x, y, q, adjoint=True)[0][0, 0]
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(value, y, create_graph=True)
