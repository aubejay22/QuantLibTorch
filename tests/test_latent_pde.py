"""Independent d-dimensional equations for the coefficient-driven QL adapter."""
import itertools
import math

import pytest
import torch

from quantlibtorch import diffusion_rollback, state_grid
from test_pde_adjoint import derivative_matrix


DT = torch.float64
SCHEMES = ["douglas", "hundsdorfer", "modified_craig_sneyd", "craig_sneyd"]


def axes_for(d):
    return [[-.8, -.1, .7], [-.6, .3, .9], [-.7, .5], [-.4, .8]][:d]


def problem(d=2):
    axes = axes_for(d)
    x = state_grid(axes)
    t = torch.tensor([0., .09, .2], dtype=DT)
    features = .7*t[:, None, None]+x[None]*.4+x.sum(-1)[None, :, None]*.2
    drift = .025*torch.sin(features)
    factor = (.2*torch.eye(d, dtype=DT)[None, None]
              + .025*torch.sin(features[..., None]+torch.arange(d, dtype=DT)*.3))
    killing = .02+.005*torch.cos(features.sum(-1))
    terminal = torch.cos(.6*x.sum(-1))+.03*x.square().sum(-1)
    return axes, t.tolist(), (drift, factor, killing, terminal)


def spatial(axes, axis, order):
    result = torch.ones(1, 1, dtype=DT)
    for i in reversed(range(len(axes))):
        matrix = derivative_matrix(axes[i], order) if i == axis else torch.eye(len(axes[i]), dtype=DT)
        result = torch.kron(result, matrix)
    return result


def dense_reference(drift, factor, killing, terminal, axes, times, steps, scheme, horizon=.2):
    """Global dense equations using polynomial moments and torch.linalg.solve.

    No native coefficients or solver output is reused in the reference.
    """
    d, n = len(axes), terminal.numel()
    eye = torch.eye(n, dtype=DT)
    first = [spatial(axes, i, 1) for i in range(d)]
    second = [spatial(axes, i, 2) for i in range(d)]
    covariance = factor@factor.transpose(-1, -2)
    dt = horizon/steps
    theta = .5+math.sqrt(3)/6 if scheme == "hundsdorfer" else 1/3 if scheme == "modified_craig_sneyd" else .5

    def sample(field, t):
        if len(times) == 1:
            return field[0]
        i = next(i for i in range(len(times)-1) if times[i] <= t <= times[i+1])
        w = (t-times[i])/(times[i+1]-times[i])
        return (1-w)*field[i]+w*field[i+1]

    state = terminal
    for step in reversed(range(steps)):
        t = (step+.5)*dt
        b, c, k = sample(drift, t), sample(covariance, t), sample(killing, t)
        directions = [b[:, i, None]*first[i]+.5*c[:, i, i, None]*second[i]-k[:, None]*eye/d for i in range(d)]
        mixed = sum((c[:, i, j, None]*(first[i]@first[j]) for i, j in itertools.combinations(range(d), 2)), torch.zeros_like(eye))
        total = mixed+sum(directions)

        def sweep(rhs, anchor):
            for op in directions:
                rhs = torch.linalg.solve(eye-theta*dt*op, rhs-theta*dt*(op@anchor))
            return rhs

        predictor = state+dt*(total@state)
        y = sweep(predictor, state)
        if scheme == "douglas":
            state = y
        elif scheme == "hundsdorfer":
            state = sweep(predictor+.5*dt*(total@(y-state)), y)
        else:
            mu = 1/3 if scheme == "modified_craig_sneyd" else .5
            corrected = predictor+mu*dt*(mixed@(y-state))
            if scheme == "modified_craig_sneyd":
                corrected = corrected+(.5-mu)*dt*(total@(y-state))
            state = sweep(corrected, state)
    return state


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_time_dependent_nd_equations_and_every_field_vjp(d, scheme, mode):
    axes, times, fields = problem(d)
    actual_fields = [f.clone().requires_grad_() for f in fields]
    reference_fields = [f.clone().requires_grad_() for f in fields]
    b, l, k, u = actual_fields
    actual = diffusion_rollback(b, l, u, killing=k, axes=axes, times=times, horizon=.2,
                                time_steps=3, scheme=scheme, mode=mode)
    expected = dense_reference(*reference_fields, axes, times, 3, scheme)
    seed = torch.sin(torch.arange(u.numel(), dtype=DT)*1.3+.2)
    grad = torch.autograd.grad(actual, actual_fields, seed)
    ref_grad = torch.autograd.grad(expected, reference_fields, seed)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    for a, e in zip(grad, ref_grad):
        torch.testing.assert_close(a, e, rtol=2e-10, atol=3e-12)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_manufactured_all_mixed_pairs_with_time_varying_rank_one_covariance(d, scheme, mode):
    axes = axes_for(d)
    x = state_grid(axes)
    theta = torch.tensor(.13, dtype=DT, requires_grad=True)
    weights = torch.arange(1, d+1, dtype=DT)*.2
    factor = torch.stack((theta*weights, (theta+.08)*weights))[:, None, :, None].expand(2, len(x), d, 1)
    drift = torch.zeros(2, len(x), d, dtype=DT)
    terminal = sum((i+j+1)*x[:, i]*x[:, j] for i, j in itertools.combinations(range(d), 2))
    pair_weight = sum((i+j+1)*weights[i]*weights[j] for i, j in itertools.combinations(range(d), 2))
    result = diffusion_rollback(drift, factor, terminal, axes=axes, times=[0., .2], horizon=.2,
                                time_steps=3, scheme=scheme, mode=mode)
    exact = terminal+.1*(theta.square()+(theta+.08).square())*pair_weight
    torch.testing.assert_close(result, exact, rtol=2e-12, atol=2e-13)
    grad, = torch.autograd.grad(result.sum(), theta)
    ref, = torch.autograd.grad(exact.sum(), theta)
    torch.testing.assert_close(grad, ref, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_noncontiguous_inputs_ownership_scalar_path_and_differences(mode):
    axes, times, fields = problem(3)
    # Fresh strided views of original leaves, preserving gradients and storage.
    leaves = [torch.stack((f, f), -1).requires_grad_() for f in fields]
    args = [f[..., 0] for f in leaves]
    before = [f.detach().clone() for f in leaves]
    def run(fields, kernels="auto"):
        b, l, k, u = fields
        return diffusion_rollback(b, l, u, axes=axes, times=times, killing=k, horizon=.2,
                                   time_steps=2, mode=mode, kernels=kernels).square().sum()
    a = run(args)
    g = torch.autograd.grad(a, args)
    scalar = run(args, "scalar")
    gs = torch.autograd.grad(scalar, args)
    torch.testing.assert_close(a, scalar, rtol=2e-13, atol=2e-13)
    for x, y in zip(g, gs):
        torch.testing.assert_close(x, y, rtol=2e-11, atol=2e-12)
    directions = [torch.cos(torch.arange(f.numel(), dtype=DT)).reshape_as(f)*.1 for f in fields]
    h = 1e-5
    with torch.no_grad():
        fd = (run([f+h*v for f, v in zip(fields, directions)])-run([f-h*v for f, v in zip(fields, directions)]))/(2*h)
    torch.testing.assert_close(sum((x*v).sum() for x, v in zip(g, directions)), fd, rtol=1e-6, atol=2e-8)
    for old, new in zip(before, leaves):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_full_hessian_vector_against_dense_equations():
    axes, times, fields = problem(2)
    p = torch.tensor(.6, dtype=DT, requires_grad=True)
    b, l, k, u = fields
    actual = diffusion_rollback(b*p, l*(p+.4), u, killing=k, axes=axes, times=times,
                                horizon=.2, time_steps=2).square().sum()
    expected = dense_reference(b*p, l*(p+.4), k, u, axes, times, 2, "hundsdorfer").square().sum()
    ag, = torch.autograd.grad(actual, p, create_graph=True)
    eg, = torch.autograd.grad(expected, p, create_graph=True)
    ah, = torch.autograd.grad(ag, p)
    eh, = torch.autograd.grad(eg, p)
    torch.testing.assert_close(ah, eh, rtol=1e-10, atol=1e-11)


def test_adjoint_rejects_second_derivatives():
    axes, times, fields = problem(2)
    b, l, k, u = [x.requires_grad_() for x in fields]
    value = diffusion_rollback(b, l, u, axes=axes, times=times, killing=k,
                                horizon=.2, time_steps=2, mode="adjoint")
    with pytest.raises(RuntimeError, match="first.order|higher|create_graph"):
        torch.autograd.grad(value.sum(), l, create_graph=True)


@pytest.mark.parametrize("bad", ["axes", "times", "dtype", "shape", "nan", "horizon", "terminal", "killing", "mode", "scheme", "kernels", "steps", "overflow"])
def test_rejects_invalid_contracts(bad):
    axes, times, (b, l, k, u) = problem(2)
    kw = dict(axes=axes, times=times, killing=k, horizon=.2, time_steps=2)
    if bad == "axes": kw["axes"] = [axes[0], [0., 0., 1.]]
    elif bad == "times": kw["times"] = [0., .01, .1]
    elif bad == "dtype": b = b.float()
    elif bad == "shape": l = l[..., 0]
    elif bad == "nan": l = l*float("nan")
    elif bad == "horizon": kw["horizon"] = 0.
    elif bad == "terminal": u = u[:-1]
    elif bad == "killing": kw["killing"] = k[0]
    elif bad in ("mode", "scheme", "kernels"): kw[bad] = "invalid"
    elif bad == "steps": kw["time_steps"] = 0
    elif bad == "overflow": l = l*1e200
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        diffusion_rollback(b, l, u, **kw)
