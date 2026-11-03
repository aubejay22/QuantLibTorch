"""Forward-mode incoming VJP weights must survive full-mode native boundaries."""
import pytest
import torch
from torch.autograd import forward_ad

from quantlibtorch import fd_vanilla, tridiagonal_solve_pcr
from test_diffusion_arithmetic import arguments as assembly_inputs
from test_operator_arithmetic import arguments as apply_inputs
from test_pcr_stage import arguments as stage_inputs, equations as stage_reference, stage

DTYPE = torch.float64


def compare_incoming_duals(inputs, actual, expected, selected='all', create_graph=False):
    actual = (actual,) if isinstance(actual, torch.Tensor) else actual
    expected = (expected,) if isinstance(expected, torch.Tensor) else expected
    with forward_ad.dual_level():
        weights = []
        for i, value in enumerate(actual):
            positions = torch.arange(value.numel(), dtype=DTYPE).reshape_as(value)
            weight, tangent = torch.cos(positions+.2), torch.sin(positions+.4)
            weights.append(forward_ad.make_dual(weight, tangent) if selected in (i, 'all') else weight)
        ga = torch.autograd.grad(actual, inputs, tuple(weights), create_graph=create_graph)
        ge = torch.autograd.grad(expected, inputs, tuple(weights), create_graph=create_graph)
        nonzero_reference = False
        for a, e in zip(ga, ge):
            pa, ta = forward_ad.unpack_dual(a)
            pe, te = forward_ad.unpack_dual(e)
            torch.testing.assert_close(pa, pe, rtol=2e-10, atol=2e-11)
            ta = torch.zeros_like(pa) if ta is None else ta
            te = torch.zeros_like(pe) if te is None else te
            nonzero_reference |= bool(te.abs().max() > 1e-12)
            torch.testing.assert_close(ta, te, rtol=2e-10, atol=2e-11)
        assert nonzero_reference


@pytest.mark.parametrize('kind', ['assembly', 'apply'])
@pytest.mark.parametrize('create_graph', [False, True])
def test_full_operator_incoming_dual_weights(kind, create_graph):
    if kind == 'assembly':
        xs = assembly_inputs()
        actual = torch.ops.quantlibtorch._hook_diffusion(*xs, False, 0)
        expected = torch.ops.quantlibtorch._hook_diffusion(*xs, False, 0, 0)
    else:
        xs = apply_inputs()
        actual = torch.ops.quantlibtorch._hook_apply(*xs, 4, False)
        expected = torch.ops.quantlibtorch._hook_apply(*xs, 0, False)
    compare_incoming_duals(xs, actual, expected, create_graph=create_graph)


@pytest.mark.parametrize('n,distance,nrhs', [(3, 1, 2), (8, 4, 3)])
@pytest.mark.parametrize('selected', [0, 1, 2, 3, 'all'])
@pytest.mark.parametrize('create_graph', [False, True])
def test_pcr_stage_each_incoming_dual(n, distance, nrhs, selected, create_graph):
    xs = stage_inputs(n, nrhs)
    compare_incoming_duals(xs, stage(xs, distance), stage_reference(*xs, distance),
                           selected=selected, create_graph=create_graph)


@pytest.mark.parametrize('n', [3, 9])
@pytest.mark.parametrize('nrhs', [1, 3])
def test_complete_pcr_incoming_dual_against_dense(n, nrhs):
    xs = stage_inputs(n, nrhs)
    xs = (xs[0][1:].detach().requires_grad_(), xs[1], xs[2][:-1].detach().requires_grad_(), xs[3])
    matrix = torch.diag(xs[1])+torch.diag(xs[0], -1)+torch.diag(xs[2], 1)
    compare_incoming_duals(xs, tridiagonal_solve_pcr(*xs), torch.linalg.solve(matrix, xs[3]))


@pytest.mark.parametrize('kernels', ['auto', 'pcr'])
def test_financial_full_incoming_dual_against_original(kernels):
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    actual = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels=kernels, mode='full')
    expected = fd_vanilla(sigma, time_steps=2, grid_points=11, kernels='scalar', mode='full')
    compare_incoming_duals((sigma,), actual, expected)
