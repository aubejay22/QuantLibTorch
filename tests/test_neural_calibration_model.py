"""Model restrictions and genuine joint optimizer updates in the example."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from quantlibtorch import state_grid

spec = importlib.util.spec_from_file_location("calibration_example", Path(__file__).parents[1]/"examples/calibrate_neural_sde.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("m", [1, 2])
def test_neural_variance_covariance_and_boundary_structure(m):
    model = example.NeuralVarianceModel(m)
    axes = [[math.log(s) for s in [30., 100., 300.]]]+[[0., .04, .5]]*m
    x = state_grid(axes)
    times = torch.tensor([0., .3, 1.], dtype=torch.float64)
    drift, factor = model(times, x)
    covariance = factor@factor.transpose(-1,-2)
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-14
    for i in range(m):
        zero = x[:, i+1] == 0
        high = x[:, i+1] == .5
        assert (drift[:, zero, i] > 0).all()
        assert (drift[:, high, i] < 0).all()
        assert (factor[:, zero, i+1] == 0).all()
    torch.testing.assert_close(covariance[...,0,0],factor[...,0,0].square(),rtol=0,atol=0)
    assert (model.initial_latent()>0).all() and (model.initial_latent()<.12).all()


def test_both_modes_jointly_optimize_network_and_initial_state():
    data = dict(valuation_date="2025-01-01",spot=100.,rate=.03,dividend_yield=.01,times=[0.,.5,1.],
        meshes={"small":dict(x=[math.log(s) for s in [40.,65.,85.,100.,120.,160.,220.]],
                             v=[0.,.02,.05,.2],steps_per_year=3)},
        quotes=[dict(strike=95.,maturity="2025-04-01",price=7.2),
                dict(strike=105.,maturity="2026-01-01",price=7.3)])
    results = [example.train(data,mode=mode,iterations=3,level="small") for mode in ["full","adjoint"]]
    for result in results:
        assert result["final_rmse"] < result["history"][0]["price_rmse"]
        assert result["parameters"]["raw_initial"] != result["initial_parameters"]["raw_initial"]
        assert result["parameters"]["net.2.weight"] != result["initial_parameters"]["net.2.weight"]
    torch.testing.assert_close(torch.tensor(results[0]["final_prices"]),torch.tensor(results[1]["final_prices"]),rtol=1e-6,atol=1e-6)
