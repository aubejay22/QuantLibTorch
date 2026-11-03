"""Reusable Heston model configuration and graph lifetime."""
from datetime import date

import pytest
import torch

from quantlibtorch import (EquityMarket, FdHestonEngine, HestonMesh, HestonModel,
                          PricingLayer, VanillaOption, fd_heston, heston_mesh)


def test_model_parameters_buffers_and_state_dict():
    model = HestonModel()
    assert set(dict(model.named_parameters())) == {"v0", "kappa", "theta", "sigma", "rho"}
    assert all(p.dtype == torch.float64 for p in model())
    assert set(model.state_dict()) == set(dict(model.named_parameters()))
    fixed = HestonModel(trainable=False)
    assert not list(fixed.parameters())
    assert len(list(fixed.buffers())) == 5
    fixed.load_state_dict(model.state_dict())
    for a, b in zip(model(), fixed()):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kwargs", [{"v0": 0.}, {"rho": -1.}, {"sigma": -.2},
                                    {"kappa": float("nan")}, {"theta": torch.tensor(.04)}])
def test_model_rejects_invalid_initialization(kwargs):
    with pytest.raises((ValueError, TypeError)):
        HestonModel(**kwargs)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_layer_fresh_graph_and_market_overrides(mode):
    mesh = heston_mesh(x_grid=11, v_grid=7, time_steps=3)
    model = HestonModel()
    engine = FdHestonEngine(mesh, time_steps=3, mode=mode)
    layer = PricingLayer(VanillaOption(100., date(2026, 1, 1)), EquityMarket(),
                         engine, model=model)
    assert len(list(layer.parameters())) == 5
    for _ in range(2):
        rate = torch.tensor(.032, dtype=torch.float64, requires_grad=True)
        price = layer(rate=rate)
        expected = fd_heston(*model(), mesh=mesh, time_steps=3, mode=mode, rate=rate)
        torch.testing.assert_close(price, expected, rtol=0, atol=0)
        grads = torch.autograd.grad(price, (*model(), rate))
        assert all(torch.isfinite(g) for g in grads)
        assert grads[-1].abs() > 0.


def test_mesh_configuration_is_passive_and_copied():
    x = [3., 4., 4.5, 5., 6.]
    mesh = HestonMesh(x, [0., .01, .04, .1, .3])
    x[0] = 2.
    assert mesh.log_spots[0] == 3.
    with pytest.raises(TypeError, match="tensor"):
        HestonMesh(torch.tensor(x), mesh.variances)
    with pytest.raises(ValueError, match="zero"):
        HestonMesh(x, [.001, .01, .04, .1, .3])
    with pytest.raises((TypeError, ValueError)):
        heston_mesh(v0=torch.tensor(.04, dtype=torch.float64))
