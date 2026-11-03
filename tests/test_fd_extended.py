from datetime import date
import pytest
import torch
from quantlibtorch import fd_vanilla
from test_fd_engine import stock, SMALL, MID, EXPIRY


@pytest.mark.parametrize("scheme", ["craig_sneyd", "modified_craig_sneyd", "hundsdorfer"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
@pytest.mark.parametrize("exercise", ["european", "american", "bermudan"])
def test_additional_schemes_with_events_match_stock(scheme, mode, exercise):
    kwargs = dict(scheme=scheme, exercise=exercise, option_type="put", spot=94.,
                  cash_dividends=[(MID, 2.)], **SMALL)
    if exercise == "bermudan":
        kwargs["exercise_dates"] = [MID, EXPIRY]
    vol = torch.tensor(.23, dtype=torch.float64, requires_grad=True)
    value = fd_vanilla(vol, mode=mode, **kwargs)
    vega, = torch.autograd.grad(value, vol)
    expected = stock(volatility=.23, bump=1e-5, **kwargs)
    assert value.item() == pytest.approx(expected["price"], abs=3e-10)
    assert vega.item() == pytest.approx(expected["parallel_vol_shift_derivative"], rel=4e-6, abs=3e-6)


@pytest.mark.parametrize("exercise", ["european", "american", "bermudan"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_escrowed_dividends_match_stock(exercise, mode):
    kwargs = dict(cash_dividend_model="escrowed", cash_dividends=[(MID, 2.1)],
                  exercise=exercise, option_type="put", **SMALL)
    if exercise == "bermudan":
        kwargs["exercise_dates"] = [MID, EXPIRY]
    vol = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    value = fd_vanilla(vol, mode=mode, **kwargs)
    vega, = torch.autograd.grad(value, vol)
    expected = stock(volatility=.21, bump=1e-5, **kwargs)
    assert value.item() == pytest.approx(expected["price"], abs=3e-10)
    assert vega.item() == pytest.approx(expected["parallel_vol_shift_derivative"], rel=4e-6, abs=3e-6)
