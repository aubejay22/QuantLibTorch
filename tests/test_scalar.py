"""Scalar compatibility invariants required by QuantLib financial engines."""
import math
import sys

import pytest
import torch

import quantlibtorch  # registers the native adapter checks


def test_quantlib_null_is_the_upstream_sentinel_not_zero():
    real_null, integer_null = torch.ops.quantlibtorch._scalar_null()
    assert real_null == torch.finfo(torch.float32).max
    assert integer_null == 2**31 - 1


@pytest.mark.parametrize("value,exponent", [
    (sys.float_info.max, 0),
    (math.frexp(sys.float_info.max)[0], 1024),
    (math.ldexp(1., -100), 1100),
    (1e300, -2000),
    (1.5, -1074),
    (math.ulp(0.), 1074),
    (math.ulp(0.), 1024),
    (-0., 2**31-1),
    (math.inf, -2**31),
    (-math.inf, 2**31-1),
    (math.nan, 0),
    (1., 2**31-1),
    (1., -2**31),
])
def test_ldexp_values_match_standard_double_at_extremes(value, exponent):
    try:
        expected = math.ldexp(value, exponent)
    except OverflowError:
        expected = math.copysign(math.inf, value)
    actual = torch.ops.quantlibtorch._scalar_ldexp(
        torch.tensor(value, dtype=torch.float64), exponent
    ).item()
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected
        assert math.copysign(1., actual) == math.copysign(1., expected)


@pytest.mark.parametrize("value", [sys.float_info.max, math.ulp(0.), -6.0, -0.0])
def test_frexp_ldexp_round_trip(value):
    mantissa, exponent = torch.ops.quantlibtorch._scalar_frexp(
        torch.tensor(value, dtype=torch.float64)
    )
    assert (mantissa.item(), exponent) == math.frexp(value)
    result = torch.ops.quantlibtorch._scalar_ldexp(mantissa, exponent).item()
    assert result == value
    assert math.copysign(1., result) == math.copysign(1., value)


def test_exponent_helpers_preserve_normal_range_autograd():
    value = torch.tensor(6., dtype=torch.float64, requires_grad=True)

    def roundtrip(x):
        mantissa, exponent = torch.ops.quantlibtorch._scalar_frexp(x)
        return torch.ops.quantlibtorch._scalar_ldexp(mantissa, exponent)

    assert torch.autograd.gradcheck(roundtrip, (value,))
    assert torch.autograd.gradcheck(
        lambda x: torch.ops.quantlibtorch._scalar_ldexp(x, -3), (value,)
    )
    # Standalone exponent derivatives can exceed float64 at extreme exponents;
    # that representational limit is separate from the value checks above.
    torch.testing.assert_close(torch.autograd.grad(roundtrip(value), value)[0], value.new_tensor(1.))


@pytest.mark.parametrize("value,expected", [
    (3.9, 3), (-3.9, -3), (float(-2**63), -2**63),
    (math.nextafter(float(2**63), 0.), 2**63-1024),
])
def test_explicit_integral_conversion(value, expected):
    assert torch.ops.quantlibtorch._scalar_integer(torch.tensor(value, dtype=torch.float64)) == expected


@pytest.mark.parametrize("value", [float(2**63), math.nextafter(float(-2**63), -math.inf), math.inf, math.nan])
def test_integral_conversion_rejects_overflow_and_nonfinite(value):
    with pytest.raises(RuntimeError, match="range|nonfinite"):
        torch.ops.quantlibtorch._scalar_integer(torch.tensor(value, dtype=torch.float64))


def test_integral_conversion_rejects_differentiable_input():
    with pytest.raises(RuntimeError, match="differentiable"):
        torch.ops.quantlibtorch._scalar_integer(torch.tensor(3., dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize("tensor_constant", [False, True])
def test_mixed_scalar_arithmetic_values_vjps_and_copy_semantics(tensor_constant):
    x = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    c = 2.7
    actual, passive = torch.ops.quantlibtorch._scalar_arithmetic(x, c, tensor_constant)
    expected = torch.stack((x+c, c-x, x*c, c/x, x**c, c**x, x.exp().log(),
                            x/c, x, x.new_tensor(c*c-c)))
    assert not passive
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    weights = torch.linspace(-.8, 1.1, actual.numel(), dtype=x.dtype)
    torch.testing.assert_close(torch.autograd.grad(actual, x, weights)[0],
                               torch.autograd.grad(expected, x, weights)[0])
    assert x.item() == 1.3  # Compound updates must not mutate an alias or input.
    fn = lambda value: torch.ops.quantlibtorch._scalar_arithmetic(value, c, tensor_constant)[0]
    assert torch.autograd.gradcheck(fn, (x,))
    assert torch.autograd.gradgradcheck(fn, (x,))


@pytest.mark.parametrize("requires_grad", [False, True])
def test_tensor_constructor_never_demotes_to_passive(requires_grad):
    x = torch.tensor(1.3, dtype=torch.float64, requires_grad=requires_grad)
    with torch.no_grad():
        result, passive = torch.ops.quantlibtorch._scalar_arithmetic(x, 2., False)
    assert not passive
    assert not result.requires_grad
