"""A small differentiable PDE experiment using QuantLib's native linear solver.

This module assembles a log-space Black--Scholes/local-volatility PDE in Python.
It does not wrap an existing QuantLib instrument, curve, or pricing engine.
Only the supplied local-volatility samples are differentiable; financial inputs
and the mesh are fixed Python numbers.  Both differentiation modes execute the
compiled QuantLib tridiagonal solver at every backward-Euler time step.
"""

from __future__ import annotations

import math
import numbers

import torch

from .solver import tridiagonal_solve


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a fixed real number, not a tensor")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def log_moneyness_grid(
    interior_nodes: int,
    *,
    log_moneyness_min: float = -1.5,
    log_moneyness_max: float = 1.5,
) -> torch.Tensor:
    """Return the interior ``log(S / strike)`` nodes as a CPU float64 tensor."""
    if isinstance(interior_nodes, bool) or not isinstance(interior_nodes, int):
        raise TypeError("interior_nodes must be an integer")
    if interior_nodes < 2:
        raise ValueError("at least two interior nodes are required")
    x_min = _finite_number("log_moneyness_min", log_moneyness_min)
    x_max = _finite_number("log_moneyness_max", log_moneyness_max)
    if x_min >= x_max:
        raise ValueError("log_moneyness_min must be less than log_moneyness_max")
    return torch.linspace(
        x_min, x_max, interior_nodes + 2, dtype=torch.float64, device="cpu"
    )[1:-1]


def european_call(
    local_vol: torch.Tensor,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.03,
    dividend: float = 0.01,
    maturity: float = 1.0,
    log_moneyness_min: float = -1.5,
    log_moneyness_max: float = 1.5,
    mode: str = "full",
) -> torch.Tensor:
    """Price a European call with a sampled local-volatility field.

    ``local_vol`` must be a finite, strictly positive CPU float64 tensor of
    shape ``(time_steps, interior_nodes)``.  Row ``i`` holds volatility at
    calendar time ``i * maturity / time_steps`` on :func:`log_moneyness_grid`.
    Rows are ordered from valuation toward expiry; rollback visits them in
    reverse order.  Arbitrary neural-network outputs can supply these samples.

    The generator in ``x = log(S / strike)`` is
    ``0.5 * sigma**2 * dxx + (rate-dividend-0.5*sigma**2) * dx - rate``.
    Spatial derivatives use centered differences and time uses backward Euler.
    The lower boundary is zero; the upper boundary is the positive part of
    ``S_max * exp(-dividend*tau) - strike * exp(-rate*tau)``.  The returned
    scalar is linearly interpolated at ``spot``.  Boundary truncation, spatial
    resolution, and time discretization errors remain the caller's concern;
    this prototype is not a universal pricing engine.

    The centered stencil must have nonnegative off-diagonals and the implicit
    discount step must remain positive. Unsupported drift-dominated meshes
    raise an error requiring refinement instead of returning negative prices.

    ``mode='full'`` differentiates QuantLib's elimination operations.
    ``mode='adjoint'`` uses a custom transpose-solve VJP per time step.
    Assembly, boundary contributions, and dependencies between steps remain
    in PyTorch's graph in both cases.  Parameters other than ``local_vol``
    are fixed numbers: this API does not provide their Greeks.
    """
    if not isinstance(local_vol, torch.Tensor):
        raise TypeError("local_vol must be a torch.Tensor")
    if local_vol.device.type != "cpu" or local_vol.dtype != torch.float64:
        raise ValueError("local_vol must use CPU float64")
    if local_vol.layout != torch.strided or local_vol.ndim != 2:
        raise ValueError("local_vol must be a dense tensor of shape (steps, nodes)")
    steps, nodes = local_vol.shape
    if steps < 1 or nodes < 2:
        raise ValueError("local_vol requires at least one time step and two nodes")
    if not bool(torch.isfinite(local_vol).all()) or not bool((local_vol > 0).all()):
        raise ValueError("local_vol must contain finite, strictly positive values")
    if mode not in ("full", "adjoint"):
        raise ValueError("mode must be 'full' or 'adjoint'")

    spot = _finite_number("spot", spot)
    strike = _finite_number("strike", strike)
    rate = _finite_number("rate", rate)
    dividend = _finite_number("dividend", dividend)
    maturity = _finite_number("maturity", maturity)
    x_min = _finite_number("log_moneyness_min", log_moneyness_min)
    x_max = _finite_number("log_moneyness_max", log_moneyness_max)
    if spot <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("spot, strike, and maturity must be strictly positive")
    x = log_moneyness_grid(
        nodes, log_moneyness_min=x_min, log_moneyness_max=x_max
    )
    x_spot = math.log(spot) - math.log(strike)
    if not x_min < x_spot < x_max:
        raise ValueError("spot must lie strictly inside the log-moneyness domain")

    dt = maturity / steps
    dx = (x_max - x_min) / (nodes + 1)
    if 1 + dt * rate <= 0:
        raise ValueError("time step is too large for this negative rate; increase time steps")
    with torch.no_grad():
        variance_check = local_vol.square()
        if bool((variance_check / dx < (rate - dividend - 0.5 * variance_check).abs()).any()):
            raise ValueError(
                "centered stencil is not monotone for these coefficients; "
                "refine the spatial mesh (increase nodes or narrow the domain)"
            )
    values = torch.clamp(strike * torch.expm1(x), min=0)
    lower_boundary = local_vol.new_zeros(())
    s_max = strike * math.exp(x_max)
    for i in range(steps - 1, -1, -1):
        tau = (steps - i) * dt
        upper_boundary = local_vol.new_tensor(
            max(s_max * math.exp(-dividend * tau) - strike * math.exp(-rate * tau), 0)
        )
        variance = local_vol[i].square()
        diffusion = 0.5 * variance / (dx * dx)
        drift = (rate - dividend - 0.5 * variance) / (2 * dx)
        lower_generator = diffusion - drift
        upper_generator = diffusion + drift
        diagonal = 1 + dt * (2 * diffusion + rate)
        # A = I - dt*L; move known Dirichlet values from A to the RHS.
        boundary_rhs = torch.cat(
            (
                dt * lower_generator[:1] * lower_boundary,
                local_vol.new_zeros(nodes - 2),
                dt * upper_generator[-1:] * upper_boundary,
            )
        )
        values = tridiagonal_solve(
            -dt * lower_generator[1:],
            diagonal,
            -dt * upper_generator[:-1],
            values + boundary_rhs,
            mode=mode,
        )

    grid_position = (x_spot - x_min) / dx
    left = min(int(math.floor(grid_position)), nodes)
    weight = grid_position - left
    with_boundaries = torch.cat(
        (lower_boundary.reshape(1), values, upper_boundary.reshape(1))
    )
    return (1 - weight) * with_boundaries[left] + weight * with_boundaries[left + 1]
