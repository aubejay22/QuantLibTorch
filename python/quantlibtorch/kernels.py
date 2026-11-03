"""Tensor numerical kernels with explicit batch, node, and RHS axes.

These are numerical primitives, not a batched pricing engine. They use whole
tensor ATen operations on CPU/CUDA. Financial batch plans are exposed separately
by LatentBatch and MultiAssetBatch.
"""
import torch

from . import solver as _solver  # register the shared native dispatcher library


def tridiagonal_apply(lower: torch.Tensor, diagonal: torch.Tensor,
                      upper: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Apply tridiagonal operators to one or more right-hand sides.

    Bands have shapes ``(..., N-1)``, ``(..., N)``, ``(..., N-1)``; ``rhs``
    has shape ``(..., N, R)``, including an explicit final axis for R=1.
    Leading batch dimensions broadcast using PyTorch rules. For example,
    ``(B, N)`` diagonals and ``(B, N, R)`` values represent B operators, while
    ``(N,)`` diagonals share one operator across the batch. N >= 2 and R >= 1.

    All inputs must be dense float64 tensors on the same CPU or CUDA device.
    Noncontiguous views, shared coefficients, and higher derivatives work via
    composite autograd. Inputs are never mutated. As with PyTorch arithmetic,
    nonfinite values propagate; no value scan or host synchronization occurs.
    A batch member can represent a contract's numerical system, but this
    function does not build meshes, schedules, payoffs, or option prices.
    """
    return torch.ops.quantlibtorch.tridiagonal_apply(lower, diagonal, upper, rhs)


def band_axpyb(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
              b: torch.Tensor) -> torch.Tensor:
    """Assemble row bands ``y + a*x + diag(b)`` with broadcast batch axes.

    ``x`` and ``y`` have shape ``(..., 3, N)`` in lower/diagonal/upper order.
    ``a`` and ``b`` have shape ``(..., N)`` or ``(..., 1)``; b shifts only the
    central diagonal. Endpoints are retained, as in QuantLib row storage;
    to use with tridiagonal_apply, take lower[..., 1:] and upper[..., :-1].
    Device, dtype, autograd, and nonfinite policies match tridiagonal_apply.
    """
    return torch.ops.quantlibtorch.band_axpyb(a, x, y, b)


def tridiagonal_solve_pcr(lower: torch.Tensor, diagonal: torch.Tensor,
                          upper: torch.Tensor, rhs: torch.Tensor, *,
                          check_pivots: bool = True) -> torch.Tensor:
    """Experimental parallel cyclic reduction with full autograd.

    Shapes/device/dtype follow tridiagonal_apply. Leading axes and multiple
    RHS broadcast, including shared matrices. CPU stages fuse the elementary
    elimination and its first backward; higher derivatives use graph-connected
    tensor chain rules. CUDA and forward AD retain the tensor recurrence.
    QLT_PCR_NATIVE_STAGE=0 selects the tensor recurrence on CPU as well.
    Function transforms and fake/wrapped tensors retain the tensor path. For
    one RHS, the tensor path also avoids expanding shared matrix batches and
    handles workloads exceeding 32,768 broadcast rows (a measured CPU cutoff).

    PCR requires nonzero diagonal pivots at EVERY parallel elimination stage;
    its admissible pivot patterns differ from upstream Thomas elimination.
    Nonsingularity alone is insufficient. Well-scaled, strictly diagonally
    dominant systems are the intended initial workload. Floating-point ordering
    differs from QuantLib's recurrence. The original solver remains available.

    With check_pivots=True, nonfinite inputs, zero/nonfinite pivots or nonfinite
    output raise. These checks synchronize CUDA at each stage. Disabling checks
    avoids host reads; failures then propagate as nonfinite arithmetic, and the
    caller is responsible for admissibility and residual/error checks.
    """
    return torch.ops.quantlibtorch.tridiagonal_solve_pcr(lower, diagonal, upper, rhs, check_pivots)
