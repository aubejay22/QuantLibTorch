#pragma once

#include <ql/math/array.hpp>
#include <array>
#include <tuple>

namespace QuantLib { class TripleBandLinearOp; }

namespace quantlibtorch {

// Equation VJP for the actual upstream directional solve. reverse_index maps
// the solver's flattened elimination order to the original mesh layout.
// Coefficients and scales remain graph-connected outside the custom forward.
// CPU float64, first-order reverse AD; no dense matrix or elimination tape.
QuantLib::Array directional_adjoint_split(
    const QuantLib::TripleBandLinearOp& op, const QuantLib::Array& rhs,
    QuantLib::Real a, QuantLib::Real b, const QuantLib::Real* lower,
    const QuantLib::Real* diagonal, const QuantLib::Real* upper,
    const QuantLib::Size* reverse_index);

// A square indexed stencil has coefficients/indices [K,N] and state/seed [N].
// Each output row is the ordered sum coefficients[k,i]*state[indices[k,i]].
// The transpose accumulates duplicate destinations, including boundary folds.
// Functional tensor operations retain coefficient, state and higher derivatives.
at::Tensor structured_stencil_apply(const at::Tensor& coefficients,
    const at::Tensor& indices, const at::Tensor& state);
at::Tensor structured_stencil_transpose(const at::Tensor& coefficients,
    const at::Tensor& indices, const at::Tensor& seed);
std::tuple<at::Tensor, at::Tensor> structured_stencil_vjp(
    const at::Tensor& coefficients, const at::Tensor& indices,
    const at::Tensor& state, const at::Tensor& seed);

// Upstream NinePointLinearOp order: 00,01,02,10,11,12,20,21,22.
// A null neighbor pointer denotes the identity indices (the 11/center term).
// The wrapper owns tensor copies of topology and retains original coefficient
// tensors; no borrowed operator pointer is kept by its autograd graph.
QuantLib::Array nine_point_apply(const QuantLib::Array& rhs,
    const std::array<const QuantLib::Real*, 9>& coefficients,
    const std::array<const QuantLib::Size*, 9>& indices);

} // namespace quantlibtorch
