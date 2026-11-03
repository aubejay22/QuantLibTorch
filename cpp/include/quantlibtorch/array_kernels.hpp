#pragma once

#include <ql/types.hpp>

namespace quantlibtorch {
// Called immediately before upstream std::transform. Inputs are fully packed
// before any destination handles are rebound, including overlapping operands.
// False leaves the original operation in charge (passive arrays / ablation).
bool array_binary(const QuantLib::Real* first, const QuantLib::Real* last,
                  const QuantLib::Real* second, QuantLib::Real* output, char op);
bool array_scalar(const QuantLib::Real* first, const QuantLib::Real* last,
                  QuantLib::Real* output, QuantLib::Real scalar, char op, bool reverse);
}
