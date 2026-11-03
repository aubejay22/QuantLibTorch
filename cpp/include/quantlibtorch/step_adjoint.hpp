#pragma once
#include <ql/math/array.hpp>

namespace QuantLib {
class FdmLinearOpComposite;
class BoundaryConditionSchemeHelper;
}

namespace quantlibtorch {
// Only a fixed-time, one-dimensional Black-Scholes operator with no boundary
// condition callbacks is eligible. State and coefficient graphs remain inputs
// to the explicit first-order VJP, or the full-mode tensor composition under
// its separate switch. Kinds: Douglas=0, implicit=1, explicit=2.
bool try_step_adjoint(QuantLib::FdmLinearOpComposite& map,
    const QuantLib::BoundaryConditionSchemeHelper& boundaries, QuantLib::Array& state,
    QuantLib::Time dt, QuantLib::Real theta, QuantLib::Time time, int kind);
}
