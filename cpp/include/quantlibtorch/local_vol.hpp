#pragma once

#include <ql/math/array.hpp>

namespace QuantLib { class LocalVolTermStructure; }
namespace quantlibtorch {
// Returns false for other term structures or differentiable coordinates;
// those cases retain QuantLib's scalar sampling and exception policy.
bool sample_local_variances(const QuantLib::LocalVolTermStructure* local_vol,
    QuantLib::Time time, const QuantLib::Array& spots, QuantLib::Array& variance);
}
