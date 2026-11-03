#pragma once

#include <ql/math/array.hpp>

namespace QuantLib { class TripleBandLinearOp; }

namespace quantlibtorch {

// Owned by the same object as one immutable pair of spatial operators. Only
// passive scalar constants may be retained; tensor-valued bases are repacked.
// No global cache or borrowed pointer outlives the owning QuantLib operator.
class DiffusionBasisCache {
    friend bool try_diffusion_assembly(QuantLib::TripleBandLinearOp&,
        const QuantLib::TripleBandLinearOp&, const QuantLib::TripleBandLinearOp&,
        const QuantLib::Array&, QuantLib::Real, QuantLib::Real, DiffusionBasisCache*, at::Tensor*);
    at::Tensor first_, second_;
    bool checked_ = false, passive_ = false;
};

// The mode belongs to one synchronous pricing call, never to a process-wide
// flag. Nested solves restore the previous mode, including on exceptions.
bool adjoint_enabled();
bool operator_adjoint_enabled();
bool operator_arithmetic_enabled();
class AdjointMode {
  public:
    explicit AdjointMode(bool enabled);
    ~AdjointMode();
    AdjointMode(const AdjointMode&) = delete;
    AdjointMode& operator=(const AdjointMode&) = delete;
  private:
    bool previous_;
};

QuantLib::Array adjoint_split(const QuantLib::TripleBandLinearOp& op,
    const QuantLib::Array& rhs, QuantLib::Real a, QuantLib::Real b,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal,
    const QuantLib::Real* upper);

QuantLib::Array adjoint_tridiagonal(const QuantLib::Array& lower, const QuantLib::Array& diagonal,
    const QuantLib::Array& upper, const QuantLib::Array& rhs);

QuantLib::Array adjoint_apply(const QuantLib::TripleBandLinearOp& op, const QuantLib::Array& rhs,
    const QuantLib::Real* lower, const QuantLib::Real* diagonal, const QuantLib::Real* upper,
    const QuantLib::Size* left, const QuantLib::Size* right);

// The same full-AD elementary application, without an intermediate Array.
bool try_full_apply_tensor(const QuantLib::TripleBandLinearOp& op,
    const at::Tensor& lower, const at::Tensor& diagonal, const at::Tensor& upper,
    const at::Tensor& rhs, const QuantLib::Size* left, const QuantLib::Size* right,
    at::Tensor& output);

// An optional original-result owner is cleared on every attempt and receives
// the active matrix only after successful assembly. Its caller must also clear
// it before any alternative assembly route or operation that can change output.
bool try_diffusion_assembly(QuantLib::TripleBandLinearOp& output,
    const QuantLib::TripleBandLinearOp& dx, const QuantLib::TripleBandLinearOp& dxx,
    const QuantLib::Array& variance, QuantLib::Real rate, QuantLib::Real yield,
    DiffusionBasisCache* immutable_basis = nullptr, at::Tensor* assembled_bands = nullptr);

} // namespace quantlibtorch
