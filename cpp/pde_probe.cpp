#include <quantlibtorch/adjoint.hpp>
#include <quantlibtorch/tensor_kernels.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/predefined1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondordermixedderivativeop.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <torch/library.h>
#include <ATen/record_function.h>
#include <cmath>

namespace {
using namespace QuantLib;
using Doubles = c10::List<double>;

Array tensor_array(const at::Tensor& values) {
    const auto flat = values.reshape({-1});
    Array result(flat.numel());
    quantlibtorch::unpack_array(flat, result.begin());
    return result;
}

Array field(const at::Tensor& coefficients, int64_t component) {
    return tensor_array(coefficients.select(0, component));
}

// A coefficient adapter for QuantLib's existing spatial operators. It owns
// graph-connected fields through the assembled operators. Axes and fields
// are fixed in time; no scalar coefficient is detached or updated in place.
// L = bx Dx + by Dy + ax Dxx + ay Dyy + mixed Dxy - killing I.
// Each directional part carries half of the killing term.
class CoefficientPdeOp final : public FdmLinearOpComposite {
  public:
    CoefficientPdeOp(const ext::shared_ptr<FdmMesher>& mesher,
                     const at::Tensor& coefficients)
    : x_(directional(mesher, coefficients, 0)),
      y_(directional(mesher, coefficients, 1)),
      mixed_(SecondOrderMixedDerivativeOp(0, 1, mesher).mult(field(coefficients, 4))) {}

    Size size() const override { return 2; }
    void setTime(Time t1, Time t2) override {
        QL_REQUIRE(t1 <= t2, "invalid PDE time interval");
    }
    Array apply(const Array& state) const override {
        return x_.apply(state) + y_.apply(state) + mixed_.apply(state);
    }
    Array apply_mixed(const Array& state) const override {
        return mixed_.apply(state);
    }
    Array apply_direction(Size direction, const Array& state) const override {
        return directional(direction).apply(state);
    }
    Array solve_splitting(Size direction, const Array& rhs, Real scale) const override {
        return directional(direction).solve_splitting(rhs, scale, 1.0);
    }
    Array preconditioner(const Array& rhs, Real scale) const override {
        return solve_splitting(1, solve_splitting(0, rhs, scale), scale);
    }

  private:
    static TripleBandLinearOp directional(const ext::shared_ptr<FdmMesher>& mesher,
                                         const at::Tensor& coefficients, Size direction) {
        return FirstDerivativeOp(direction, mesher).mult(field(coefficients, direction))
            .add(SecondDerivativeOp(direction, mesher).mult(field(coefficients, direction+2)))
            .add(-0.5*field(coefficients, 5));
    }
    const TripleBandLinearOp& directional(Size direction) const {
        QL_REQUIRE(direction < 2, "PDE direction must be zero or one");
        return direction == 0 ? x_ : y_;
    }
    TripleBandLinearOp x_, y_;
    NinePointLinearOp mixed_;
};

std::vector<Real> fixed_axis(const Doubles& axis, const char* name) {
    TORCH_CHECK(axis.size() >= 2, name, " must contain at least two nodes");
    std::vector<Real> values;
    values.reserve(axis.size());
    double previous = 0.0;
    for (size_t i = 0; i < axis.size(); ++i) {
        const double value = axis.get(i);
        TORCH_CHECK(std::isfinite(value), name, " nodes must be finite");
        TORCH_CHECK(i == 0 || value > previous, name, " nodes must strictly increase");
        values.emplace_back(value);
        previous = value;
    }
    return values;
}

FdmSchemeDesc selected_scheme(int64_t scheme) {
    switch (scheme) {
      case 0: return FdmSchemeDesc::Douglas();
      case 1: return FdmSchemeDesc::Hundsdorfer();
      case 2: return FdmSchemeDesc::ModifiedCraigSneyd();
      case 3: return FdmSchemeDesc::CraigSneyd();
      default: TORCH_CHECK(false, "PDE scheme must be 0, 1, 2 or 3");
    }
}

at::Tensor pde2d_probe(const at::Tensor& coefficients, const at::Tensor& terminal,
    const Doubles& x, const Doubles& y, double horizon, int64_t steps,
    int64_t scheme, bool adjoint) {
    RECORD_FUNCTION("qlt::pde2d_forward", std::vector<c10::IValue>());
    for (const auto& value : {coefficients, terminal}) {
        TORCH_CHECK(value.device().is_cpu() && value.scalar_type() == at::kDouble
                    && value.layout() == at::kStrided, "PDE inputs must be dense CPU float64 tensors");
        TORCH_CHECK(at::isfinite(value).all().item<bool>(), "PDE inputs must be finite");
    }
    TORCH_CHECK(coefficients.dim() == 3 && coefficients.size(0) == 6 && terminal.dim() == 2,
                "PDE shapes must be coefficients [6, ny, nx] and terminal [ny, nx]");
    TORCH_CHECK(coefficients.size(1) == static_cast<int64_t>(y.size())
                && coefficients.size(2) == static_cast<int64_t>(x.size())
                && terminal.size(0) == coefficients.size(1) && terminal.size(1) == coefficients.size(2),
                "PDE tensor dimensions must match fixed axes");
    TORCH_CHECK(std::isfinite(horizon) && horizon > 0.0, "PDE horizon must be finite and positive");
    TORCH_CHECK(steps > 0, "PDE steps must be positive");
    const auto x_values = fixed_axis(x, "x"), y_values = fixed_axis(y, "y");
    const auto descriptor = selected_scheme(scheme);
    quantlibtorch::AdjointMode mode(adjoint);
    quantlibtorch::TensorKernelMode kernel_mode(4);
    const auto mesher = ext::make_shared<FdmMesherComposite>(
        ext::make_shared<Predefined1dMesher>(x_values),
        ext::make_shared<Predefined1dMesher>(y_values));
    const auto op = ext::make_shared<CoefficientPdeOp>(mesher, coefficients);
    auto state = tensor_array(terminal);
    // Empty callback boundaries retain QuantLib's one-sided first derivative
    // and zero second-derivative boundary rows. No implicit-Euler damping:
    // that would invoke a multidimensional Krylov solve with a separate contract.
    FdmBackwardSolver solver(op, {}, {}, descriptor);
    solver.rollback(state, horizon, 0.0, static_cast<Size>(steps), 0);
    auto result = quantlibtorch::pack_array(state.begin(), state.size()).reshape(terminal.sizes());
    TORCH_CHECK(at::isfinite(result).all().item<bool>(), "non-finite PDE solution");
    return result;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("_pde2d_probe(Tensor coefficients, Tensor terminal, float[] x, float[] y, "
          "float horizon, int steps, int scheme, bool adjoint) -> Tensor");
}
TORCH_LIBRARY_IMPL(quantlibtorch, CompositeImplicitAutograd, m) {
    m.impl("_pde2d_probe", TORCH_FN(pde2d_probe));
}
