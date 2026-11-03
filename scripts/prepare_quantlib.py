"""Generate auditable scalar compatibility overlays; never edit the checkout.

Only sources in the explicit CMake manifest and their transitive QuantLib
headers are considered. Numerical algorithms are unchanged by math-adl.patch;
the optional solve dispatch is recorded separately in adjoint-hook.patch.
Profiler scopes are recorded separately in profiling-scopes.patch.
Vector application/assembly dispatch is recorded in tensor-kernels.patch.
Elementwise arrays, sampling and requested engine outputs have separate patches.
Multidimensional solve and stencil hooks are in structured-operators.patch.
Explicit fixed geometry and NPV-only engine hooks are in engine-mesh-hooks.patch.
Unexposed Heston MC/density exclusions are in heston-process-compatibility.patch.
"""
import argparse
import difflib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MATH = r"fabs|abs|sqrt|exp|log|pow|expm1|log1p|sin|cos|tan|sinh|cosh|tanh|asinh|acos|atan|floor|ceil|lround|isfinite|isnan|isinf"

# Function-body instrumentation only: no numerical expressions are changed.
# Nested scopes include children; self times exclude them. The profiler is
# disabled by default, so these scopes do not time every call in normal runs.
SCOPES = {
    "ql/pricingengines/basket/fd2dblackscholesvanillaengine.cpp": {
        "Fd2dBlackScholesVanillaEngine::calculate": "basket_engine"},
    "ql/methods/finitedifferences/operators/fdm2dblackscholesop.cpp": {
        "Fdm2dBlackScholesOp::setTime": "basket_operator_assembly"},
    "ql/methods/finitedifferences/operators/ninepointlinearop.cpp": {
        "NinePointLinearOp::apply": "mixed_stencil_apply"},
    "ql/methods/finitedifferences/solvers/fdm2dimsolver.cpp": {
        "Fdm2DimSolver::performCalculations": "basket_rollback_and_interpolation"},
    "ql/pricingengines/vanilla/fdblackscholesvanillaengine.cpp": {
        "FdBlackScholesVanillaEngine::calculate": "engine"},
    "ql/methods/finitedifferences/operators/fdmblackscholesop.cpp": {
        "FdmBlackScholesOp::setTime": "operator_assembly"},
    "ql/methods/finitedifferences/operators/triplebandlinearop.cpp": {
        "TripleBandLinearOp::axpyb": "band_axpyb",
        "TripleBandLinearOp::mult": "band_mult",
        "TripleBandLinearOp::apply": "band_apply",
        "TripleBandLinearOp::solve_splitting": "band_solve"},
    "ql/methods/finitedifferences/solvers/fdmbackwardsolver.cpp": {
        "FdmBackwardSolver::rollback": "rollback"},
    "ql/methods/finitedifferences/solvers/fdm1dimsolver.cpp": {
        "Fdm1DimSolver::performCalculations": "rollback_and_interpolation",
        "Fdm1DimSolver::thetaAt": "theta_interpolation"},
    "ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.cpp": {
        "FdmStepConditionComposite::applyTo": "events"},
    "ql/methods/finitedifferences/utilities/fdmdividendhandler.cpp": {
        "FdmDividendHandler::applyTo": "dividends"},
}


def instrument(name: str, content: str) -> str:
    if name not in SCOPES:
        return content
    content = content.replace("namespace QuantLib {",
                              "#include <ATen/record_function.h>\n\nnamespace QuantLib {", 1)
    for function, label in SCOPES[name].items():
        pattern = (r"(\b(?:void|Array|Real|TripleBandLinearOp)\s+" + re.escape(function)
                   + r"\s*\([^;{}]*\)\s*(?:const\s*)?\{)")
        content, count = re.subn(pattern, lambda m: m[0]
            + f'\n        RECORD_FUNCTION("qlt::{label}", std::vector<c10::IValue>());', content)
        if count != 1:
            raise RuntimeError(f"Expected one profiling anchor for {function}, found {count}")
    return content


def closure(source: Path) -> list[str]:
    manifest = (ROOT / "cmake/quantlib_sources.cmake").read_text(encoding="utf-8")
    pending = re.findall(r"ql/[\w/]+\.cpp", manifest)
    # Template curve instantiations are provided by the Python bridge.
    pending += ["ql/termstructures/yield/discountcurve.hpp"]
    # Generic latent pricing instantiates the upstream 3D solver template.
    pending += ["ql/methods/finitedifferences/solvers/fdmndimsolver.hpp"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or not (source / name).is_file():
            continue
        seen.add(name)
        content = (source / name).read_text(encoding="utf-8")
        pending += re.findall(r'#include\s+[<"](ql/[^>"]+)[>"]', content)
    return sorted(seen)


def tensor_hooks(content: str) -> str:
    """Narrow reversible dispatch; keep the upstream scalar implementations."""
    content = content.replace("namespace QuantLib {",
        "#include <quantlibtorch/tensor_kernels.hpp>\n\nnamespace QuantLib {", 1)
    anchors = {
        '''Array TripleBandLinearOp::apply(const Array& r) const {
        QL_REQUIRE(r.size() == mesher_->layout()->size(), "inconsistent length of r");''': '''
        if (quantlibtorch::tensor_kernel_mode() >= 1)
            return quantlibtorch::vector_apply(r, lower_.get(), diag_.get(), upper_.get(),
                                               i0_.get(), i2_.get());
''',
        '''TripleBandLinearOp TripleBandLinearOp::mult(const Array& u) const {

        TripleBandLinearOp retVal(direction_, mesher_);

        const Size size = mesher_->layout()->size();''': '''
        if (quantlibtorch::tensor_kernel_mode() >= 2) {
            quantlibtorch::vector_mult(size, u, lower_.get(), diag_.get(), upper_.get(),
                retVal.lower_.get(), retVal.diag_.get(), retVal.upper_.get());
            return retVal;
        }
''',
        '''void TripleBandLinearOp::axpyb(const Array& a,
                                   const TripleBandLinearOp& x,
                                   const TripleBandLinearOp& y,
                                   const Array& b) {
        const Size size = mesher_->layout()->size();''': '''
        if (quantlibtorch::tensor_kernel_mode() >= 2) {
            quantlibtorch::vector_axpyb(size, a, x.lower_.get(), x.diag_.get(), x.upper_.get(),
                y.lower_.get(), y.diag_.get(), y.upper_.get(), b,
                lower_.get(), diag_.get(), upper_.get());
            return;
        }
''',
    }
    for anchor, hook in anchors.items():
        if content.count(anchor) != 1:
            raise RuntimeError("Expected one tensor hook anchor: " + anchor)
        content = content.replace(anchor, anchor + hook)
    return content


def array_hooks(content: str) -> str:
    """Dispatch the existing elementwise transforms; retain all size checks,
    moves, operand order, and the original scalar fallback. No reductions or
    discretization expressions are changed.
    """
    content = content.replace("namespace QuantLib {",
        "#include <quantlibtorch/array_kernels.hpp>\n\nnamespace QuantLib {", 1)
    iterator = r"(?:\w+\.)?(?:begin|end)\(\)"
    binary = (r"std::transform\((" + iterator + r"),\s*(" + iterator + r"),\s*("
              + iterator + r"),\s*(" + iterator + r"),\s*std::(plus|minus|multiplies|divides)<>\(\)\);")

    def binary_hook(m):
        first, last, second, output, name = m.groups()
        op = {"plus": "+", "minus": "-", "multiplies": "*", "divides": "/"}[name]
        return (f"if (!quantlibtorch::array_binary({first}, {last}, {second}, {output}, '{op}'))\n"
                + "            " + m[0])

    content, binary_count = re.subn(binary, binary_hook, content)
    scalar = (r"std::transform\((" + iterator + r"),\s*(" + iterator + r"),\s*("
              + iterator + r"),\s*\[=\]\(Real y\) -> Real \{ return (\w+) ([+*/-]) (\w+); \}\);")

    def scalar_hook(m):
        first, last, output, lhs, op, rhs = m.groups()
        if (lhs == "y") == (rhs == "y"):
            raise RuntimeError("Unexpected scalar transform operands: " + m[0])
        value, reverse = (rhs, "false") if lhs == "y" else (lhs, "true")
        return (f"if (!quantlibtorch::array_scalar({first}, {last}, {output}, {value}, '{op}', {reverse}))\n"
                + "            " + m[0])

    content, scalar_count = re.subn(scalar, scalar_hook, content)
    if (binary_count, scalar_count) != (20, 20):
        raise RuntimeError(f"Unexpected array transform counts: {binary_count}, {scalar_count}")
    return content


def spline_hooks(content: str) -> str:
    content = content.replace("namespace QuantLib {",
        "#include <quantlibtorch/spline_kernels.hpp>\n\nnamespace QuantLib {", 1)
    changes = {
        "                for (Size i=0; i<n_-1; ++i) {\n                    dx_[i]":
        "                if (!quantlibtorch::spline_slopes(this->xBegin_, this->xEnd_, this->yBegin_, dx_.data(), S_.data()))\n"
        "                for (Size i=0; i<n_-1; ++i) {\n                    dx_[i]",
        "                if (da_==CubicInterpolation::Spline) {\n":
        "                if (da_==CubicInterpolation::Spline) {\n"
        "                    const bool tensorRhs = quantlibtorch::spline_rhs(dx_, S_, tmp_);\n",
        "                        tmp_[i] = 3.0*(dx_[i]*S_[i-1] + dx_[i-1]*S_[i]);":
        "                        if (!tensorRhs) tmp_[i] = 3.0*(dx_[i]*S_[i-1] + dx_[i-1]*S_[i]);",
        "                if (monotonic_) {":
        "                if (monotonic_ && !quantlibtorch::spline_filter(dx_, S_, tmp_, monotonicityAdjustments_)) {",
        "                // cubic coefficients\n                for":
        "                // cubic coefficients\n"
        "                if (!quantlibtorch::spline_coefficients(dx_, S_, tmp_, a_.data(), b_.data(), c_.data()))\n                for",
        "                primitiveConst_[0] = 0.0;":
        "                const Array tensorIncrements = quantlibtorch::spline_increments(this->yBegin_, n_, dx_, a_, b_, c_);\n"
        "                primitiveConst_[0] = 0.0;",
        "                    primitiveConst_[i] = primitiveConst_[i-1]\n":
        "                    if (!tensorIncrements.empty()) {\n"
        "                        primitiveConst_[i] = primitiveConst_[i-1] + tensorIncrements[i-1];\n"
        "                        continue;\n                    }\n"
        "                    primitiveConst_[i] = primitiveConst_[i-1]\n",
    }
    for anchor, replacement in changes.items():
        if content.count(anchor) != 1:
            raise RuntimeError("Expected one spline anchor: " + anchor)
        content = content.replace(anchor, replacement)
    return content


def diffusion_adjoint_hooks(name: str, content: str) -> str:
    if name == "ql/methods/finitedifferences/operators/triplebandlinearop.hpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/adjoint.hpp>\n\nnamespace QuantLib {", 1)
        content, count = re.subn(r"(class TripleBandLinearOp\b[^;{]*\{)", lambda match: match[0] + '''
        friend bool quantlibtorch::try_diffusion_assembly(TripleBandLinearOp&,
            const TripleBandLinearOp&, const TripleBandLinearOp&, const Array&, Real, Real);
''', content)
    elif name == "ql/methods/finitedifferences/operators/fdmblackscholesop.cpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/adjoint.hpp>\n\nnamespace QuantLib {", 1)
        anchor = '''                mapT_.axpyb(r - q - 0.5*v, dxMap_,
                            dxxMap_.mult(0.5*v), Array(1, -r));'''
        count = content.count(anchor)
        content = content.replace(anchor,
            "                if (!quantlibtorch::try_diffusion_assembly(mapT_, dxMap_, dxxMap_, v, r, q))\n"+anchor)
    else:
        return content
    if count != 1:
        raise RuntimeError(f"Expected one diffusion adjoint anchor in {name}, found {count}")
    return content


def basis_cache_hooks(name: str, content: str) -> str:
    # Keep storage/reuse changes separate from compatibility and numerical hooks.
    if name == "ql/methods/finitedifferences/operators/triplebandlinearop.hpp":
        anchor = "const TripleBandLinearOp&, const TripleBandLinearOp&, const Array&, Real, Real);"
        replacement = anchor[:-2] + ", quantlibtorch::DiffusionBasisCache*);"
    elif name == "ql/methods/finitedifferences/operators/fdmblackscholesop.hpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/adjoint.hpp>\n\nnamespace QuantLib {", 1)
        anchor = "        const TripleBandLinearOp dxxMap_;"
        replacement = anchor + "\n        quantlibtorch::DiffusionBasisCache tensorBasis_;"
    elif name == "ql/methods/finitedifferences/operators/fdmblackscholesop.cpp":
        anchor = "try_diffusion_assembly(mapT_, dxMap_, dxxMap_, v, r, q)"
        replacement = "try_diffusion_assembly(mapT_, dxMap_, dxxMap_, v, r, q, &tensorBasis_)"
    else:
        return content
    if content.count(anchor) != 1:
        raise RuntimeError("Expected one immutable basis cache anchor in " + name)
    return content.replace(anchor, replacement)


def assembled_band_hooks(name: str, content: str) -> str:
    # Retain a graph-connected result only in its private operator owner.
    # Keep this storage/lifetime patch separate from numerical substitutions.
    if name == "ql/methods/finitedifferences/operators/triplebandlinearop.hpp":
        changes = {"Real, Real, quantlibtorch::DiffusionBasisCache*);":
                   "Real, Real, quantlibtorch::DiffusionBasisCache*, at::Tensor*);"}
    elif name == "ql/methods/finitedifferences/operators/fdmblackscholesop.hpp":
        changes = {"        quantlibtorch::DiffusionBasisCache tensorBasis_;":
                   "        quantlibtorch::DiffusionBasisCache tensorBasis_;\n        at::Tensor tensorBands_;"}
    elif name == "ql/methods/finitedifferences/operators/fdmblackscholesop.cpp":
        changes = {
            "    void FdmBlackScholesOp::setTime(Time t1, Time t2) {":
            "    void FdmBlackScholesOp::setTime(Time t1, Time t2) {\n"
            "        tensorBands_ = at::Tensor(); // invalidate before rates, sampling, or any exception\n",
            "try_diffusion_assembly(mapT_, dxMap_, dxxMap_, v, r, q, &tensorBasis_)":
            "try_diffusion_assembly(mapT_, dxMap_, dxxMap_, v, r, q, &tensorBasis_, &tensorBands_)",
        }
    else:
        return content
    for anchor, replacement in changes.items():
        if content.count(anchor) != 1:
            raise RuntimeError("Expected one assembled band anchor in " + name)
        content = content.replace(anchor, replacement)
    return content


def step_adjoint_hooks(name: str, content: str) -> str:
    headers = {
        "ql/methods/finitedifferences/operators/triplebandlinearop.hpp": "TripleBandLinearOp",
        "ql/methods/finitedifferences/operators/fdmblackscholesop.hpp": "FdmBlackScholesOp",
        "ql/methods/finitedifferences/schemes/boundaryconditionschemehelper.hpp": "BoundaryConditionSchemeHelper",
    }
    schemes = {
        "ql/methods/finitedifferences/schemes/douglasscheme.cpp": ("theta_", 0),
        "ql/methods/finitedifferences/schemes/impliciteulerscheme.cpp": ("theta", 1),
        "ql/methods/finitedifferences/schemes/expliciteulerscheme.cpp": ("theta", 2),
    }
    if name not in headers and name not in schemes:
        return content
    content = content.replace("namespace QuantLib {",
        "#include <quantlibtorch/step_adjoint.hpp>\n\nnamespace QuantLib {", 1)
    if name in headers:
        declaration = '''
        friend bool quantlibtorch::try_step_adjoint(FdmLinearOpComposite&,
            const BoundaryConditionSchemeHelper&, Array&, Time, Real, Time, int);
'''
        content, count = re.subn(r"(class " + headers[name] + r"\b[^;{]*\{)",
                                  lambda match: match[0]+declaration, content)
    else:
        theta, kind = schemes[name]
        content, count = re.subn(r"bcSet_\.setTime\((?:std::)?max\(0\.0, t-dt_\)\);",
            lambda match: match[0] + f'''
        if (quantlibtorch::try_step_adjoint(*map_, bcSet_, a, dt_, {theta}, t, {kind}))
            return;
''', content)
    if count != 1:
        raise RuntimeError(f"Expected one step adjoint anchor in {name}, found {count}")
    return content


def structured_hooks(name: str, content: str) -> str:
    """Extend numerical equation boundaries; retain original 1D dispatch."""
    if name == "ql/methods/finitedifferences/operators/triplebandlinearop.cpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/structured_ops.hpp>\n\nnamespace QuantLib {", 1)
        anchor = '''            QL_REQUIRE(mesher_->layout()->dim().size() == 1,
                       "adjoint currently requires one dimension");'''
        if content.count(anchor) != 1:
            raise RuntimeError("Expected one dimensional splitting-adjoint guard")
        content = content.replace(anchor, '''            if (mesher_->layout()->dim().size() != 1)
                return quantlibtorch::directional_adjoint_split(*this, r, a, b,
                    lower_.get(), diag_.get(), upper_.get(), reverseIndex_.get());''')
    elif name == "ql/methods/finitedifferences/operators/ninepointlinearop.cpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/structured_ops.hpp>\n"
            "#include <quantlibtorch/tensor_kernels.hpp>\n\nnamespace QuantLib {", 1)
        anchor = '''        QL_REQUIRE(u.size() == mesher_->layout()->size(),"inconsistent length of r "
                    << u.size() << " vs " << mesher_->layout()->size());'''
        if content.count(anchor) != 1:
            raise RuntimeError("Expected one nine-point application anchor")
        content = content.replace(anchor, anchor + '''
        if (quantlibtorch::tensor_kernel_mode() >= 1)
            return quantlibtorch::nine_point_apply(u,
                {a00_.get(), a01_.get(), a02_.get(), a10_.get(), a11_.get(),
                 a12_.get(), a20_.get(), a21_.get(), a22_.get()},
                {i00_.get(), i01_.get(), i02_.get(), i10_.get(), nullptr,
                 i12_.get(), i20_.get(), i21_.get(), i22_.get()});
''')
    return content


def heston_process_compatibility(name: str, content: str) -> str:
    if name == "ql/processes/hestonprocess.cpp":
        # PDE pricing only uses the original constructor, parameter accessors,
        # time, drift/diffusion and apply. Unexposed MC/density implementations
        # instantiate std::complex<Real> and Boost distribution internals that
        # do not support TorchReal. Fail explicitly instead of casting inputs.
        for anchor in ("Array HestonProcess::apply", "    const Handle<Quote>& HestonProcess::s0()",
                       "    Real HestonProcess::varianceDistribution", "    Array HestonProcess::evolve",
                       "    Real HestonProcess::pdf"):
            if content.count(anchor) != 1:
                raise RuntimeError("Expected one Heston process anchor: " + anchor)
        a = content.index("    namespace {", content.index("Array HestonProcess::apply"))
        b = content.index("    const Handle<Quote>& HestonProcess::s0()")
        content = content[:a] + '''    Array HestonProcess::evolve(Time, const Array&, Time, const Array&) const {
        QL_FAIL("Heston Monte Carlo evolution is not exposed by QuantLibTorch");
    }
    Real HestonProcess::pdf(Real, Real, Time, Real) const {
        QL_FAIL("Heston transition density is not exposed by QuantLibTorch");
    }
''' + content[b:]
        a = content.index("    Real HestonProcess::varianceDistribution")
        content = content[:a] + '''    Real HestonProcess::varianceDistribution(Real, Real, Time) const {
        QL_FAIL("Heston variance sampling is not exposed by QuantLibTorch");
    }
}
'''
        content = content.replace("#include <boost/math/distributions/non_central_chi_squared.hpp>", "")
    return content


def engine_mesh_hooks(name: str, content: str) -> str:
    if name == "ql/pricingengines/vanilla/fdhestonvanillaengine.cpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/fixed_mesh.hpp>\n#include <quantlibtorch/pricing_output.hpp>\n\nnamespace QuantLib {", 1)
        start = "        // 1.1 The variance mesher"
        end = "            new FdmMesherComposite(equityMesher, vMesher));"
        if content.count(start) != 1 or content.count(end) != 1:
            raise RuntimeError("Expected one original Heston mesher construction")
        a, b = content.index(start), content.index(end) + len(end)
        original_mesh = content[a:b].replace(
            "        const ext::shared_ptr<FdmMesher> mesher(\n            new FdmMesherComposite(equityMesher, vMesher));",
            "        mesher = ext::make_shared<FdmMesherComposite>(equityMesher, vMesher);")
        content = (content[:a] + "        auto mesher = quantlibtorch::pricing_mesh_override();\n"
                   "        if (!mesher) {\n" + original_mesh + "\n        }" + content[b:])
        start = "        results_.delta = solver->deltaAt(spot, v0);"
        end = "        results_.theta = solver->thetaAt(spot, v0);"
        if content.count(start) != 1 or content.count(end) != 1:
            raise RuntimeError("Expected one original Heston Greek block")
        a, b = content.index(start), content.index(end) + len(end)
        content = content[:a] + "        if (!quantlibtorch::price_only()) {\n" + content[a:b] + "\n        }" + content[b:]
    if name == "ql/pricingengines/vanilla/fdblackscholesshoutengine.cpp":
        content = content.replace("namespace QuantLib {",
            "#include <quantlibtorch/fixed_mesh.hpp>\n#include <quantlibtorch/pricing_output.hpp>\n\nnamespace QuantLib {", 1)
        start = "        const auto mesher = ext::make_shared<FdmMesherComposite>("
        end = "                divAdj));"
        if content.count(start) != 1 or content.count(end) != 1:
            raise RuntimeError("Expected one original shout mesher construction")
        a, b = content.index(start), content.index(end) + len(end)
        original_mesh = content[a:b].replace("const auto mesher =", "mesher =")
        content = content[:a] + "        auto mesher = quantlibtorch::pricing_mesh_override();\n        if (!mesher) {\n" + original_mesh + "\n        }" + content[b:]
        start = "        results_.delta = solver->deltaAt(spot);"
        end = "        results_.theta = solver->thetaAt(spot);"
        if content.count(start) != 1 or content.count(end) != 1:
            raise RuntimeError("Expected one original shout Greek block")
        a, b = content.index(start), content.index(end) + len(end)
        content = content[:a] + "        if (!quantlibtorch::price_only()) {\n" + content[a:b] + "\n        }" + content[b:]
    return content


def qualify_friend_parameters(name: str, content: str) -> str:
    # GCC resolves qualified friend declarators in the friend's namespace.
    # Spell out QuantLib parameter types; this changes no function signature
    # or numerical body. Keep the compatibility diff separate from AD hooks.
    expected = {
        "ql/methods/finitedifferences/operators/triplebandlinearop.hpp": 2,
        "ql/methods/finitedifferences/operators/fdmblackscholesop.hpp": 1,
        "ql/methods/finitedifferences/schemes/boundaryconditionschemehelper.hpp": 1,
    }
    if name not in expected:
        return content
    pattern = r"(friend bool quantlibtorch::(?:try_step_adjoint|try_diffusion_assembly)\()([^;]+)(\);)"
    types = r"(?<![:\w])(?:TripleBandLinearOp|FdmLinearOpComposite|BoundaryConditionSchemeHelper|Array|Time|Real)\b"
    content, count = re.subn(pattern, lambda match: match[1] + re.sub(
        types, lambda item: "QuantLib::" + item[0], match[2]) + match[3], content)
    if count != expected[name]:
        raise RuntimeError(f"Expected {expected[name]} friend declarations in {name}, found {count}")
    return content


def prepare(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("overlay output must be separate from the QuantLib source checkout")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "ql").resolve().is_relative_to(output):
        raise ValueError("overlay ql directory must remain inside the output directory")
    generated_manifest = output / "generated-files.json"
    if generated_manifest.exists():
        previous = json.loads(generated_manifest.read_text(encoding="utf-8"))
    else:
        # Migrate overlays made before the manifest existed. This directory is
        # generated build output; only QuantLib files inside its ql subtree are
        # candidates, and every path is checked before removal below.
        previous = [p.relative_to(output).as_posix()
                    for p in (output / "ql").rglob("*") if p.is_file()]
    changes = []
    hooks = []
    kernels = []
    sampling = []
    arrays = []
    outputs = []
    splines = []
    tridiagonal = []
    parallel = []
    operator_adjoints = []
    operator_arithmetic = []
    step_adjoints = []
    diffusion_adjoints = []
    basis_caches = []
    assembled_bands = []
    structured_operators = []
    profiling = []
    heston_compatibility = []
    engine_meshes = []
    friend_compatibility = []
    generated = []
    for name in closure(source):
        original = (source / name).read_text(encoding="utf-8")
        names = sorted(set(re.findall(r"std::(" + MATH + r"|min|max)\b", original)))
        imports = "\n".join(f"    using std::{fn};" for fn in names)
        updated = original
        if names:
            updated = re.sub(r"namespace QuantLib(?P<nested>::\w+)? \{",
                             lambda m: m.group(0) + "\n" + imports, updated, count=1)
            updated = re.sub(r"std::(" + MATH + r"|min|max)(?=[(<])", r"\1", updated)
        if name == "ql/utilities/null.hpp":
            updated = updated.replace("std::is_floating_point_v<T>",
                                      "std::is_floating_point_v<T> || std::is_same_v<T, Real>")
        changes.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                          fromfile="a/" + name, tofile="b/" + name))
        if name == "ql/math/interpolations/cubicinterpolation.hpp":
            before_spline = updated
            updated = spline_hooks(updated)
            splines.extend(difflib.unified_diff(before_spline.splitlines(True), updated.splitlines(True),
                                              fromfile="a/" + name, tofile="b/" + name))
        if name == "ql/methods/finitedifferences/tridiagonaloperator.cpp":
            before_tridiagonal = updated
            updated = updated.replace("namespace QuantLib {",
                "#include <quantlibtorch/adjoint.hpp>\n#include <quantlibtorch/spline_kernels.hpp>\n\nnamespace QuantLib {", 1)
            anchor = "        Real bet = diagonal_[0];"
            if updated.count(anchor) != 1:
                raise RuntimeError("Expected one tridiagonal solve anchor")
            updated = updated.replace(anchor, '''        if (quantlibtorch::adjoint_enabled() && quantlibtorch::tridiagonal_adjoint_enabled()) {
            result = quantlibtorch::adjoint_tridiagonal(lowerDiagonal_, diagonal_, upperDiagonal_, rhs);
            return;
        }
''' + anchor)
            tridiagonal.extend(difflib.unified_diff(before_tridiagonal.splitlines(True), updated.splitlines(True),
                                                  fromfile="a/" + name, tofile="b/" + name))
            before_parallel = updated
            updated = updated.replace("namespace QuantLib {",
                "#include <quantlibtorch/parallel_solve.hpp>\n\nnamespace QuantLib {", 1)
            updated = updated.replace(anchor, '''        if (quantlibtorch::tensor_kernel_mode() == 4 && !quantlibtorch::adjoint_enabled()
            && at::GradMode::is_enabled() && rhs.size() >= 8
            && quantlibtorch::try_pcr_tridiagonal(lowerDiagonal_, diagonal_, upperDiagonal_, rhs, result))
            return;
''' + anchor)
            parallel.extend(difflib.unified_diff(before_parallel.splitlines(True), updated.splitlines(True),
                                               fromfile="a/" + name, tofile="b/" + name))
        if name == "ql/math/array.hpp":
            before_arrays = updated
            updated = array_hooks(updated)
            arrays.extend(difflib.unified_diff(before_arrays.splitlines(True), updated.splitlines(True),
                                             fromfile="a/" + name, tofile="b/" + name))
        if name == "ql/pricingengines/vanilla/fdblackscholesvanillaengine.cpp":
            before_output = updated
            updated = updated.replace("namespace QuantLib {",
                "#include <quantlibtorch/pricing_output.hpp>\n\nnamespace QuantLib {", 1)
            anchor = '''        results_.delta = solver->deltaAt(spot);
        results_.gamma = solver->gammaAt(spot);
        results_.theta = solver->thetaAt(spot);'''
            if updated.count(anchor) != 1:
                raise RuntimeError("Expected one engine Greeks anchor")
            indented = "\n".join("    " + line for line in anchor.splitlines())
            updated = updated.replace(anchor,
                "        if (!quantlibtorch::price_only()) {\n" + indented + "\n        }")
            outputs.extend(difflib.unified_diff(before_output.splitlines(True), updated.splitlines(True),
                                              fromfile="a/" + name, tofile="b/" + name))
        if name == "ql/methods/finitedifferences/operators/triplebandlinearop.cpp":
            before_hook = updated
            updated = updated.replace("namespace QuantLib {", "#include <quantlibtorch/adjoint.hpp>\n\nnamespace QuantLib {", 1)
            anchor = '''Array TripleBandLinearOp::solve_splitting(const Array& r, Real a, Real b) const {
        QL_REQUIRE(r.size() == mesher_->layout()->size(), "inconsistent size of rhs");'''
            assert updated.count(anchor) == 1
            updated = updated.replace(anchor, anchor + '''
        if (quantlibtorch::adjoint_enabled()) {
            QL_REQUIRE(mesher_->layout()->dim().size() == 1,
                       "adjoint currently requires one dimension");
            return quantlibtorch::adjoint_split(*this, r, a, b,
                                                lower_.get(), diag_.get(), upper_.get());
        }
''')
            hooks.extend(difflib.unified_diff(before_hook.splitlines(True), updated.splitlines(True),
                                            fromfile="a/" + name, tofile="b/" + name))
            before_kernels = updated
            updated = tensor_hooks(updated)
            kernels.extend(difflib.unified_diff(before_kernels.splitlines(True), updated.splitlines(True),
                                              fromfile="a/" + name, tofile="b/" + name))
            before_parallel = updated
            updated = updated.replace("namespace QuantLib {",
                "#include <quantlibtorch/parallel_solve.hpp>\n\nnamespace QuantLib {", 1)
            updated = updated.replace(anchor, anchor + '''
        if (quantlibtorch::tensor_kernel_mode() == 3) {
            QL_REQUIRE(!quantlibtorch::adjoint_enabled(), "PCR pricing supports full AD only");
            QL_REQUIRE(mesher_->layout()->dim().size() == 1, "PCR hook requires one dimension");
            return quantlibtorch::pcr_split(r, a, b, lower_.get(), diag_.get(), upper_.get());
        }
        if (quantlibtorch::tensor_kernel_mode() == 4 && !quantlibtorch::adjoint_enabled()
            && at::GradMode::is_enabled() && mesher_->layout()->dim().size() == 1 && r.size() >= 8) {
            Array candidate;
            if (quantlibtorch::try_pcr_split(r, a, b, lower_.get(), diag_.get(), upper_.get(), candidate))
                return candidate;
        }
''')
            parallel.extend(difflib.unified_diff(before_parallel.splitlines(True), updated.splitlines(True),
                                               fromfile="a/" + name, tofile="b/" + name))
            before_operators = updated
            apply_anchor = '''Array TripleBandLinearOp::apply(const Array& r) const {
        QL_REQUIRE(r.size() == mesher_->layout()->size(), "inconsistent length of r");'''
            if updated.count(apply_anchor) != 1:
                raise RuntimeError("Expected one operator adjoint anchor")
            updated = updated.replace(apply_anchor, apply_anchor + '''
        if (quantlibtorch::tensor_kernel_mode() >= 1 && quantlibtorch::adjoint_enabled()
            && quantlibtorch::operator_adjoint_enabled())
            return quantlibtorch::adjoint_apply(*this, r, lower_.get(), diag_.get(), upper_.get(),
                                                i0_.get(), i2_.get());
''')
            operator_adjoints.extend(difflib.unified_diff(before_operators.splitlines(True), updated.splitlines(True),
                                                        fromfile="a/" + name, tofile="b/" + name))
            before_arithmetic = updated
            updated = updated.replace(apply_anchor, apply_anchor + '''
        if (quantlibtorch::tensor_kernel_mode() >= 1 && !quantlibtorch::adjoint_enabled()
            && quantlibtorch::operator_arithmetic_enabled())
            return quantlibtorch::adjoint_apply(*this, r, lower_.get(), diag_.get(), upper_.get(),
                                                i0_.get(), i2_.get());
''')
            operator_arithmetic.extend(difflib.unified_diff(before_arithmetic.splitlines(True), updated.splitlines(True),
                                                           fromfile="a/" + name, tofile="b/" + name))
        if name == "ql/methods/finitedifferences/operators/fdmblackscholesop.cpp":
            before_sampling = updated
            updated = updated.replace("namespace QuantLib {",
                "#include <quantlibtorch/local_vol.hpp>\n\nnamespace QuantLib {", 1)
            anchor = "            Array v(mesher_->layout()->size());\n"
            if updated.count(anchor) != 1:
                raise RuntimeError("Expected one local volatility sampling anchor")
            updated = updated.replace(anchor, anchor + '''            if (!(illegalLocalVolOverwrite_ < 0.0
                  && quantlibtorch::sample_local_variances(localVol_.get(), 0.5*(t1+t2), x_, v)))
''')
            sampling.extend(difflib.unified_diff(before_sampling.splitlines(True), updated.splitlines(True),
                                               fromfile="a/" + name, tofile="b/" + name))
        before_diffusion = updated
        updated = diffusion_adjoint_hooks(name, updated)
        diffusion_adjoints.extend(difflib.unified_diff(before_diffusion.splitlines(True), updated.splitlines(True),
                                                     fromfile="a/" + name, tofile="b/" + name))
        before_basis = updated
        updated = basis_cache_hooks(name, updated)
        basis_caches.extend(difflib.unified_diff(before_basis.splitlines(True), updated.splitlines(True),
                                               fromfile="a/" + name, tofile="b/" + name))
        before_bands = updated
        updated = assembled_band_hooks(name, updated)
        assembled_bands.extend(difflib.unified_diff(before_bands.splitlines(True), updated.splitlines(True),
                                                  fromfile="a/" + name, tofile="b/" + name))
        before_step = updated
        updated = step_adjoint_hooks(name, updated)
        step_adjoints.extend(difflib.unified_diff(before_step.splitlines(True), updated.splitlines(True),
                                                fromfile="a/" + name, tofile="b/" + name))
        before_structured = updated
        updated = structured_hooks(name, updated)
        structured_operators.extend(difflib.unified_diff(
            before_structured.splitlines(True), updated.splitlines(True),
            fromfile="a/" + name, tofile="b/" + name))
        before_profile = updated
        updated = instrument(name, updated)
        profiling.extend(difflib.unified_diff(before_profile.splitlines(True), updated.splitlines(True),
                                             fromfile="a/" + name, tofile="b/" + name))
        before_heston = updated
        updated = heston_process_compatibility(name, updated)
        heston_compatibility.extend(difflib.unified_diff(before_heston.splitlines(True), updated.splitlines(True),
                                          fromfile="a/" + name, tofile="b/" + name))
        before_mesh = updated
        updated = engine_mesh_hooks(name, updated)
        engine_meshes.extend(difflib.unified_diff(before_mesh.splitlines(True), updated.splitlines(True),
                                          fromfile="a/" + name, tofile="b/" + name))
        before_friends = updated
        updated = qualify_friend_parameters(name, updated)
        friend_compatibility.extend(difflib.unified_diff(before_friends.splitlines(True), updated.splitlines(True),
                                          fromfile="a/" + name, tofile="b/" + name))
        if updated == original:
            continue
        target = (output / name).resolve()
        if not target.is_relative_to(output / "ql"):
            raise ValueError(f"Unsafe overlay target: {target}")
        generated.append(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text(encoding="utf-8") != updated:
            target.write_text(updated, encoding="utf-8")
    (output / "math-adl.patch").write_text("".join(changes), encoding="utf-8")
    (output / "adjoint-hook.patch").write_text("".join(hooks), encoding="utf-8")
    (output / "tensor-kernels.patch").write_text("".join(kernels), encoding="utf-8")
    (output / "sampling-kernel.patch").write_text("".join(sampling), encoding="utf-8")
    (output / "array-kernels.patch").write_text("".join(arrays), encoding="utf-8")
    (output / "pricing-output.patch").write_text("".join(outputs), encoding="utf-8")
    (output / "spline-kernels.patch").write_text("".join(splines), encoding="utf-8")
    (output / "tridiagonal-adjoint.patch").write_text("".join(tridiagonal), encoding="utf-8")
    (output / "parallel-solve.patch").write_text("".join(parallel), encoding="utf-8")
    (output / "operator-adjoint.patch").write_text("".join(operator_adjoints), encoding="utf-8")
    (output / "operator-arithmetic.patch").write_text("".join(operator_arithmetic), encoding="utf-8")
    (output / "step-adjoint.patch").write_text("".join(step_adjoints), encoding="utf-8")
    (output / "diffusion-adjoint.patch").write_text("".join(diffusion_adjoints), encoding="utf-8")
    (output / "basis-cache.patch").write_text("".join(basis_caches), encoding="utf-8")
    (output / "assembled-bands.patch").write_text("".join(assembled_bands), encoding="utf-8")
    (output / "structured-operators.patch").write_text("".join(structured_operators), encoding="utf-8")
    (output / "profiling-scopes.patch").write_text("".join(profiling), encoding="utf-8")
    (output / "heston-process-compatibility.patch").write_text("".join(heston_compatibility), encoding="utf-8")
    (output / "engine-mesh-hooks.patch").write_text("".join(engine_meshes), encoding="utf-8")
    (output / "friend-namespace-compatibility.patch").write_text("".join(friend_compatibility), encoding="utf-8")
    for name in set(previous) - set(generated):
        target = (output / name).resolve()
        if not target.is_relative_to(output / "ql"):
            raise RuntimeError(f"Refusing to remove unsafe overlay path: {target}")
        if target.is_file():
            target.unlink()
    generated_manifest.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
