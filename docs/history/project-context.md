> Historical snapshot through Phase 26, archived during release preparation.
> Statements of current support and next work describe their original checkpoints.
> Use [current architecture](../architecture.md), [capabilities](../capabilities.md)
> and [release support](../release.md) for current contracts. Command paths are relative
> to the repository root. Markdown links have been relocated; evidence is unchanged.

# QuantLibTorch — Research Context

## Current integration milestone (Phase 26, 2026-11-02)

The extended autonomous goal builds on `092abf1` and keeps the practical library
priority. New original `FdHestonVanillaEngine` and `FdBlackScholesShoutEngine`
adapters expose explicit original/custom fixed meshes, model/market tensors,
early exercise/shout opportunities and original dividend handling. Heston
supports five model parameters, spot and curves; full AD supports damping and
higher derivatives, while first-order Heston adjoints require zero damping.
Shout supports constant Black volatility, spot/curves, six schemes and damping.
These scalar engines are CPU float64; neither is a replacement for the existing
CPU/CUDA neural latent or batch pricing APIs.

`fd_latent`, `LatentBatch` and `MultiAssetBatch` additionally support fresh
graph-connected rates/yields and dated discount factors. Checkpoint replay
propagates their VJPs; passive geometry/event caches never retain active graphs.
Full higher derivatives and CPU/CUDA market-gradient parity are tested.
Heston/shout support Actual/365 Fixed or Actual/360; original calendar helpers
prepare dates explicitly with NullCalendar, WeekendsOnly or TARGET.

The explanatory Heston, shout and financial-input notebooks are included in
source distributions. Local package version is 0.3.0. Original Heston MC/density
methods that need unsupported complex scalars explicitly fail in a separate
compatibility overlay; their inclusion is not claimed. Current validation,
checkpoint and precise limitations belong in the Phase 26 section at the top
of `docs/architecture.md` and in `docs/capabilities.md`.
The final native build passes **2,224 tests**, with one intentional skip
(135 added tests). All three notebooks execute, including CPU/CUDA financial
input parity. Native/source fingerprints and independent-reference coverage
are recorded in `docs/benchmarks/library_phase26_validation.json`.
Implementation checkpoint: **`c80448a`**. Distribution evidence is recorded in
`docs/benchmarks/packaging_phase26.json`.


## Preserved Phase 25 baseline (2026-11-02)

The user explicitly redirected priority from numerical-analysis research to
broader practical QuantLib functionality. Preserve Approach A, original APIs,
full AD, equation adjoints, CPU/CUDA infrastructure and all prior evidence.
The Phase 25 roadmap is in `docs/roadmap.md`; exact support and limitations
are in `docs/capabilities.md`, installation in `docs/install.md`.

Phase 25 adds reusable financial configuration and `PricingLayer`, original
analytic European engines with differentiable market inputs, six 1D FD
schemes and escrowed dividends, plus American/Bermudan exercise and fixed spot
cash dividends in compatible CPU/CUDA batches. One/two latent factors support
exercise; packed cash dividends currently require one latent factor. Local
ABI-matched wheels include native binaries and reject stale build artifacts.
Validation and checkpoints are recorded at the top of `docs/architecture.md`.
The final native build passes **2,089 tests** with one intentional skip; all
149 added API, financial-input, event and packaging tests pass. Installed
Windows wheel pricing and gradients are independently smoke-tested.
Implementation checkpoint: **`6e3e6a3`**.

Next practical priorities: fixed-mesh rate/curve sensitivities, original model
and FD engine adapters, broader mesh/boundary/convention controls, packed
damping and portable Windows/Linux release verification. Do not make further
calibration-identifiability or extended convergence studies the primary work.

## Preserved Phase 24 priority (2026-11-01)

Phase 24 prioritizes multi-asset × multi-contract CUDA pricing while preserving
the Phase 23 baseline `27ba330` and the original single-contract engines.
Implementation checkpoint `43d6113` adds `MultiAssetBatch`, separate per-asset
operators/curves/initial states, compatibility grouping, device-resident
QuantLib natural spline readout, and checkpointed first-order equation VJPs.
The full native build passes **1,940 tests** (75 new), with one intentional skip.
Full AD and two latent factors are covered. Across 36 workload reports, matched
prices/parameter gradients differ by at most 4.62e-14 / 1.15e-14. On 20 x 30
contracts, 41 x 17 nodes and 24 steps, complete objective/VJP time is
56.632 s optimized serial CPU, 14.310 s contract-batched CPU, 4.065 s two-level
CPU, and **.7593 s CUDA** (5.35x versus batched CPU, one CPU thread). CUDA peak
allocated/reserved memory is **244.28 / 272 MiB**, returning to 32.42 MiB live
allocation after each cleanup. The profiled pricing/VJP region transfers only
two one-byte validation flags to the host; setup/logging transfers are separate.
Small batches still favor CPU. At 32 x 32 / 21 x 9, CUDA reaches 2,358 contracts/s.

Actual-workload neural/embedding/initial-state finite differences agree within
3.56e-10. Multi-asset Heston and frozen-neural CPU/CUDA refinements agree within
1.60e-13 in price and 4.10e-12 in recorded sensitivities. The neural 41 x 17 grid
still moves by up to .030043 dollars against 161 x 65; speed does not certify
accuracy. Phase 24 supports European compatible groups, fixed financial
configuration, float64, and a restricted checked CUDA solve domain. Full AD is
not checkpointed. Next measured priorities are kernel fusion, device packing,
checkpoint/contract chunk selection, and explicit event/boundary contracts.
See the Phase 24 record in `docs/architecture.md` and its preservation manifest.

### Preserved Phase 23 foundation

Phase 23 implements the authorized tensorized CPU coefficient assembly and
compatible-contract batching milestone, followed by a measured CUDA prototype.
Approach A and the pinned QuantLib infrastructure remain: spatial bases come
from the original operators, the four ADI step bodies are generated unchanged
apart from their array type, and the original rollback scheduler, payoff
averaging, curves and spline readout are retained. No stock-double QuantLib is
linked into the TorchReal library. The validated CPU checkpoint is `93cd935`.

`LatentBatch` and `diffusion_rollback_batch` expose packed execution, and
`LatentCalibration(execution="batched")` groups European quotes by expiry.
Only fixed geometry/payoffs/configuration are cached; every model graph is
fresh. Incompatible meshes/schemes/curves/expiry configurations are rejected;
cash dividends, exercise schedules, damping and custom boundaries are rejected
by batching. Original single-contract capabilities and serial calibration remain.
Both full AD/higher derivatives and first-order equation adjoints are validated.
The complete native build passes **1,865 tests**, including **109 new cases**.

Across 47 isolated workers, maximum matched price/gradient differences are
2.49e-14 / 1.25e-11. On 81 x 17 with nine common-expiry contracts, the complete
adjoint objective/VJP falls from 57.673 to .4566 s; peak process growth falls
from 684.6 to 190.7 MiB. The 27-contract, three-expiry 41 x 17 workload falls
from 46.203 to .7385 s. These include logging and cleanup; setup is separate.
Both modes reproduce the Phase 22 fit in 15 iterations/16 evaluations: .007409
RMSE, initial variance .043203, 8.689 s adjoint / 7.959 s full AD. Gradient norm
1.813 still fails the optimizer tolerance; there is no new identifiability claim.

CUDA keeps coefficient/state tensors resident but uses connected CPU financial
readout and checked tensor PCR. It rejects systems outside its strict forward
row diagonal-dominance domain. The RTX 4060 Laptop GPU is slower on smaller
tested workloads, but 161 x 65 / 96 steps / nine contracts gives 6.090 s versus
6.913 s optimized CPU. GPU peak allocated/reserved memory is 3.42/4.53 GiB;
live allocation returns to 21.41 MiB after each cleanup. Packed CPU peak growth
at that size is 3,875.6 MiB. No broad GPU advantage or constant-memory claim.

Frozen neural prices still move by up to .016354 dollars from the training
mesh to 161 x 65, exceeding the fit residual. New per-price network-direction
and raw-initial-state sensitivity refinements, analytic-Heston parameter bumps,
and Black-Scholes vega diagnostics show why latent-axis and combined refinement
matter. Numerical accuracy remains first-class; agreement of two discrete
implementations is not an error bound against the continuous model.

See the Phase 23 record at the top of `docs/architecture.md`, the raw
`docs/benchmarks/packed_phase23_*` reports, preservation manifest and source
archive. The next measured priorities are trajectory checkpointing/contract
chunking, QuantLib-equivalent tensor financial interpolation, and reducing
GPU launch/synchronization overhead. Extending event/boundary batching requires
explicit contracts and independent validation. Major architectural pivots still
require discussion with the user.

## Phase 22 foundation

The user requested a realistic multi-contract neural SDE calibration benchmark
before committing to a full GPU implementation. Preserve Phase 21 implementation
`2cd9e5f` and evidence checkpoint `3e66775`. Continue Approach A and both
differentiation modes; a major pivot away from QuantLib still requires discussion.
The validated Phase 22 implementation checkpoint is **`c7d6d38`**; its source,
binary and evidence hashes are in
[the preservation manifest](../benchmarks/calibration_phase22_preservation.json).

Phase 22 adds `OptionQuote` and `LatentCalibration` over the existing `fd_latent`
engine, with absolute model time across maturities and exact per-contract
gradient accumulation to limit memory. A bounded neural stochastic-variance
example jointly optimizes its network and initial variance. Nine independent
analytic-Heston quotes cover three strikes and maturities, with six held-out
quotes. Calibration on 81 x 17 nodes reduces price RMSE from 0.397705 to
0.007409 dollars in 15 L-BFGS iterations / 16 objective evaluations (572.8 s).
The initial variance moves from .03 to .043203. On 81 x 33 nodes, frozen-model
training/held-out RMSE is .005847 / .002838 dollars. The final gradient norm
has not met the optimizer tolerance; this is an accurate price fit, not a
claim of optimizer convergence or unique model identification.
The 161 x 65 / 96-steps-per-year frozen-model check gives training/held-out
RMSE .003773 / .004384 dollars. Individual prices move by up to .016354 dollars
from the training mesh, so calibration residual alone is not a numerical error
certificate. The benchmark's synthetic quote scale is .05 dollars.

Financial checks independently reconstruct the risk-neutral log-price
generator, including Ito correction, covariance, discounting and boundary
rows. Analytic Black-Scholes reduction, put-call parity, price bounds, strike
shape, mesh/time/domain controls and separate neural/initial-state finite
differences accompany the fit. The native pricing arithmetic and upstream
QuantLib dependency closure are unchanged; native additions are diagnostic
profiling scopes and a financial-generator probe. Full details and the next
batching milestone are recorded at the top of
[docs/architecture.md](architecture.md).

The complete build passes **1,756 tests**. Across 44 isolated workload workers,
matched price vectors are identical and maximum recorded gradient difference
is 2.70e-12. At 21 x 9, adjoints reduce nine-contract objective/VJP time from
10.107 to 2.308 s and peak process growth from 309.6 to 59.7 MB. The actual
81 x 17 adjoint workload takes 34.182 s and 717.8 MB of peak growth. Full AD
was compared on smaller grids; no refined full-mode performance is inferred.
The next recommended milestone is CPU packed tensor coefficient assembly and
common-expiry contract batching through QuantLib's existing ADI stages, followed
by a measured device-resident GPU prototype. Current scalar mixed-coefficient
assembly and autograd indexing dominate the measured small-network workload;
neural field evaluation alone is not the acceleration target.

## Phase 21 foundation

The preceding phase generalized Approach A to jointly time/state-dependent
neural SDEs with vector latent states. Phase 21 adds a dimension-independent
tensor coefficient operator around QuantLib's existing stencils and ADI
rollback, with PSD covariance from diffusion loadings. `fd_latent` reuses the
original 2D/3D financial solvers for one/two latent factors, imposes risk-neutral
stock drift from yield curves, and supports differentiable latent initial states.
The previous validated milestone is `4e90e37`. The complete Phase 21 contracts,
independent tests, measured evidence and extension limits are recorded at the
top of [docs/architecture.md](architecture.md).

Phase 21 implementation checkpoint `2cd9e5f` passes **1,742 tests** (102 new).
Generic equations/VJPs cover 1D through 4D; Heston prices match the original
QuantLib operator within 1.25e-14 in recorded refinements, and the refined
analytic error is 6.92e-5. Full/adjoint neural training agrees for one/two latent
factors. Measured adjoint totals are 2.77x--3.97x faster for the selected small
CPU workloads; the 21 x 13 peak process growth falls from 344.6 to 80.0 MB.
The APIs remain CPU float64 and unbatched, with first-order equation adjoints,
fixed meshes/times, model-specific boundary requirements and retained rollback
trajectories. Broader financial dimensions, boundary contracts, coefficient
materialization and checkpointing are concrete next work.

The user approved implementation following the
[architectural review](architecture.md#architectural-review-general-differentiable-pde-solvers-2026-11-01).
The priority is **general differentiable QuantLib numerical infrastructure**,
with meaningful validated progress beyond the 1D engine. Preserve existing
optimized methods, full AD, local equation adjoints, and financial conventions.
This supersedes the review's earlier feedback gate and prospective optimization
targets in historical phase reports.

The review finds that current local solve/step adjoints already implement the
required discrete coefficient VJPs, composing through the complete supported
rollback without large solution Jacobians. The main limitations are software
coverage and contracts: a 1D vanilla/local-volatility bridge, three exposed
schemes, first-order-only equation adjoints, no general transpose/residual
interface, and no supported multidimensional engine at the review baseline.
The approved direction is to extend Approach A with structured
operator/solve/stage capabilities while retaining existing optimized backends.
The completed milestone integrates QuantLib's actual 2D two-asset local-volatility
basket engine, mixed-derivative stencil, and ADI schemes, with independent
manufactured/dense references and stock QuantLib financial checks.

The clean pre-implementation Git baseline is
`5b0de25e50246abdd3cd939fca1ae7a6678589d9`. It preserves all Phase 19 source,
tests, benchmark reports, source archives, and the architectural assessment.

Phase 20 passes the complete 1,640-test suite, including 168 new cases. Its
structured directional equation adjoints and indexed-stencil pullbacks compose
through four original QuantLib ADI schemes. Dense 2D/3D equation and 2D-stage
VJPs agree within 2.23e-15; stock basket prices match exactly in the recorded
checks, and coefficient bumps agree within 5.05e-9. On a 31 × 21 grid with ten
steps, measured neural-loss total falls from 6.43 s in full AD to 2.25 s with
adjoints; cold process growth falls from 1.15 GB to 360 MB. All 282 matched
1D baseline/current price/VJP samples are identical. See the
[Phase 20 record](architecture.md#phase-20-structured-adjoints-and-a-real-multidimensional-quantlib-engine)
for source preservation, raw evidence, architecture and limitations. Krylov
damping, custom boundaries, general joint-state financial coefficient fields,
higher-order equation adjoints, trajectory checkpointing, batching and GPU
financial execution remain open.

The Phase 19 work already in progress is preserved. The selected build passes
1,472 tests plus 319 feature-disabled checks. Matched price/VJP and curvature
samples are identical across the change. It removes unrequested full-mode
assembly-gradient work; the selected timing repeat is mixed and does not show
a uniform speedup. The source snapshot, reproducible overlay, test evidence,
and pilot/selected measurements are recorded in
[the Phase 19 report](architecture.md#phase-19-requested-full-mode-assembly-gradients).

## 1. Vision and scope

Develop a differentiable option-pricing framework that can be integrated into PyTorch training. The primary application is **PDE-constrained inverse problems** in which neural networks parameterize coefficients of stochastic models. The long-term goal is efficient calibration of **neural SDEs**, including two-factor models and American-style options.

The aim is not merely to calculate scalar Greeks. The pricing layer should accept neural model coefficients and return accurate **vector–Jacobian products** (VJPs) so that a pricing loss can be differentiated with respect to network parameters.

Retain as much of QuantLib's financial functionality as practical: instruments, conventions, yield curves, dividend schedules, calendar handling, exercise types, pricing-engine interfaces, and established numerical behavior.

## 2. Example mathematical problems

### Neural local volatility under a risk-neutral measure

\[
 dS_t=(r-q)S_t\,dt+\sigma_\theta(t,S_t)S_t\,dW_t^{\mathbb Q}.
\]

Train a neural network to minimize a loss of the form

\[
 \mathcal J(\theta)=\sum_i w_i\big(V_i^{\mathrm{PDE}}(\theta)-V_i^{\mathrm{market}}\big)^2+\lambda R(\theta).
\]

The desired differentiation chain is

\[
\theta\xrightarrow{\text{PyTorch}}\Sigma_\theta\xrightarrow{\text{PDE solver}}\mathbf V\xrightarrow{\text{loss}}\mathcal J,
\qquad
\nabla_\theta\mathcal J=J_{\Sigma_\theta}(\theta)^\top\nabla_\Sigma\mathcal J.
\]

### Two-factor neural SDE (long-term)

\[
\begin{aligned}
 dS_t&=(r-q)S_t\,dt+S_t\sigma_\theta(t,S_t,Y_t)\,dW_t^{S,\mathbb Q},\\
 dY_t&=b_\theta(t,S_t,Y_t)\,dt+a_\theta(t,S_t,Y_t)\,dW_t^{Y,\mathbb Q},\\
 d\langle W^S,W^Y\rangle_t&=\rho\,dt.
\end{aligned}
\]

Initially preserve the risk-neutral stock drift, rather than learning an arbitrary drift under Q, and specify sufficient conditions for martingality. Learning the P-dynamics and market price of risk is a potential future extension, not the immediate implementation target.

## 3. Current architecture options

- **A — QuantLib scalar integration with LibTorch.** Investigate customizing `QL_REAL` (as QuantLibAAD does with XAD) using a scalar-compatible `TorchReal` wrapper, with the eventual possibility of custom adjoint operations. **Current priority for feasibility testing.** Do not assume this will automatically enable efficient GPU vectorization.
- **B — QuantLib + XAD + PyTorch bridge.** Retain QuantLib's C++ PDE solver and use XAD reverse-mode AD to calculate coefficient sensitivities, then pass VJPs to PyTorch. Potential CPU baseline; not currently preferred as the main development direction.
- **C — Native tensor PDE solver.** Implement/reimplement key numerical operations in PyTorch/LibTorch or C++/CUDA, while optionally retaining QuantLib for instruments and conventions. More control over batching, GPU execution, and custom adjoints; more numerical engineering work.
- **Hybrid possibility.** Keep the QuantLib instrument/market-data layer, but replace only performance-critical numerical kernels or engines. This may be preferable to either a global scalar substitution or a complete QuantLib rewrite.

**Approach A now runs the actual QuantLib finite-difference vanilla engine on
CPU with both full AD and custom first-order VJPs.** See `docs/architecture.md`
for the implemented financial scope, measured results, and limitations.
Whole-library compatibility, efficient scaling, GPU execution, and two-factor
integration remain open; scalar substitution alone does not establish them.

**Phase 18 extends the original assembly-matrix handoff to first-order steps
and inference.** Each step saves its original matrix and returns one fresh
matrix gradient, preserving the existing arithmetic, pivot checks, and explicit
first-order contract. Full-mode solver states and higher/mixed derivatives are
unchanged. The default build passes 1,371 tests, including 91 new checks, plus
403 feature-disabled checks. The selected CPU-8 HighQoS repeat improves neural
pricing loss from 7.44 to 6.08 ms (1.22x), and 60/121 surface pricing from
12.69 to 11.59 ms (1.09x). The 120/241 total is unchanged in this repeat;
pilot gains and inference timings vary. All 1,504 matched price/VJP samples
are identical, and combined curvature validation covers 1,128 samples. At
30/61, graph size falls from 387 to 288 nodes and backward tensor calls from
2,624 to 1,700; saved-tensor bytes are unchanged. Source snapshots, unchanged
full-mode controls, remaining assembly/materialization work, and candidates for
reducing full-mode derivative overhead are recorded in `docs/architecture.md`.

**Phase 17 retains the original assembled band tensor across the full-mode
operator/step boundary.** Its private owner is invalidated before every
assembly, while original Array coefficients remain available. One matrix input
and gradient remove row-selection edges; original tensors, higher/mixed AD,
solver elimination states, and fallback behavior are preserved. The selected
default build passes 1,280 tests, including 93 new checks, plus 409
owner-disabled checks. In matched CPU-8 HighQoS workers, 30/61 neural loss
improves from 8.63 to 7.35 ms (1.17x) and 120/241 surface pricing from 31.12
to 28.62 ms (1.09x) against Phase 16. Gains vary by workload and run. The
same selected run measures 10.04 to 7.35 ms and 40.70 to 28.62 ms against
Phase 15. All 940 matched owner-on/off price/VJP samples are identical;
combined curvature validation covers 2,256 samples. At 30/61, graph size
falls from 488 to 389 nodes and backward tensor calls from 3,183 to 2,358.
A slower prototype with temporary row views remains archived. Source snapshots,
runtime controls, limitations, and remaining adjoint packing/materialization
work are recorded in `docs/architecture.md`.

**Phase 16 combines full-mode band/RHS preparation in one elementary autograd
operation.** Original inputs, expression order, guarded solve selection, and
the complete elimination-state graph remain preserved. The selected build
passes 1,187 tests, including 101 new checks, plus 294 preparation-disabled
checks. With fresh HighQoS workers pinned to logical CPU 8, 30/61 neural
pricing loss improves from 9.53 to 7.21 ms (1.32x), and 60/121 surface pricing
from 17.76 to 13.67 ms (1.30x). Default-Thomas surface/neural Hessian-vector
workloads improve from 36.94/39.65 to 34.46/36.39 ms. The unrestricted pilot
also improves; an intermediate repeat shows host timing changes on unchanged
paths and remains archived separately. The library does not set affinity.
Validation covers 5,660 price/VJP comparisons and 1,504 curvature samples,
with maximum combined HVP difference 1.43e-13. The 30/61 graph falls from
807 to 488 nodes; saved-tensor byte counts rise slightly. Remaining band
selection/materialization costs motivate further investigation. Measurements,
runtime controls, source snapshots, and limitations are in `docs/architecture.md`.

**Phase 15 fuses time interpolation and squaring at the local-volatility
sampling boundary.** Original coefficient rows remain saved, higher
derivatives and incoming dual weights use functional formulas, and transformed
inputs retain the original expression. The selected default build passes
1,086 tests and 216 disabled-operation checks; the pilot also passes 74 checks
with spatial caching disabled. Reversed-order selected 60/121 surface adjoint
pricing improves from 21.95 to 19.49 ms (1.13x), and 120/241 from 45.20 to
42.30 ms (1.07x). Full-mode gains are smaller and mixed; inference and
Hessian-vector timings show no consistent gain. All 1,598 selected matched
price/VJP samples are identical. Combined curvature validation covers 1,128
samples with maximum HVP difference 1.32e-13. At 30/61 the full graph falls
from 906 to 807 nodes, with unchanged saved-tensor byte counts. Source
snapshots, controls, limitations, and the remaining step/assembly work are
recorded in `docs/architecture.md`.

**Phase 14 composes supported full-mode steps directly from existing tensor
kernels.** It reuses the identical Douglas application and avoids repeated
Array conversions while preserving expression ordering, auto-domain/residual
checks, full elimination-state derivatives, and original fallbacks. Events,
financial objects, active settings, and boundary callbacks keep their existing
handling. The selected default build passes 1,012 tests and 229 step-disabled
fallback checks; the opt-in revision also passes 78 checks with native apply
forward and Thomas arithmetic disabled. Matched selected 120/241 full-mode
surface pricing improves from 83.57 to 65.36 ms (1.28x); 30/61 neural loss
improves from 18.99 to 16.15 ms (1.18x). Default-Thomas 30/61 surface/neural
Hessian-vector workloads improve from 72.79/72.98 to 64.93/65.45 ms.
Validation covers 3,070 fixed-reference price/VJP comparisons and 1,128
cross-revision curvature samples, with maximum HVP difference 1.78e-13.
Adjoint/inference paths are controls, and PCR curvature results are mixed.
Source snapshots, matched selected measurements, validation, and remaining
coefficient/step costs are recorded in `docs/architecture.md`.

**Phase 13 reuses immutable spatial derivative bases within each price.**
Each QuantLib operator owns its cache; only passive C++ constants are retained.
Tensor-valued bases, transformed inputs, and aliases keep the existing path.
That selected default build passed 934 tests and 120 cache-disabled fallback
checks, including fresh graphs, changing meshes, exception cleanup, higher
derivatives, and forward AD. Matched selected 120/241 surface adjoints improve
from 51.28 to 47.25 ms and inference from 35.78 to 30.00 ms. Gains vary by
workload: 30/61 neural full mode is slightly slower in the selected repeat.
At 30/61, profiling confirms one basis pack plus
32 reuses and 384 fewer forward tensor calls. Selected validation covers 4,806
fixed-reference price/VJP comparisons; its 1,410 cache-on/off samples are
identical. The opt-in comparison separately records
1,432 identical cache-on/off price/VJP samples and 564 curvature samples against
solver references. Hessian-vector timings show no distinct gain, and graph
sizes remain unchanged. An existing fake-tracing limitation with all passive
constants disabled is documented separately. Source snapshots, measured
improvements, limitations, and the remaining solve/assembly bookkeeping costs
are recorded in `docs/architecture.md`.

**Phase 12 fixes incoming forward tangents in full-mode VJPs.** The earlier
assembly/application/PCR-stage native backward paths lost tangents carried by
VJP weights when `create_graph=False`. Eighteen new tests reproduced the gap;
all 30 mixed-AD tests now pass by selecting the existing tensor backward
formulas for those inputs. That revision passed 911 tests and 10 configured
mixed-AD fallback checks. Ordinary pricing/VJPs and first-order adjoint contracts
are unchanged. Timing controls retain surface/neural adjoints near 12.3 ms and
full Hessian-vector workloads near 72 ms at 30/61. Negative controls, source
snapshots, 2,580 pricing/VJP comparisons, and 752 cross-revision curvature samples
are recorded in `docs/architecture.md`. Repeated packing of immutable spatial
derivative operators motivated Phase 13's work on reuse with preserved tensor
graphs and fresh state between prices.

**Phase 11 enables native CPU arithmetic inside the first-order assembly,
application, and step boundaries.** Matched selected 30/61 surface adjoint
pricing improves from 23.8 to 12.0 ms (1.98×), neural loss from 24.3 to 12.7 ms
(1.91×), and surface inference from 15.8 to 6.7 ms (2.35×), against Phase 10.
The selected build passes 881 tests, 139 fallback tests, and 108 step-only
tests. Forward expressions, reflected endpoints, scaled solve ordering,
distinct pivot checks, and the compact transpose-solve VJP are retained.
A near-singular case verifies rejection of a tiny nonzero transpose pivot
even when the forward result is finite. Original input tensors remain saved;
the first-order-only contract and unsupported-configuration fallbacks persist.
Full-mode timings and graph-connected higher derivatives retain Phase 10's
path. The financial API remains CPU-only and unbatched. Source snapshots,
measurements, validation, and remaining bookkeeping costs are recorded in
`docs/architecture.md`.

**Phase 10 enables fused CPU Thomas elimination and its differentiated reverse
recurrence inside the existing guarded auto domain.** Selected matched 30/61
surface pricing improves from 24.8 to 19.3 ms (1.29×), neural loss from 26.0 to
20.3 ms (1.28×), and 60/121 surface pricing from 51.9 to 38.6 ms (1.34×), against
Phase 9. Complete 30/61 surface/neural Hessian-vector workloads improve from
202.9/206.8 to 73.0/72.8 ms (2.78×/2.84×). The selected default build passes
810 tests plus 233 explicit fallback
tests. Pivots, factors, and intermediate RHS values retain gradient edges;
three differentiable bidiagonal recurrences reverse all intermediate-output
dependencies for higher derivatives. A correct but slow scalar-recorded
higher-derivative prototype was rejected and remains archived. The native
forward preserves the pinned compact QuantLib recurrence's expression order,
attribution, and license. Domain/residual checks, original-solver fallbacks,
financial objects/events, and first-order adjoint contracts are preserved.
Forward duals and transforms retain tensor paths. CPU batch probes improve,
but the financial API remains one contract on CPU. See `docs/architecture.md`
for selected measurements, curvature checks, source snapshots, limitations,
and the remaining array/operator bookkeeping and adjoint-copy costs.

**Phase 9 enables native CPU solver-selection predicates and full-AD elementary
operator arithmetic.** Matched 30/61 surface pricing improves from 41.1 to
24.5 ms (1.68×), neural loss from 43.2 to 26.7 ms (1.62×), and 60/121 surface
pricing from 87.0 to 51.3 ms (1.69×), against Phase 8. The selected default build
passes 726 tests plus 213 explicit fallback tests. Original coefficient tensors
remain saved for graph-connected higher derivatives; forward AD and function
transforms retain tensor paths. Direct forward arithmetic preserves the pinned
QuantLib expression ordering and neighbor indices, with source attribution;
an earlier wrapped-forward version had a copying regression and remains archived.
Financial objects, event order, admissibility/residual decisions, original-solver
fallbacks, and first-order adjoint contracts are preserved. Source snapshots,
7,590 fixed-reference comparisons, and remaining solve/bookkeeping bottlenecks
are recorded in `docs/architecture.md`. This phase makes no new GPU pricing claim.

**Phase 8 fuses elementary PCR elimination stages and their ordinary backward
on CPU, while retaining graph-connected higher derivatives.** Full-AD 30/61
surface pricing improves from 93.2 to 43.0 ms (2.17×), and neural pricing loss
from 91.7 to 42.1 ms (2.18×), against Phase 7 in matched HighQoS workers.
The selected build passes 513 tests. Forward duals, function transforms, and
fake tensors retain functional tensor operations. Measured single-RHS batch
regressions are avoided by retaining the tensor recurrence when bands would
expand across RHS batches or the workload exceeds 32,768 broadcast rows.
Multi-RHS CPU batches also improve; updated GPU comparisons must use this
stronger CPU baseline. The financial engine, admissibility/residual checks,
events, and original-solver fallbacks are unchanged. See `docs/architecture.md`
for evidence, source snapshots, limitations, and remaining bottlenecks.

**Phase 7 adds graph-connected spatial reuse and native VJPs for supported
steps, band application, and local-volatility operator assembly.** Each boundary
calls the original QuantLib forward algorithms and returns all coefficient
dependencies. Unsupported step/operator configurations retain the original
path. The final build passes 417 tests, including active-setting/boundary
fallbacks, dense references, lazy-negative views, and numerical differences.
Matched neural-loss pricing at 30/61 under explicit Windows HighQoS improves
from 41.3 to 24.4 ms in adjoint mode (1.69×); full AD improves from 95.8 to
90.1 ms (1.06×). The surface adjoint improves 1.80×. System-managed workers show
185 to 91 ms (2.04×) for neural adjoint, with large runtime transitions linked
by controlled diagnostics to Windows execution-speed policy. Benchmark-only
`--windows-qos high` makes that environment choice explicit; the library does
not change caller process policy. PCR arithmetic fusion was slower and remains opt-in. See the Phase 7
section in `docs/architecture.md` for retained evidence and remaining bottlenecks.

**Phase 6 adds vector spline expressions and guarded parallel cyclic reduction
for full AD.** The default `kernels="auto"` selects the tensor solver only for
well-scaled, strictly diagonally dominant systems with an acceptable residual;
other systems retain the original QuantLib solver. `kernels="vectorized"`
preserves upstream full-AD Thomas elimination. Matched 30/61 CPU comparisons
show a further roughly 4.1× improvement for full AD and 1.45× for adjoint over
Phase 5. The build passes 328 tests, including fallback, higher derivatives,
forward AD, spline public data, and independent batched CPU/CUDA solve VJPs.
Standalone CUDA solves do not establish batched or GPU financial pricing.

**Phase 5 substantially reduces CPU engine overhead** with persistent tensor
arrays, vector coefficient sampling/array arithmetic, and native-value execution
inside the verified solve adjoint. The measured 30/61 adjoint price/VJP workloads
improve by 49–63× over Phase 4; full AD and batched/GPU pricing remain active
performance investigations. See `docs/architecture.md` for raw evidence,
derivative contracts, limitations, and the next measured bottlenecks.

**Phase 4 targets tensor-level parallelism and eventual batched GPU pricing
and gradients within Approach A.** Band application and assembly now have
vectorized hooks in the existing QuantLib engine. Standalone tensor primitives
support broadcast batch axes and multiple right-hand sides on CPU/CUDA, with
independent value/VJP checks. The financial engine remains a single-contract
CPU API; tensor-kernel CUDA support is not yet GPU option pricing.

CPU correctness comes first, while CPU and GPU batch-throughput experiments
inform the next kernel boundaries. Large single-contract scalar overhead is
not grounds to reject the architecture before investigating persistent tensor
state, batched solves, and device-resident rollback. Moving individual
TorchReal scalar operations to CUDA is not the intended acceleration strategy.
See `docs/architecture.md` for measured results, limitations, and the staged
plan. A whole-engine replacement or other major architectural pivot still
requires discussion.


## 4. Differentiation strategies to investigate

A central objective of QuantLibTorch is to investigate and compare **two reverse-mode differentiation strategies** for numerical PDE pricing solvers.

These strategies should be treated as distinct research directions. Neither is assumed to be superior before numerical and computational validation.

### Strategy 1 — Full automatic differentiation through the numerical solver

Use a general-purpose reverse-mode automatic differentiation framework, such as PyTorch autograd or XAD, to differentiate the operations executed by the numerical PDE solver.

This may include:
- Assembly of finite-difference operators.
- Internal operations of linear-system solvers.
- Time-stepping operations.
- Interpolation and boundary conditions.
- Bermudan exercise projections.

The objective is to investigate whether QuantLib's existing numerical algorithms can be made compatible with automatic differentiation, with particular interest in LibTorch integration.

Evaluate gradient correctness, computational overhead, memory requirements, GPU compatibility, and performance.

### Strategy 2 — Custom discrete adjoints / analytical vector–Jacobian products

Investigate custom reverse-mode adjoints derived from the mathematical structure of the numerical algorithms, rather than automatically differentiating every elementary operation executed internally.

For example, consider a discrete time-stepping equation:

\[
A_n(\theta)u_{n+1}=B_n(\theta)u_n.
\]

Given an upstream adjoint \(\bar u_{n+1}\), introduce the adjoint variable \(\lambda_n\) satisfying

\[
A_n(\theta)^\top\lambda_n=\bar u_{n+1}.
\]

The adjoint contribution to the previous state is

\[
\bar u_n \mathrel{+}=B_n(\theta)^\top\lambda_n.
\]

For a perturbation of the model parameters, the corresponding contribution is

\[
d\mathcal J_n
=
\lambda_n^\top
\left[
(dB_n)u_n-(dA_n)u_{n+1}
\right].
\]

This permits differentiation of a linear solve without recording the elementary operations of its factorization or elimination algorithm.

Investigate custom adjoints at both the individual numerical-operation level and, where advantageous, the full time-stepping solver level.

The adjoint must still account for dependencies across time steps, exercise projections, and other parameter-dependent operations.

Explore memory-efficient reverse propagation, checkpointing, recomputation, and tensor-based CPU/GPU implementations.




## 5. QuantLib source locations to inspect

- QuantLib main repository: https://github.com/lballabio/QuantLib
- `ql/qldefines.hpp` — `QL_REAL` default and `QL_INCLUDE_FIRST` hook.
- `ql/types.hpp` — aliases using `QL_REAL`.
- QuantLibAAD: https://github.com/auto-differentiation/QuantLibAAD
- `ql/qlaad.hpp` in QuantLibAAD — `xad::AReal<double>` integration, math and Boost compatibility.
- XAD: https://github.com/auto-differentiation/xad
- QuantLibAAD Python bridge: https://github.com/auto-differentiation/

**External research reference:** Lei Fan and Justin Sirignano (2024), *Machine Learning Methods for Pricing Financial Derivatives*: https://arxiv.org/abs/2406.00459. Review the exact forward discretization, neural coefficient parameterization, Bermudan projection and GPU batching before claiming replication.
