> Historical snapshot through Phase 26, archived during release preparation.
> Statements of current support and next work describe their original checkpoints.
> Use [current architecture](../architecture.md), [capabilities](../capabilities.md)
> and [release support](../release.md) for current contracts. Command paths are relative
> to the repository root. Markdown links have been relocated; evidence is unchanged.

# Approach A: QuantLib pricing-engine integration

## Phase 26: original Heston/shout engines and differentiable financial markets

The extended integration session starts from clean checkpoint `092abf1` and
preserves Approach A, prior APIs and numerical backends. Four integrated
milestones extend useful QuantLib functionality; the acceptance gates are in
[roadmap.md](roadmap.md), with exact public contracts in
[capabilities.md](../capabilities.md).

### Original engines and explicit geometry

`fd_heston` runs the original `FdHestonVanillaEngine`, `HestonProcess`,
`HestonModel`, `FdmHestonOp` and `FdmHestonSolver`. `heston_mesh` calls the
original engine's solver-description construction to obtain its variance and
equity axes using passive reference numbers. Pricing supplies those fixed axes
through a scoped mesh hook, while retaining original payoff averaging, boundary
rows, dividend handling, exercise conditions, ADI rollback and interpolation.
No Heston PDE is independently reimplemented. Five model parameters, spot,
flat rate/yield and noninitial dated curve factors retain their graphs.
European, American and Bermudan calls/puts and spot cash dividends are supported.
Four ADI schemes work in full/adjoint modes. Full mode supports original damping
and higher reverse derivatives; adjoint damping explicitly rejects the
unsupported multidimensional Krylov backward path.

`fd_shout` runs original `FdBlackScholesShoutEngine` with
`FdmShoutLogInnerValueCalculator` and its original escrowed-dividend adjustment.
American and Bermudan schedules specify shout opportunities; European permits
only maturity. `shout_mesh` constructs the original equity mesher with the same
escrowed adjustment. Six 1D schemes, damping, constant Black volatility,
spot/rate/yield/curve sensitivities, full higher derivatives and first-order
equation adjoints reuse the existing numerical infrastructure. Independent
stock engine prices are checked at the mesh preparation parameters; fixed-mesh
derivatives are checked against centered differences and scalar/full/adjoint
paths. Stock Python bindings omit the shout inner-value calculator, so stock
automatic-mesh bumps are explicitly labelled as different geometry sensitivities.

`HestonMesh`/`ShoutMesh` accept immutable custom axes as well as original
automatic meshes. Geometry is explicit and passive; parameter differences must
reuse it. `HestonModel` registers five ordinary trainable parameters or fixed
buffers, while custom modules may impose constraints and return transformed
tensors. `FdHestonEngine`/`FdShoutEngine` compose with `PricingLayer` and construct
fresh graphs on each call. Neither scalar engine is claimed to run on CUDA or
batch contracts through the packed latent infrastructure.

The pinned source checkout is not edited or linked as a stock-double library.
`engine-mesh-hooks.patch` separately records fixed-mesh injection and omission
of unused Greek calculations. `heston-process-compatibility.patch` records
explicit failure stubs for unexposed Monte Carlo evolution, density and variance
sampling methods whose complex/special-function implementations cannot compile
with TorchReal. PDE-used process construction, parameters, drift/diffusion,
time and original quantile mesher are preserved. Generic scalar math overlays
remain separate; no unsupported method silently casts active parameters.

### Reusable financial graphs and conventions

`financial_curve` builds original `FlatForward` or log-linear `DiscountCurve`
objects from tensor inputs. `fd_latent` retains their graphs through coefficient
assembly, drift and discounting. `LatentBatch.price` accepts fresh financial
overrides; `MultiAssetBatch.price(markets=[...])` accepts per-asset overrides.
Missing keys inherit passive plan defaults; explicit curve `None` restores a
flat curve. Active shadowed flat rates and active first discount factors fail
explicitly, including forward-dual inputs.

Batch geometry, event maps, payoff values and spline bases stay passive.
Original QuantLib forwards are evaluated over the actual rollback intervals
on CPU, then transferred to CUDA with their graphs. Multi-asset checkpoint
replay returns rate/yield VJPs to the original input curves. Passive market
defaults do not request extra rate/yield replay gradients. Full mode preserves
higher derivatives; adjoint mode rejects higher/forward derivatives. Existing
exercise, dividend and neural-model batching contracts remain intact.

Heston/shout engine curves and process/volatility times consistently use either
Actual/365 Fixed (default) or Actual/360. `adjust_date`, `advance_date` and
`year_fraction` expose original NullCalendar, WeekendsOnly, TARGET, five
business-day conventions, business-day/period advancement and end-of-month
rules. Dates are adjusted explicitly before instrument construction; no pricing
API silently changes exercise/dividend dates. Other engines retain their prior
day-count convention.

### Validation and remaining boundaries

Focused independent tests cover stock Heston original/frozen-mesh prices,
five-parameter and market/curve bumps, full damping, custom nonuniform axes,
shout stock prices and fixed-geometry derivatives, early-exercise and dividend
events, higher derivatives, per-asset/checkpoint gradients and CPU/CUDA parity.
Calendar tests compare 75 holiday/month-end adjustments and 216 period advances
plus leap-year, negative and zero day-count periods with isolated stock QuantLib.
Executed explanatory notebooks cover Heston, shout, and financial-input graphs.
The final native Windows build succeeds with its source fingerprint verified.
`python -m pytest -q --junitxml=build/phase26_pytest.xml` reports **2,224 passed,
1 skipped, 23 warnings in 689.59 seconds**. All 2,089 baseline tests are retained;
135 new cases cover the integrated features. The skip is the existing CPU
Thomas strict-dominance exception; warnings are existing tracing/deprecation
warnings. The pinned QuantLib checkout remains clean. Raw test/native/overlay
hashes and requirement evidence are in
[library_phase26_validation.json](../benchmarks/library_phase26_validation.json).
Implementation checkpoint **`c80448a`** preserves all four milestones, tests,
executed notebooks and initial regression evidence. The final local 0.3.0 wheel
and source archive are checked separately in
[packaging_phase26.json](../benchmarks/packaging_phase26.json), including installed
Heston/shout gradients and original calendar behavior. No portable/Linux release
claim is made.

The executed 15-by-9-node, eight-step Heston notebook gives call value
9.01333440255 and initial-variance derivative 50.2067408265 versus the fixed-mesh
bump 50.2067409247. The 31-node, 12-step American shout example gives
11.2158975411 and vega 52.0494995458 versus 52.0494995481 by fixed-mesh bump.
The financial-input notebook checks matched CPU/CUDA prices and rate, discount
factor and spot gradients, then performs three optimizer steps with fresh
graphs. These modest teaching grids are examples, not general accuracy targets.

No new GPU speedup or continuous-model error bound is inferred. Automatic mesh
derivatives, Heston adjoint damping, Heston/shout batching, stochastic local-vol
leverage/quanto, batch dividend-amount derivatives, richer payoff classes and
further calendar/day-count support remain subsequent milestones. Existing
`fd_vanilla`/`fd_basket` markets remain passive. CEV would require supported
active noncentral-chi-square boundary functions; barrier/rebate engines require
explicit knock-in/out boundary contracts and independent tests. Neither is
claimed supported merely from a dependency audit.

## Phase 25: general-purpose financial API, inputs, events and packaging

The user redirected priority toward practical library functionality, preserving
Approach A and the clean Phase 24 baseline `6121346`. The implementation plan
and acceptance gates are in [roadmap.md](roadmap.md); the exact supported
capability matrix is in [capabilities.md](../capabilities.md). Historical records
below describe their respective checkpoints, not today's feature limits.

### Reusable API and financial-input differentiation

`VanillaOption`, `Exercise`, `EquityMarket` and reusable engine configurations
compose through `PricingLayer(torch.nn.Module)`. An optional registered neural
model supplies tensors; market tensors enter via forward-call overrides. Fixed
configuration is copied and validated. Every call creates a fresh graph.
Existing functions, calibration objects and batch plans remain public.

`analytic_european` exposes the already compiled, original
`AnalyticEuropeanEngine` and `AnalyticDividendEuropeanEngine`. `SimpleQuote`,
`FlatForward`, `DiscountCurve`, `BlackConstantVol`, payoffs and `FixedDividend`
receive graph-connected TorchReal scalars. Supported inputs are volatility,
spot, strike, continuous rate/yield, noninitial dated discount factors and cash
amounts. CPU float64 full AD and higher reverse derivatives are tested; no
equation adjoint or CUDA claim applies. The first discount is fixed at one and
active rates shadowed by dated curves raise. The analytic cash-dividend model
is QuantLib's discounted-spot approximation, distinct from FD spot jumps.

The original `fd_vanilla` now exposes Craig–Sneyd, modified Craig–Sneyd and
Hundsdorfer in addition to Douglas, Crank–Nicolson and implicit Euler, plus
the original engine's escrowed-dividend model. No pricing formulas, meshes or
event ordering are replaced. Its market data remains passive: simply making
automatic-mesher inputs tensors would violate existing passive geometry/cache
contracts. Fixed-mesh market sensitivities are a subsequent milestone.

### CPU/CUDA financial events

`LatentBatch` and `MultiAssetBatch` support American/Bermudan exercise with one
or two latent factors, and spot cash dividends with one latent factor. The
original `FdmStepConditionComposite` supplies stopping times, `FdmLogInnerValue`
supplies intrinsic values, and `FiniteDifferenceModel` supplies callbacks,
including maturity events and the theta snapshot before the first event.
Tensor exercise uses the same strict comparison as upstream, keeping the
continuation derivative at equality. Cash maps are obtained by applying the
actual `FdmDividendHandler` to cardinal vectors; they preserve clamp,
interpolation and extrapolation semantics. Dividends precede exercise at
coincident dates. This passive linear-map adapter is separate from unchanged
upstream ADI stage generation. `prepare_packed.py` records event-source hashes.

Multi-asset grouping includes exercise and dividend schedules. Compatible
assets retain separate strikes, grids, curves and dividend amounts. Different
schedules form separate groups, preserving order. Checkpointed backward
replays all financial events. Full AD and first-order equation VJPs remain
available on CPU/CUDA with existing checked solve restrictions. Two-latent
cash dividends raise because the upstream handler's multidimensional layout
does not cover that full tensor product. Damping/custom boundaries remain
unsupported; dividend maps use quadratic stock-axis storage at plan setup.

### Validation and measured event workload

The new engine/input tests pass 41 cases: isolated stock-QuantLib prices and
market bumps, gamma, dated curves/cash/neural gradcheck and gradgradcheck,
validation failures, and new FD schemes/dividend models with all three
exercise styles and both modes. The API tests pass 30 cases, including neural
registration, repeated graph construction, direct-functional equality and
configuration validation. Three standalone layer examples run successfully;
the FD full/adjoint Bermudan example gives price 8.72662740 and network-gradient
norm 4.30803150 in both modes.

The 67 new packed-event tests pass CPU/CUDA, four schemes, full/adjoint,
one/two latent states, original scalar references, isolated stock Heston
prices/all six parameter bumps, initial-state sensitivities, coincident and
maturity events, asset-specific dividends, schedule grouping, checkpoint
spacing and curvature. Existing numerical and benchmark evidence is preserved.

[financial_events_phase25.json](../benchmarks/financial_events_phase25.json)
records a four-asset/six-put, 21×9-node, 12-step workload with two dividends,
shared neural fields, one warmup and three measured complete price/VJP samples.
Setup is separate, CPU uses one thread, and GPU timings are synchronized.
Maximum matched price/parameter-gradient discrepancies are 2.84e-14/4.52e-14.

| Events | CPU full / adjoint, s | CUDA full / adjoint, s |
| --- | ---: | ---: |
| American + cash dividends | .125828 / .153513 | .452515 / .475113 |
| Bermudan + cash dividends | .131143 / .144758 | .449201 / .483892 |

This small workload favors CPU. CUDA peak allocation is approximately 86.97
MiB in full AD and 21.28 MiB with checkpointed adjoints. These measurements
establish event feature parity and the stated workload behavior; they do not
extend the prior European throughput claim to larger event-heavy portfolios.
The report retains the measured binary's original hash. A subsequent native
rebuild for final packaging has a different binary hash; all recorded event
source hashes still match. Final regression evidence identifies that later
binary separately, without relabeling the timing samples.

### Distribution and remaining work

Local wheels include the native library and conservative Python/platform tags.
Packaging verifies pinned QuantLib identity, PyTorch/Python/platform/C++ ABI,
binary hash and native-source fingerprint; missing/stale native artifacts fail
with explicit build instructions. Runtime checks ABI metadata. Source archives
include native build inputs, examples, tests and licenses, excluding binaries
and third-party checkouts. Editable setup remains compatible with prebuild
installation. Windows wheel installation into a separate directory followed
by native pricing/gradient execution is verified. Portable release binaries
and Linux runtime verification remain future work. See [install.md](../install.md).

Next work is fixed-mesh rate/curve sensitivities, additional original model/FD
engines, mesh/boundary/convention controls, packed damping and a verified
Windows/Linux release matrix. Full AD remains the general fallback only for
tested compatible operators. No new whole-library differentiability claim is
made.

### Final integrated regression

Implementation checkpoint **`6e3e6a3`** preserves the Phase 25 library changes,
tests, documentation and raw financial-event evidence.

The final native Windows build succeeds. `python -m pytest -q
--junitxml=build/phase25_pytest.xml` reports **2,089 passed, 1 skipped,
23 warnings in 463.47 seconds**, adding 149 tests to the 1,940-test baseline.
The intentional skip is the existing CPU Thomas test of a CUDA-only strict
dominance restriction; warnings are existing tensor-tracing/deprecation
warnings. The pinned QuantLib checkout is clean. Final native metadata,
test-output hashes and requirement evidence are recorded in
[library_phase25_validation.json](../benchmarks/library_phase25_validation.json).
[packaging_phase25.json](../benchmarks/packaging_phase25.json) records local wheel
and source-archive hashes plus isolated installed analytic, FD and American
dividend-batch prices/gradients. No remote Linux execution is claimed.

## Phase 24: independent assets, device financial readout, checkpointed VJPs

`MultiAssetBatch` extends the existing packed QuantLib integration to independent
asset operators with multiple contract right-hand sides. Its public prices are
`[assets, contracts]`, fields are `[assets, time, nodes, directions, ...]`, and
the internal solution is `[assets, nodes, contracts]`. Node ordering remains
QuantLib's coordinate-zero-fastest tensor-product layout. The original scalar
and Phase 23 packed paths remain independently available. The clean baseline
is `27ba330`; its DLL is also copied to `build/phase24_baseline` locally.

Each asset has its own jointly state-dependent drift/diffusion, initial state,
curve data, grid coordinates, strikes and call/put types. Within an asset,
`LatentBatch.from_contracts` retains its strict configuration checks. Across
assets, equal grid lengths, absolute coefficient times, rollback horizon,
scheme, step count, device and contract count permit a common execution group.
Different grid coordinates and curves are supported using separate stacked
QuantLib bases and forward rates. Incompatible shapes or schedules execute in
separate groups in the original asset order. Per-asset tensor lists support
different group shapes. Ragged contract counts require separate plans. Cash
dividends, early exercise, damping and custom boundaries still raise explicitly.
Noise-factor counts must agree within an execution group; differently sized
loading matrices require separate plans. Curves and all numerical configuration
remain fixed rather than differentiable. Multiple expiries within one asset
require separate compatible plans (or the existing maturity-grouping calibration
adapter); the rectangular multi-asset interface does not pad or merge them.

All spatial derivative bases, averaged payoffs and curve rates are obtained
from the pinned QuantLib objects. The unchanged QuantLib scheduler supplies
the exact interval list, including its theta snapshot. `prepare_packed.py`
generates a second adapter from the same four original ADI bodies, changing
only the enclosing class name and `Array` type. The Phase 23 generated file
remains byte-identical. No new discretization or stock-double QuantLib linkage
is introduced, and the original scalar compatibility overlay is unchanged.

Natural cubic cardinal polynomials are constructed by QuantLib's own
`CubicInterpolation` on passive basis vectors at plan creation. Tensor evaluation
then reproduces bicubic/multicubic natural spline readout on the selected device,
including initial-state derivatives. This exchanges per-price CPU interpolation
and grid transfers for fixed basis storage of order `assets * sum(axis_size^2)`.
Tensor reductions change floating-point summation order; explicit comparisons
cover nonuniform meshes, exact interior knots, first derivatives and curvature.

The CUDA path keeps coefficients, rollback states, readout and gradients on
device. PCR retains the Phase 23 strict forward row-dominance restriction and
the `1e-10` scaled residual requirement, including transpose solves. Validation
flags are aggregated and inspected once per group forward and once per complete
adjoint, with no per-stage host scalar extraction. Failures raise at the call
boundary and leave the CUDA context usable. This defers error reporting; it
does not skip validation or make unsupported equations admissible.

Full AD retains its elementary operation graph and higher derivatives. The
first-order adjoint saves segment boundaries, rematerializes each segment,
and applies stencil and transposed directional-solve VJPs. Shared coefficient
VJPs sum over contracts, never over the independent asset dimension. Local
checkpoint leaves are explicit first-order rematerialization boundaries: their
exact VJPs return to the original drift/covariance tensors and through `L L^T`
to the neural graph. No solution Jacobian is formed. With `S` actual intervals
and segment length `K`, solution storage scales roughly as
`O((S/K + K) * assets * contracts * nodes)`, plus coefficient/network storage
and fixed bases. This is not constant-memory execution. Full AD is not
checkpointed by this interface; its memory can be substantially larger.

The native build passes **1,940 tests**, including 75 new tests (one CPU-only
domain skip), covering
CPU/CUDA, all four schemes, one/two latent factors, full and checkpointed
adjoints, all field/state VJPs against existing engines, independent dense
natural splines, finite differences, a joint neural/initial-state training step,
heterogeneous grouping, invalid calls and graph cleanup. Implementation
checkpoint `43d6113` preserves the implementation. A final regression repeat
strengthens the asset fixtures to use different nonuniform spacings, so sharing
one asset's derivative basis accidentally cannot pass the asset comparison.

### Complete neural pricing and loss-gradient throughput

The [isolated workload reports](../benchmarks/multi_asset_phase24_workloads/commands_scaling.json)
use PyTorch 2.10.0+cu126, float64, an RTX 4060 Laptop GPU (8,188 MiB reported
device memory, driver 566.36), and one CPU intra/inter-op thread with Windows
HighQoS and logical-processor-8 affinity. The library never sets these policies.
Each timing worker has one warm-up and three measured repetitions. Setup is
separate. Timing includes shared neural evaluation, per-asset embeddings and
bounded initial-state transformations, financial pricing, mean squared loss,
backward, price/gradient materialization, gradient clearing and explicit GC.
Forward-only runs include the same pricing/residual/materialization/cleanup
work with autograd disabled. Explicit device synchronization brackets measured
forward/backward stages. This is a controlled single-thread CPU comparison,
not a claim against a tuned all-core CPU implementation.

The serial baseline uses the existing **tensorized, native-Thomas one-contract
`LatentBatch`**, streaming exact loss gradients and reevaluating the neural
fields per contract. It is much faster than the historical scalar TorchReal
engine. The contract-batched control uses the Phase 23 plan with all contracts
for one asset, iterating over assets. The new CPU/CUDA paths execute independent
assets together and use device spline readout. These controls separate gains
from contract sharing, asset batching and financial readout; they are not
standalone linear-solver benchmarks.

| Assets x contracts | Grid / steps | Serial CPU, s | Contract-batched CPU, s | Two-level CPU, s | Two-level CUDA, s | CUDA contracts/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 x 1 | 21 x 9 / 12 | .07455 | .07288 | .07858 | .37080 | 2.70 |
| 1 x 16 | 21 x 9 / 12 | .74997 | .18225 | .08562 | .37211 | 43.0 |
| 8 x 16 | 21 x 9 / 12 | 5.81659 | 1.25636 | .17389 | .37973 | 337.1 |
| 20 x 30 | 21 x 9 / 12 | 28.60178 | 5.16789 | .61645 | .39360 | 1,524.4 |
| 32 x 32 | 21 x 9 / 12 | 44.75838 | 8.20834 | .93140 | .43426 | 2,358.1 |
| 20 x 30 | 41 x 17 / 24 | 56.63157 | 14.30997 | 4.06494 | .75931 | 790.2 |

The 20 x 30 refined case is **5.35x faster than the new batched CPU path** and
74.58x faster than optimized serial CPU. Forward-only time on that case is
15.46324 / 3.97091 / .97847 / .29455 seconds in the same column order; CUDA
therefore prices approximately 2,037 contracts/s including the stated overhead.
On the smaller 21 x 9 mesh, 20 x 30 CUDA is 1.57x faster than batched CPU, and
32 x 32 is 2.14x faster. CUDA remains slower at 1 x 1, 1 x 16 and 8 x 16.
The one-contract CPU adjoint also has a small checkpoint/recomputation overhead
against the preserved single-asset path. GPU acceleration is workload-dependent.

Across **36 workload reports and 95 matched sample comparisons**, maximum
price and parameter-gradient differences are **4.62e-14 / 1.15e-14**. The
[summary verifier](../benchmarks/multi_asset_phase24_summary.json) compares all
matched price vectors and every neural, embedding and initial-state gradient,
and additionally checks the later memory/full-mode reports. Setup times, raw
timing variation and inference samples remain in each worker report. These
matches establish agreement on the selected discrete equations; neither the
coarse grid nor agreement between implementations certifies financial accuracy.

The additional full-AD 1 x 16 case takes .07622 s on CPU / .34401 s on CUDA;
checkpointed adjoints are not universally faster. A two-latent 2 x 3 workload
on 9 x 5 x 5 nodes with four steps takes .08370 / .22634 s. This validates
integrated 3D execution and gradients on both devices, not converged 3D finance.

### Memory, transfers and synchronization

Separate three-cycle memory workers measure the 20 x 30 / 41 x 17 / 24-step
adjoint objective. Host values are OS process working-set high-water marks;
growth is above the post-setup worker's existing high-water mark. They include
native/runtime allocations, not just tensors. GPU values are PyTorch allocator
statistics and exclude other driver allocations. These are distinct domains.

| Execution | Peak host RSS, MiB | Host peak growth, MiB | Peak GPU allocated / reserved, MiB |
| --- | ---: | ---: | ---: |
| Optimized serial CPU | 718.08 | 56.17 | — |
| Contract-batched CPU | 824.39 | 316.75 | — |
| Two-level CPU, K=4 | 701.39 | 186.20 | — |
| Two-level CUDA, K=4 | 1,034.79 | 415.06 | 244.28 / 272.00 |

CUDA live allocation returns to **32.4238 MiB** after each of all three cleanup
cycles. Cached reservation remains. Host private bytes need not return to their
initial value: the K=4 CUDA worker records 1,765.11 / 1,785.75 / 1,785.87 MiB.
These figures do not establish an overall host+device memory saving over serial
CPU. Checkpoint spacing is material: K=1 / 4 / 12 gives peak device allocation
223.97 / 244.28 / 444.41 MiB (reservation 242 / 272 / 458 MiB), with identical
post-cleanup live allocation. Local stage memory has a significant constant;
the optimum K is workload-dependent. No constant-memory claim is made.

The separate 8 x 16 CUDA profiler records **two one-byte device-to-host copies**
over neural evaluation, price, residual and backward: the forward and adjoint
validation flags. There are zero host-to-device copies in that region, no
solution-grid/coefficient transfers and no CPU financial readout. Model/plan
setup transfers occur before profiling. Normal timing additionally includes
explicit price/loss/gradient logging transfers; the transfer-attribution worker
disables that materialization. The trace contains two `aten::item` and two
`cudaStreamSynchronize` calls. Five `cudaDeviceSynchronize` calls include the
benchmark stage/cleanup brackets and profiler finalization. There is no
per-directional-solve host synchronization.

The trace also records **560 device-to-device copies totaling 105,828,928 bytes**;
device residency does not eliminate packing/reordering or kernel-launch costs.
The complete CPU/CUDA traces are preserved as gzip archives next to their
[profiler reports](../benchmarks/multi_asset_phase24_workloads/profile_8x16_21x9_12_multi_cuda_1_adjoint_k4.json).
Profiler durations and memory are not used as ordinary timing/memory samples.

### Independent gradients and numerical refinement

On the actual **20 x 30 / 41 x 17** neural workload, two central-difference
sizes independently check shared-network, asset-embedding and raw-initial-state
directions. Maximum absolute error over both devices and all six checks is
**3.56e-10**; at the smaller network bump it is at most **3.25e-11**. Raw AD,
finite differences, directions' definitions and both bump sizes are retained
in the [CPU](../benchmarks/multi_asset_phase24_gradients_cpu.json) and
[CUDA](../benchmarks/multi_asset_phase24_gradients_cuda.json) reports.

Two independent Heston assets with different parameters/rates and three strikes
each are compared with isolated stock-QuantLib 1.40 analytic prices and central
parameter bumps. The multi-asset CPU and CUDA refinements agree within
**1.60e-13 in price / 4.10e-12 in recorded sensitivities**, including the frozen
neural fixture below. Selected maximum analytic-Heston errors are:

| Grid / steps | Price, dollars | dPrice/dsigma | dPrice/drho | dPrice/dv0 |
| --- | ---: | ---: | ---: | ---: |
| 21 x 9 / 12 | .125878 | .591024 | .073051 | 5.979643 |
| 81 x 17 / 24 | .010133 | .103360 | .033300 | .236566 |
| 81 x 33 / 48 | .008466 | .027084 | .005821 | .074557 |
| 161 x 65 / 96 | .002281 | .006388 | .001074 | .016505 |

Time-only and latent/stock-only controls are preserved in the raw reports.
Time refinement alone does not remove spatial error: 81 x 17 / 48 gives a
.010387-dollar price error. Sensitivity columns have distinct parameter units;
their magnitudes should not be combined into an unscaled accuracy norm.

The frozen, jointly state-dependent neural fixture uses two assets and three
contracts per asset, the benchmark's network architecture, distinct embeddings,
and distinct raw initial states. Parameters stay fixed while grids change.
Per-price sensitivities are recorded separately for each asset, avoiding
cancellation across assets. Maximum changes relative to **161 x 65 / 96** are:

| Grid / steps | Price change, dollars | Network-direction change | Raw-initial sensitivity change |
| --- | ---: | ---: | ---: |
| 21 x 9 / 12 | .133370 | .00124135 | .00196250 |
| 41 x 9 / 12, stock only | .029820 | .00027856 | .00043755 |
| 21 x 17 / 12, latent only | .133374 | .00124024 | .00196678 |
| 21 x 9 / 24, time only | .133554 | .00123916 | .00196002 |
| 41 x 17 / 24 | .030043 | .00027684 | .00044203 |
| 81 x 33 / 48 | .005622 | .00005082 | .00008167 |

These are changes to the finest tested discrete model, not certified
continuous-model error bounds. The throughput grid remains under-resolved for
some uses: the .030043-dollar movement must not be concealed by the speedup.
Stock resolution dominates this smooth neural fixture; the Heston controls
also show the importance of latent resolution. Existing domain/boundary and
coefficient-time conventions are unchanged and require model-specific checks.
See the [validation summary](../benchmarks/multi_asset_phase24_validation_summary.json)
and its full [CPU](../benchmarks/multi_asset_phase24_convergence_cpu.json) /
[CUDA](../benchmarks/multi_asset_phase24_convergence_cuda.json) records.

The runnable 20 x 30 CUDA training example reports price RMSE decreasing from
.886675 to .671017 across five objective evaluations while updating the shared
network, embeddings and initial latent states. This is an integration smoke
test using synthetic Black-Scholes targets, not optimizer convergence, a
real-market calibration, or identification of latent dynamics.

### Reproduction, preservation and remaining work

```text
python scripts/build.py --jobs 2
python -m pytest -q
python scripts/run_multi_asset_benchmarks.py --suite scaling
python scripts/run_multi_asset_benchmarks.py --suite diagnostics
python scripts/summarize_multi_asset.py --input-dir docs/benchmarks/multi_asset_phase24_workloads --output docs/benchmarks/multi_asset_phase24_summary.json
python scripts/check_multi_asset_gradients.py --device cuda --output docs/benchmarks/multi_asset_phase24_gradients_cuda.json
python scripts/diagnose_multi_asset.py --device cuda --output docs/benchmarks/multi_asset_phase24_convergence_cuda.json
python scripts/check_multi_asset_gradients.py --device cpu --output docs/benchmarks/multi_asset_phase24_gradients_cpu.json
python scripts/diagnose_multi_asset.py --device cpu --output docs/benchmarks/multi_asset_phase24_convergence_cpu.json
python scripts/summarize_multi_asset_validation.py
python examples/multi_asset_neural_sde.py --device cuda --assets 20 --contracts 30
python scripts/preserve_multi_asset.py
```

Benchmark workers resume only complete reports with matching source/binary
fingerprints and configuration. Use a fresh output directory after code changes.
The preservation command audits the recorded workspace and requires its build/
test logs and locally saved baseline DLL; ordinary rebuilds and benchmarks do
not require that audit.
The [preservation manifest](../benchmarks/multi_asset_phase24_preservation.json)
records current source/binary hashes, generated stage provenance, the source
archive, raw evidence, and unchanged Phase 23 artifacts. The original binary
hash matches the prior preservation manifest. Original scalar solvers, scalar
compatibility patches, dependency closure, and the Phase 23 generated stage
file are unchanged. The new adapter has no stock-double QuantLib linkage.

Next work is measured fusion/reuse of directional elimination and stencil
kernels, reducing device packing, automatic checkpoint/contract-chunk selection,
and extending event/boundary compatibility with independent validation. More
heterogeneous maturities require explicit per-asset contract grouping rather
than padding/rescaling. Broader CUDA solve domains, financially converged
two-latent calibration, all-core CPU comparisons and larger neural models
remain unmeasured. The preserved APIs remain the controls for that work.

Reproduction entry points are `scripts/benchmark_multi_asset.py`, its isolated
worker driver `scripts/run_multi_asset_benchmarks.py`, and
`scripts/diagnose_multi_asset.py`. `examples/multi_asset_neural_sde.py` demonstrates
shared network training with asset embeddings and distinct trainable initial
latent states. Its default small mesh demonstrates execution, not accuracy.

## Phase 23: packed coefficients and compatible-contract calibration

The CPU implementation adds `LatentBatch`, `diffusion_rollback_batch`, and
`LatentCalibration(execution="batched")`. The original single-contract paths
remain unchanged and serial calibration remains the default. A plan owns only
fixed geometry, topology, payoffs and financial configuration; each price call
builds fresh graph-connected model coefficients. Common-expiry calibration
groups preserve the original quote order and accumulate one group at a time.

The packed adapter obtains every spatial derivative basis and reflected index
from pinned QuantLib 1.40. It assembles directional and all mixed coefficients
as whole tensors, keeping the original stock endpoint Ito closure and interval
forward rates. Directional lines share factorization across contract RHS.
The existing checked native Thomas arithmetic supplies full AD, including its
elimination-state derivatives. Equation adjoints solve the transposed system
and sum shared coefficient VJPs over contracts. No coefficient or initial-state
tensor is detached into a surrogate leaf.

`scripts/prepare_packed.py` generates the four ADI **step bodies** from the
pinned upstream sources; their only body substitution is `Array` to
`at::Tensor`. Copyright notices and source/body hashes accompany the generated
file. The unchanged `FiniteDifferenceModel` supplies the scheduler. The price
path retains the original theta-snapshot stopping time, averaged payoff and
QuantLib bicubic/multicubic interpolation. This specializes state storage and
operator kernels within Approach A; it does not replace the numerical schemes.
Thomas arithmetic ordering differs slightly from the scalar scaled directional
recurrence, so comparisons use explicit floating-point tolerances.

Plans require common axes, absolute coefficient times, expiry, curves, scheme,
steps, initial state and fields. `from_contracts` rejects conflicting fixed
configuration. European call/put strikes can vary. Exercise schedules, cash
dividends, damping and custom boundary conditions are rejected. Original
`fd_latent` American/Bermudan capabilities remain available. Mutating a packed
calibration's numerical configuration raises instead of reusing stale plans.

The CPU checkpoint passes **1,844 full-suite tests**, with one additional
configuration-mutation check passing separately. Its 88 new initial tests
cover independent dense equations/VJPs in 1D--4D, all four ADI schemes,
2D/3D neural option prices and shared gradients versus the original paths,
isolated stock QuantLib Heston prices/bumps, full-mode curvature, strided
inputs, numerical differences, invalid configurations, fresh graphs, exception
recovery and graph cleanup.

The preliminary controlled CPU pilot (one thread, HighQoS, logical CPU 8,
one warm-up and three repetitions) measures a complete three-contract,
common-expiry neural objective/VJP on 21 x 9 nodes. Serial median is **1.34165 s**
and packed median **0.09489 s**, including explicit Python GC/cleanup, a **14.14x**
ratio. Packed forward/backward/cleanup medians are .02323/.04181/.02685 s.
The sampled price vectors differ by at most **1.07e-14**. This is a pilot on a
coarse grid, not a financial convergence claim or a general scaling result.
Raw reports are in `docs/benchmarks/packed_phase23_cpu_pilot`.

The selected results below follow that CPU checkpoint. The complete workload
reports and independent accuracy checks preserve their source/binary hashes.

### Validated device-resident prototype

The CPU checkpoint is `93cd935`. The subsequent complete build passes
**1,865 tests**, including **109 packed-path tests**. CUDA validation compares
prices, shared network/coefficient gradients and initial-state derivatives with
CPU across all four ADI schemes in 2D and 3D. Full-mode second derivatives,
explicit first-order-only adjoints, unsupported CUDA solve domains and failed
call recovery are covered. The original 1,756-test coverage remains intact.

`LatentBatch(device="cuda")` retains fixed bases, topology and payoffs on the
selected GPU. The calibration adapter also retains state/time grids, targets,
model parameters and sampled neural fields there. Coefficient assembly,
applications and rollback states remain device-resident. The existing tensor
PCR kernel solves each directional line with multiple contract RHS; all stage
formulas and time scheduling are the same generated QuantLib bodies as CPU.
The prototype requires strictly row-diagonally-dominant forward systems and
checks finite solutions and a scaled residual of at most `1e-10`, including
transposed adjoint solves. A failed check raises; the problem is never changed
to satisfy the GPU solver. This is a narrower admissible domain than CPU Thomas.

The initial state and final grid values are transferred through connected
PyTorch operations for original QuantLib CPU spline readout, and the price
vector returns to the plan device. Transfer gradients remain connected.
Each GPU directional solve synchronizes for its validation check. This is a
measured prototype, not an all-GPU financial engine or a GPU speedup claim.
The library does not change process thread counts, affinity or device settings.

Trajectory storage still grows with steps, nodes and contracts. Plans share
operators and coefficients, but each contract retains its own state trajectory.
The first-order adjoint removes elimination tapes; it does not checkpoint the
entire PDE. Full AD preserves higher derivatives, while exercise/interpolation
branches remain piecewise and fixed axes/dates/strikes are not differentiable.
PSD neural covariance alone does not guarantee stable centered ADI equations,
adequate domains or a valid arbitrary latent boundary model. Numerical price
and sensitivity refinement remains necessary for each calibrated model.

### Complete calibration workloads and scaling

[47 isolated workers](../benchmarks/packed_phase23_workloads/commands.json) cover
serial CPU, packed CPU and the targeted GPU path. Each timing worker uses one
warm-up and three measurements, one PyTorch intra/inter-op thread, HighQoS and
logical CPU 8 affinity. Setup is timed separately. Each objective includes
neural field evaluation, initial-state transformation, pricing, scaled residuals,
backward, the once-per-objective model penalty, gradient/log materialization,
graph release, gradient clearing and explicit Python GC. Forward, backward and
explicit cleanup timings are also retained; their separately computed medians
need not sum to the median total. Memory and profiler workers are separate from
timing workers. Raw samples, configuration and actual runtime controls are kept.

The [summary](../benchmarks/packed_phase23_summary.json) verifies every matched
price and parameter-gradient sample. Across CPU/full/equation/GPU comparisons,
maximum price difference is **2.49e-14** and maximum gradient difference is
**1.25e-11**. These are agreement checks for the discrete problem, not bounds
on its error against the continuous financial model.

| Nodes | Contracts | Mode | Serial CPU, s | Packed CPU, s | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| 21 x 9 | 1 | full AD | 1.6933 | 0.06466 | 26.2x |
| 21 x 9 | 9 | full AD | 15.0158 | 0.11859 | 126.6x |
| 21 x 9 | 1 | equation adjoint | 0.42460 | 0.06878 | 6.17x |
| 21 x 9 | 9 | equation adjoint | 3.67735 | 0.12265 | 30.0x |
| 81 x 17 | 1 | equation adjoint | 5.84206 | 0.14554 | 40.1x |
| 81 x 17 | 3 | equation adjoint | 17.90149 | 0.27744 | 64.5x |
| 81 x 17 | 9 | equation adjoint | 57.67325 | 0.45662 | 126.3x |
| 81 x 33 | 3 | equation adjoint | 73.82573 | 0.70506 | 104.7x |
| 41 x 17 | 27, three expiries | equation adjoint | 46.20277 | 0.73854 | 62.6x |

Common-expiry cases have one-year maturity and 12/24/48 steps on the
21 x 9 / 81 x 17 / 81 x 33 grids, respectively. The 27-contract case has nine
strikes at each of three expiries and the original 24-steps/year policy, minimum
four. It shares three operators, not one operator across maturities. The
1/3-contract subsets select center/extreme strikes from nine distinct strikes.
The 9 x 5 x 5 diagnostic three-contract workload takes 4.4998/1.7401 s in serial
full/adjoint mode and 0.10544/0.10445 s packed. This establishes execution and
VJP support for two latent factors, not financially converged 3D calibration.

The three-contract 9 x 5 profile reduces ATen calls from **438,407 to 20,718**
in full mode and **157,787 to 22,222** with equation adjoints. Operator assembly
and the number of state axes no longer create a separate scalar graph at every
stencil coefficient. The one-contract controls also improve substantially,
separating tensorization/caching benefits from contract sharing. Profiles are
for attribution; their durations are not substituted for ordinary timings.

Cold process peak working-set growth above the initialized worker's prior
high-water mark is:

| Grid, nine contracts | Mode | Serial CPU, MiB | Packed CPU, MiB |
| --- | --- | ---: | ---: |
| 21 x 9 | full AD | 295.70 | 26.79 |
| 21 x 9 | equation adjoint | 57.25 | 24.66 |
| 81 x 17 | equation adjoint | 684.62 | 190.66 |

Three complete cycles record post-cleanup working set/private bytes. On the
81 x 17 packed worker, private bytes after cleanup are 858.3/862.5/863.6 MiB;
the serial worker records 779.7/971.6/975.4 MiB. These include the initialized
runtime and retained allocator memory and are not live tensor counts. A
separate weak-reference regression verifies that plan reuse does not retain
model fields or price graphs after backward and deletion. At 161 x 65 with
nine contracts and 96 steps, packed CPU peak growth is **3,875.6 MiB**: trajectory
memory remains a practical scaling limit even after removing scalar overhead.

### GPU measurements and financial optimization

The device is an **NVIDIA GeForce RTX 4060 Laptop GPU**, PyTorch 2.10.0+cu126,
float64. These totals include GPU synchronization, connected CPU spline
readout, gradient materialization and cleanup.

| Nodes | Contracts, adjoint | Packed CPU, s | CUDA prototype, s |
| --- | ---: | ---: | ---: |
| 21 x 9 | 3 | 0.09847 | 0.31868 |
| 81 x 17 | 3 | 0.27744 | 0.71356 |
| 81 x 17 | 9 | 0.45662 | 0.88900 |
| 81 x 33 | 3 | 0.70506 | 1.41389 |
| 41 x 17 | 27, three expiries | 0.73854 | 1.52357 |
| 161 x 65 | 9 | 6.91342 | 6.08971 |

The small full-AD three-contract case is also slower on CUDA: .37922 versus
.08426 s. Only the largest measured adjoint case shows a net benefit: **1.14x**
by median. Its CPU samples span 6.821--7.228 s and GPU samples 4.565--6.140 s;
GPU timings vary substantially. This supports a targeted prototype, not a
general GPU advantage or a full production GPU engine. There is no serial
scalar timing or refined full-AD throughput inference for the largest grid.

For that large GPU case, peak allocated/reserved device memory is
**3.42 / 4.53 GiB**, and allocated memory returns to **21.41 MiB** after each
of three cleanup cycles while the reservation remains cached. Host peak growth
is **685.01 MiB**, additional to GPU memory. On 81 x 17, GPU host peak growth
is 513.25 MiB with 139.21 MiB peak device allocation. These are separate memory
domains; host RSS alone cannot establish a GPU memory saving. Per-solve
validation synchronization, PCR kernel launches, CPU spline readout and
trajectory storage are explicit remaining costs.

The original Phase 22 nine-quote, three-expiry calibration is repeated on
81 x 17 with the same network, penalty, initial parameters and L-BFGS settings.
[Packed adjoints](../benchmarks/packed_phase23_fit_adjoint.json) finish in
**8.689 s** and [packed full AD](../benchmarks/packed_phase23_fit_full.json) in
**7.959 s**, each with 15 accepted iterations and 16 objective evaluations.
Both reproduce final price RMSE **.0074091333** and initial variance
**.0432027627**. Final parameters differ between modes by at most **7.78e-15**,
and prices by **1.42e-14**. The historical serial Phase 22 fit took 572.8 s;
the controlled matched workload sweep above is the primary speed comparison.
The new full-AD path is competitive with equation adjoints on this workload;
adjoints are not universally faster. Final gradient norm remains **1.8133**,
so the optimizer still stops at its iteration budget, not its gradient
tolerance. Price fit does not establish unique neural SDE identification.

Independent objective bumps on the actual 81 x 17 grid give network-direction
and raw-initial-state errors **2.32e-8 / 1.02e-9** on CPU and
**2.33e-8 / 1.02e-9** on GPU at the smaller recorded bump. Full AD is separately
checked on 21 x 9, with errors **2.57e-8 / 4.34e-10**. Both bump sizes, AD
values and signed differences are retained in the gradient reports.

### Price and sensitivity convergence

The [frozen neural report](../benchmarks/packed_phase23_neural_convergence.json)
keeps the Phase 22 fitted parameters fixed and recomputes each of its nine
prices, a fixed network directional sensitivity (`.05*cos(parameter_index)`
within each network tensor), and the derivative with respect to raw initial
variance. Entries below are maximum absolute changes **relative to the finest
tested grid**, not certified continuous-model error bounds.

| Nodes / steps per year | Price change, dollars | Network-direction sensitivity change | Raw-initial sensitivity change |
| --- | ---: | ---: | ---: |
| 21 x 9 / 12 | .185446 | .00280186 | .186116 |
| 41 x 17 / 24 | .051020 | .00084583 | .020279 |
| 81 x 17 / 24 (training) | .016354 | .00085753 | .013817 |
| 81 x 17 / 48 (time only) | .011831 | .00086773 | .012443 |
| 81 x 33 / 24 (variance only) | .014387 | .00041116 | .005564 |
| 161 x 17 / 24 (stock only) | .012795 | .00087275 | .012188 |
| 81 x 33 / 48 | .009674 | .00020004 | .004188 |

The reference is 161 x 65 / 96, whose training RMSE is .00377272. Thus the
original **.016354-dollar** mesh movement persists after acceleration. Time
or stock refinement alone can leave a sensitivity error unchanged or larger;
combined refinement and latent-axis resolution matter. These representative
neural sensitivities do not certify convergence of the entire parameter
Jacobian. Existing Phase 22 domain/coefficient-time/held-out controls remain
preserved; they are not newly generalized to arbitrary learned models.

The [independent financial report](../benchmarks/packed_phase23_accuracy.json)
uses three one-year strikes (85/100/115). Heston price sensitivities to
`sigma`, `rho` and physical `v0` are compared against central bumps of isolated
stock-QuantLib analytic prices. Black-Scholes prices and analytic vega provide
a separate reduction. Maximum errors are:

| Grid / steps | Heston price, dollars | dPrice/dsigma | dPrice/drho | dPrice/dv0 | Black price, dollars | Black vega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 x 9 / 12 | .121979 | .591024 | .073051 | 5.979643 | .113012 | .519767 |
| 81 x 17 / 24 | .010133 | .103360 | .033300 | .236566 | .007736 | .031705 |
| 81 x 33 / 48 | .007332 | .027084 | .005821 | .074557 | .007746 | .031525 |
| 161 x 65 / 96 | .001960 | .006388 | .001074 | .016505 | .002067 | .008014 |

Sensitivity columns have their own parameter units and should not be compared
as one unscaled norm. Black put-call parity residual decreases from .045568
to .001284 dollars between the first and last grids. Finite-mesh parity is
not exact; no price clipping or parity adjustment is introduced for batching.

### Reproduction, preservation and next engineering work

```text
python scripts/build.py --jobs 2
python -m pytest -q
python scripts/benchmark_packed_calibration.py prepare --output build/packed_market_check.json
python scripts/run_packed_benchmarks.py --devices cpu,cuda --output-dir build/packed_workloads
python scripts/summarize_packed.py --input-dir build/packed_workloads --output build/packed_summary.json
python scripts/run_packed_validation.py
```

The validation driver regenerates the stored fit/refinement/gradient reports;
its exact invocations are in
[the command record](../benchmarks/packed_phase23_validation_commands.json).
The workload driver resumes only matching source/binary fingerprints; use a
fresh output directory for changed implementations. Timing workers are
sequential, and no performance instrumentation is mixed into normal timings.
The [preservation manifest](../benchmarks/packed_phase23_preservation.json) and
[source archive](../benchmarks/packed_phase23_source.zip) preserve the final source,
generated-stage hashes, binary fingerprint, test logs and evidence. CPU pilot
reports are checked against `93cd935`, separately from the final binary.
The original Phase 22 artifacts, scalar coefficient solver, dependency closure
and compatibility overlay remain unchanged and are verified by the preservation
script. The market fixture and generated stage files reproduce byte for byte.

The practical default for this model size is packed CPU. The measured path
toward larger calibration workloads is to reduce trajectory storage with
checkpointing and explicit contract chunking, tensorize QuantLib-equivalent
financial interpolation, and profile/fuse the GPU stage operations and
validation reductions before expanding device claims. Further batching of
exercise/dividend events requires explicit compatible event schedules and
independent financial checks. Accuracy budgets should use price **and**
sensitivity refinement, not just the optimizer residual. These extensions can
continue within the existing QuantLib infrastructure.

## Phase 22: multi-contract neural SDE calibration readiness

Phase 21 is preserved at `2cd9e5f` and `3e66775`, including its source and
evidence archives. This phase builds a calibration workflow over the same
`fd_latent` implementation. Native changes add only a financial-generator
validation endpoint and profiling scopes; pricing arithmetic, QuantLib
stencils, exercise, interpolation and both differentiation strategies remain.

### Objective, contracts and parameterization

[calibration.py](../../python/quantlibtorch/calibration.py) adds immutable
`OptionQuote` records and `LatentCalibration`. A quote has strike, maturity,
call/put type, target price and a fixed positive residual scale. The objective
is the mean squared scaled price error. All contracts use one valuation date,
one fixed spatial mesh and one **absolute** coefficient-time grid; shorter
maturities are not assigned different time-normalized model dynamics.
Time steps grow with Actual/365 Fixed maturity, subject to a fixed minimum.
The adapter presently exposes European quotes; Phase 21's other exercise
features remain accessible through `fd_latent`.

`loss()` retains a price-vector graph and supports higher derivatives with
full AD. `backward()` evaluates the identical separable objective one contract
at a time, recomputing the network fields and initial-state transform, then
accumulating into the original parameter gradients. Each PDE graph is freed
after its backward. No coefficient tensor is detached into a surrogate input;
only returned logging values are detached. This is exact first-order gradient
accumulation, not batching or an approximation to a portfolio Jacobian.
Stateful/stochastic networks and nonseparable portfolio penalties require a
different evaluation contract. A model penalty is applied once per objective,
outside the contract loop.

[calibrate_neural_sde.py](../../examples/calibrate_neural_sde.py) uses CIR variance
coordinates with a small time/log-stock/full-latent-state network. Bounded
neural corrections learn stock leverage, latent drift, volatility of variance
and correlation. The stock drift itself remains imposed by QuantLib's equity
adapter. Diffusion loadings use one common stock driver plus independent
latent drivers, producing PSD covariance also for two latent states. At zero
variance, latent diffusion vanishes and drift is strictly inward; at the upper
variance boundary the model remains mean reverting. These properties restrict
the benchmark model class and do not solve general latent-boundary selection.

Joint L-BFGS optimization updates network parameters and a sigmoid transform
of the initial variance, constrained to `(0,.12)` inside the fixed PDE domain.
The network parameters have a small explicit quadratic penalty. Latent
units and base CIR scales are fixed. Nine vanilla quotes do not identify all
37 model/state parameters: a good price fit is not proof of recovery of the
physical latent state or a unique learned SDE.

### Independent financial and numerical controls

[prepare_calibration_market.py](../../scripts/prepare_calibration_market.py)
generates nine independent analytic-Heston calls at strikes 85/100/115 and
three dated maturities near 3/6/12 months, plus six held-out quotes at strikes
90/100/110 and two intermediate maturities. Spot is 100, `r=.03`, `q=.01`,
and Heston parameters are `(kappa,theta,sigma,rho,v0)=(1.5,.04,.3,-.6,.045)`.
Each quote uses a five-cent residual scale, a synthetic accuracy target rather
than a claimed observed bid/ask spread. The
[market fixture](../benchmarks/calibration_phase22_market.json) is reproduced
byte-for-byte by a separate pinned stock-QuantLib 1.40 process. Target prices
are analytic, not generated by the calibration PDE. These are controlled
synthetic experiments; no live market data has been calibrated.

Spatial axes come from QuantLib's `Concentrating1dMesher` and are supplied to
the existing predefined mesher. The stock domain is `[20,400]`, variance
`[0,.5]`; concentration points are fixed at log(100) and .04. A separate domain
check appends outer nodes while preserving all prior interior coordinates.
Separate experiments change stock nodes, variance nodes, rollback time steps
and coefficient-time samples so their errors can be distinguished.

The initial Heston control found maximum price errors **0.1840** on 21 x 9,
**0.05370** on 41 x 17, and **0.01958** on 81 x 17 with the same time-step
policy. Doubling time steps on 41 x 17 only reduced the maximum to **0.05139**.
The combined 81 x 33 / 48-steps-per-year refinement gave maximum error
**0.01320** and RMSE **0.007514** over the nine quotes. Thus the 41-node stock
grid is inadequate for a five-cent quote scale; calibration uses 81 x 17
and is checked again on the combined refinement. These are measured errors
for this surface, not mesh-independent error bounds.
On the final 161 x 65 / 96-steps-per-year Heston control, RMSE falls to
**0.002041** and maximum price error to **0.003596** dollars against the same
independent analytic quotes.

[test_calibration.py](../../tests/test_calibration.py) independently reconstructs
the actual financial generator using polynomial-moment derivative matrices.
It checks killing on a constant, the log-price drift including `-C00/2`,
every mixed covariance entry, time interpolation, and all boundary/corner rows
in 2D and 3D. An explicit omitted-Ito negative control fails by `C00/2` in the
interior. Applying the generator to `exp(x)` converges to discounted stock
carry as the stock mesh is refined. Separate tests compare streamed and
retained-graph multi-maturity gradients, independent network/state central
differences, repeated gradient accumulation and full-mode curvature.
[test_neural_calibration_model.py](../../tests/test_neural_calibration_model.py)
checks model covariance/boundary restrictions and actual joint optimizer
updates in both modes.

In the interior, the tested financial generator is

```text
(r - q - C00/2) D_logS + sum_i b_i D_Yi
  + (1/2) sum_i Cii D_ii + sum_{i<j} Cij D_ij - r I.
```

The first loading row represents relative stock returns; the other rows are
absolute latent diffusion coefficients. Covariance is formed as `L L^T` at each
coefficient-time sample, then covariance (rather than loadings) is interpolated.
At stock endpoints, QuantLib's Heston closure removes the variance correction
from the drift and uses its zero pure-second-derivative row, corresponding to
the linear-in-spot asymptotic closure. Latent endpoints use the original
one-sided first derivatives, zero pure second derivatives and mixed stencils.
There is no new Dirichlet boundary condition or inferred general boundary
classification. The variance model's degeneracy and inward drift at zero make
this a meaningful restricted test. PSD alone guarantees neither a monotone
discrete operator nor stability of every neural model. Stock-carry convergence
and parity checks are numerical financial controls, not a general martingale
theorem for arbitrary coefficient fields.

### Calibration and financial results

The [recorded fit](../benchmarks/calibration_phase22_fit.json) uses 81 x 17 nodes,
24 rollback steps per year (minimum four), nine coefficient-time samples,
Hundsdorfer ADI, CPU float64, and streamed adjoints. Fifteen accepted L-BFGS
iterations required 16 closure evaluations and **572.8 seconds**, including
final inference. Price RMSE fell from **0.397705** to **0.007409** dollars;
initial variance moved from **0.0300** to **0.043203**. The target Heston
initial variance is .045, but neural leverage and the latent initial state can
compensate for each other. This experiment validates joint optimization, not
parameter identification. The final scaled-objective gradient norm is **1.813**:
the run stopped at its iteration budget, not its gradient tolerance. All trial
closure values, optimizer settings, initial/final parameters and accepted
iteration count are retained.

With those parameters frozen, the 81 x 33 / 48-steps-per-year grid gives
training RMSE **0.005847**, maximum training error **0.009811**, and held-out
RMSE **0.002838** dollars across six new strike/maturity pairs. Independent
stock-QuantLib Black implied-volatility inversion gives training/held-out RMSE
**5.84 / 1.31 absolute volatility basis points** on that grid. Price errors
and implied-volatility errors differ substantially for low-vega options;
the optimization objective remains explicitly in scaled price units.

Further refinement keeps every fitted parameter fixed:

| Spatial mesh | Steps/year | Training RMSE, dollars | Held-out RMSE, dollars |
| --- | ---: | ---: | ---: |
| 81 x 17 (training) | 24 | 0.007409 | not measured |
| 81 x 33 | 48 | 0.005847 | 0.002838 |
| 81 x 33 (time control) | 96 | 0.005612 | 0.002917 |
| 161 x 33 (stock control) | 48 | 0.003729 | 0.004774 |
| 161 x 65 (combined) | 96 | 0.003773 | 0.004384 |

On 161 x 65, maximum training/held-out errors are **0.007409 / 0.007145**
dollars. The largest training-contract price change from the training mesh is
**0.016354** dollars, and from 81 x 33 is **0.009674**. Thus a sub-cent training
RMSE does not by itself certify sub-cent numerical accuracy for each contract.
The residual and discretization effects are comparable; both remain below the
five-cent synthetic quote scale. Held-out target error need not decrease
monotonically with refinement of a frozen, discretely calibrated model.
The [further-refinement report](../benchmarks/calibration_phase22_refinement.json)
retains every price and the new QuantLib mesh coordinates. On the separate
41 x 17 control, extending the domain changes prices by at most **1.13e-7**,
and doubling coefficient-time samples changes them by **3.32e-5** dollars.
These controls support the selected domain/time sampling for this fitted model;
they are not uniform guarantees for other networks or tighter quote scales.

The [financial audit](../benchmarks/calibration_phase22_quote_audit.json) reports
no sampled call/put bound, call strike-monotonicity or strike-convexity
violations, and maximum put-call parity residual **0.004255** dollars.
In the independent constant-volatility reduction, maximum analytic
Black-Scholes price error drops from **0.03528** to **0.007746** dollars, and
maximum parity residual drops from **0.02376** to **0.004367**, between
41 x 17 / 24 and 81 x 33 / 48. Parity is not exact on these finite meshes;
the measured convergence must accompany any financial-consistency claim.
The audit covers the sampled contracts, not an exhaustive arbitrage test.

Independent central differences perturb network parameters and the raw
initial-state parameter separately. On the actual 81 x 17 calibration grid,
the smaller bump gives directional errors **1.46e-8** and **6.82e-10**,
respectively, including the state transform and the network penalty. Both
full AD and adjoints are also checked against independent bumps at 21 x 9;
the unit tests additionally bump the physical latent state directly and
check full-mode curvature. These are first-order adjoints of the numerical
equations, composed through neural fields and QuantLib interpolation, without
forming a contract-by-parameter Jacobian or differentiating solver iterations.

### Measurement protocol and reproduction

Timing, memory and tensor-profiler measurements run in **separate sequential
processes**, with one PyTorch intra-op/inter-op thread, Windows HighQoS and
logical CPU 8 affinity recorded by each worker. Timings use one warm-up and
the median of three measured complete objective/VJP evaluations, including
network evaluation, price residuals, the penalty and gradient materialization.
They exclude optimizer updates. The separate fit history includes L-BFGS
line-search evaluations. Memory is cold-process peak working-set growth above
the initialized worker's previous high-water mark; reports also retain the
absolute peak and private bytes. This is process memory, not a count of only
live tensor storage, and allocator/history effects matter.

The workload sweep uses 1/3/9 contracts, streamed and retained-graph objectives,
and one/two latent factors. For 1 and 3 contracts, all options have the first
maturity; 9 contracts add two later maturities. On the 21 x 9 mesh, nine
contracts therefore require 66 rollback steps in total versus four for one
contract. The diagnostic 9 x 5 mesh uses four steps for every maturity. Contract
count scaling must not be mistaken for equal work per contract on other meshes.
The two-latent experiment measures the real 3D pricing adapter on a small
mesh; it does not establish a financially converged two-factor calibration.

Tensor-profiler traces are collected per contract on bounded diagnostic meshes,
aggregated, and released before profiling the next contract. Shape/stack/memory
recording is disabled. The [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
describes profiling overhead and additional tensor retention from shape
recording. Trace durations are used for attribution; ordinary worker timings
are used for speed comparisons. A separate `cProfile` run of the complete
81 x 17 fitted-model objective attributes Python/native boundaries without
retaining millions of tensor events. Neither type of profile is a GPU estimate.
The initial full-AD two-latent trace on 9 x 5 x 5 was stopped after its working
set reached **7.675 GB**. This was profiler bookkeeping, distinct from the
ordinary two-latent price/VJP memory measurement. The
[stopped attempt](../benchmarks/calibration_phase22_profiler_attempt.json) is
preserved. The replacement two-latent event trace uses one contract on
4 x 4 x 4; its timings are not compared with the nine-contract 2D trace.

Use the repository's normal native build and `python -m pytest` workflow first.
The selected experiment can then be reproduced with the following commands;
append `--windows-qos high --affinity 8` to the main benchmark/workload drivers
to reproduce this host's controls. The audit/refinement/Python-profile helpers
apply those controls on Windows themselves.

```text
python scripts/prepare_calibration_market.py
python scripts/benchmark_calibration.py truth --output docs/benchmarks/calibration_phase22_truth.json
python scripts/benchmark_calibration.py fit --mesh space --iterations 15 --output docs/benchmarks/calibration_phase22_fit.json
python scripts/benchmark_calibration.py convergence --parameters docs/benchmarks/calibration_phase22_fit.json --output docs/benchmarks/calibration_phase22_convergence.json
python scripts/benchmark_calibration.py finance --output docs/benchmarks/calibration_phase22_finance.json
python scripts/audit_calibration_quotes.py
python scripts/refine_calibration.py
python scripts/profile_calibration.py
python scripts/run_calibration_benchmarks.py
```

Run `benchmark_calibration.py gradients` with the fitted `--parameters`, using
`--mesh scaling --mode full`, `--mesh scaling --mode adjoint`, and
`--mesh space --mode adjoint`; the exact arguments/output names are retained in
[commands.json](../benchmarks/calibration_phase22_commands.json). Then run
`scripts/summarize_calibration.py` and `scripts/plot_calibration.py` to verify
matched results and regenerate the figure. The workload launcher resumes
completed reports; use an empty `--output-dir` for new measurements and do not
mix reports from different sources or binaries. The preservation verifier
checks every recorded source and binary fingerprint.

### Profiling and the next batching/GPU milestone

The nine-contract 9 x 5 diagnostic trace makes **1,314,684 ATen calls** with
full AD and **472,824** with adjoints. In the adjoint trace, coefficient assembly
takes **119 ms** versus **251 ms** inside native forward pricing; backward
takes **645 ms**, including **141 ms** in `SelectBackward0` and **70 ms** in
`StackBackward0`. Directional solve VJPs take only **20 ms**, about 3% of the
backward span. Network-field evaluation takes **7.5 ms** across all nine
contracts. These inclusive spans overlap where nested and must not be summed
as independent costs. They identify operation/graph overhead; profiler times
are not the uninstrumented speed results.

Source inspection explains a specific remaining cause: `TensorDiffusionOp`
uses QuantLib's original `NinePointLinearOp::mult` to scale every mixed-stencil
coefficient separately at each node. Application and directional solves have
structured tensor/equation paths, but mixed coefficient scaling still builds
many scalar multiplication/selection edges. There are `d(d-1)/2` mixed maps.
The neural coefficient tensors are small enough that accelerating only the
network would miss most of this workload. Repeating operator assembly and
rollback for every strike adds work even when mesh, model, expiry and curves
are identical. Streaming reduces concurrent graph memory but does not share
those operations.

The next concrete milestone should be **CPU tensor coefficient assembly plus
same-expiry contract batching within the existing QuantLib rollback**:

1. Keep passive mesh topology and QuantLib derivative bases, and represent
   mixed/directional coefficients in packed tensors. Vectorize mixed-stencil
   scaling and its graph connection, preserving original index order and
   boundary folds. Cache only immutable spatial data across optimizer calls;
   never retain stale model tensors or graphs.
2. Group quotes by common model fields, expiry, mesh, curves, scheme and
   exercise schedule. Start
   with the three European strikes at each expiry in this fixture. Assemble
   coefficients once per interval and add a contract/RHS axis to structured
   application and directional solves, integrating with the original ADI
   stage orchestration and QuantLib payoff/readout conventions. Preserve the
   current scalar route as an independent regression reference.
3. Retain full AD and equation VJPs. For `A X = R`, solve
   `Lambda = A^{-T} bar_X`, return `bar_R = Lambda`, and reduce
   `bar_A = -sum_contract Lambda X^T` directly into the supported bands.
   Mixed-stencil coefficient pullbacks similarly reduce over contracts.
   Neither operation requires a dense state Jacobian. Different maturities
   retain their own absolute-time rollback groups.
4. Validate price/VJP equality against the present loop, independent model and
   initial-state bumps, and the same refined financial controls. Measure
   complete objectives, peak process memory and tensor-call counts at batch
   sizes 1/3/9, first on 81 x 17 and then on a controlled two-latent case.
   A successful result must reduce measured complete-workload cost while
   preserving numerical tolerances; fewer Python calls alone is insufficient.

After that CPU milestone, prototype device-resident batched structured kernels
and measure end-to-end transfer, assembly, rollback, interpolation and backward
cost on actual GPU hardware. Current CPU-only raw-pointer kernels, scalar
`TorchReal` accesses, host financial objects, validation synchronizations and
graph retention need explicit device boundaries. Launching CUDA operations
for the present scalar work is not supported by these measurements. No GPU
speedup or memory estimate is claimed here.

Higher latent dimension still multiplies spatial nodes, adds mixed operators,
and grows saved rollback trajectories; batching does not remove this cost.
Financial interpolation currently supports one/two latent states, while the
generic coefficient rollback is dimension-independent. A converged two-factor
financial calibration, more general latent boundary contracts, trajectory
checkpointing, noisy market quotes/weights and identifiability diagnostics
remain separate work. The current outcome is a validated synthetic calibration
workflow with a measured throughput bottleneck, not a production calibration
service or a fully general neural-SDE martingale guarantee.

### Recorded CPU workload results

All rows below evaluate nine contracts and include objective gradients. Model
parameters are the same deterministic initial values across matched workers;
the separate fitted-model profile uses the calibrated parameters. `Streamed`
means one contract graph at a time. Peak growth is decimal MB above each cold
worker's initialization, measured independently of timing and profiling.

| Grid / latent factors / execution | Full AD seconds | Adjoint seconds | Full / adjoint peak growth MB |
| --- | ---: | ---: | ---: |
| 9 x 5 / one / retained graphs | 1.527 | 0.429 | 262.9 / 59.7 |
| 9 x 5 / one / streamed | 1.603 | 0.393 | 34.4 / 10.2 |
| 21 x 9 / one / streamed | 10.107 | 2.308 | 309.6 / 59.7 |
| 13 x 5 / one / streamed | 2.212 | 0.629 | not measured |
| 13 x 5 x 5 / two / streamed | 21.750 | 8.580 | 502.8 / 157.2 |
| 81 x 17 / one / streamed | not measured | 34.182 | not measured / 717.8 |

At 21 x 9, adjoints reduce time **4.38x** and peak growth **5.19x**. In the
two-latent 13 x 5 x 5 case the corresponding reductions are **2.54x / 3.20x**.
Adding the second latent axis to the matched 13 x 5 setup multiplies nodes by
five and increases runtime **9.83x** in full AD and **13.65x** with adjoints;
there are now three directional maps and three mixed pairs instead of two and
one. This is a measured small-mesh scaling result, not an asymptotic estimate.
Full AD on the refined calibration grid was not benchmarked under this host's
memory constraints; there is no inferred full-mode runtime or memory value.
The actual 81 x 17 adjoint cold worker reaches **1.251 GB absolute peak**.

For the 9 x 5 diagnostic, increasing contracts from one to nine increases
retained-graph full-AD peak growth from **32.3 to 262.9 MB**; streaming changes
it from **32.5 to 34.4 MB**. Adjoint growth changes from **10.0 to 59.7 MB**
with retained graphs, versus **9.7 to 10.2 MB** streamed. Streaming is therefore
a useful memory control, while the runtime table shows that it does not provide
the arithmetic reuse needed for a large option surface.

The 44 completed workers contain 27 timing, 13 cold-memory and four diagnostic
profile reports. Across 5,185 recorded gradient-component comparisons,
including repetitions and both objective execution modes, matched price vectors
are identical and maximum gradient difference is **2.70e-12**. Independent
finite differences are reported separately above. Every report carries source
and native-binary fingerprints; raw results are in
[calibration_phase22_workloads](../benchmarks/calibration_phase22_workloads).

On the actual 81 x 17 grid with fitted parameters, the separate Python/native
profile measures **33.675 s**: **23.930 s** in autograd's native backward engine,
**8.614 s** in native pricing, and **0.017 s** in network-field evaluation.
Graph release/loop overhead accounts for most of the remaining time. This
supports the diagnostic trace's direction on the real calibration workload;
it does not assign all backward time to a particular stencil. The model has
only 37 learned parameters, so larger-network workloads may differ.

### Validation and preservation

After the native build, the complete suite passes **1,756 tests** in **236.99 s**
with 23 existing TorchScript warnings. All previous 1,742 cases remain, plus
14 calibration/model tests. The selected new tests also pass separately.
The validated implementation checkpoint is **`c7d6d38`**.
Price refinement is a separate requirement from a correct discrete VJP;
continuum convergence of every neural parameter gradient is not established
by the finite-difference checks.

The [preservation manifest](../benchmarks/calibration_phase22_preservation.json)
records the implementation checkpoint, environment, native binary and source
hashes, validates the fingerprinted reports, reproduces the independent market
fixture, and regenerates the QuantLib overlay for byte comparison. It verifies
that the Phase 21 artifacts are unchanged, the upstream checkout remains clean
at `06edddbdb85f006c43a449b5fc846a00c67477f3`, and all prior native files other
than the diagnostic additions in `coefficient_pde.cpp` are unchanged.
The [source archive](../benchmarks/calibration_phase22_source.zip) includes the
implementation, tests, benchmark drivers, documentation, generated overlay and
sequential worker scripts. The
[test log](../benchmarks/calibration_phase22_tests.txt),
[summary](../benchmarks/calibration_phase22_summary.json),
[PNG figure](../benchmarks/calibration_phase22.png) and
[PDF figure](../benchmarks/calibration_phase22.pdf) accompany the raw evidence.
The stopped oversized profiling attempt is preserved separately from completed
measurements; no timing was substituted from that attempt.
All **53 fingerprinted reports** match the selected source and native binary;
all **85 regenerated overlay files** match byte-for-byte. The archive contains
**124 source/documentation files** in addition to the overlay and worker scripts.

## Phase 21: joint-state neural diffusions and latent-state pricing

This phase extends the validated Git milestone `4e90e37` without changing
Approach A. The original 1D and basket engines and their optimized kernels are
preserved. [coefficient_pde.cpp](../../cpp/coefficient_pde.cpp) adds a model
coefficient adapter to QuantLib's `FdmLinearOpComposite`; it does not introduce
a time-stepping algorithm or a new pricing engine. The existing
`FdmBackwardSolver`, `Fdm2DimSolver`, and `FdmNdimSolver<3>` perform rollback and
financial interpolation. All native units still use `QL_REAL=TorchReal`.
Stock QuantLib 1.40 is used only in separate validation processes.

### Coefficient and financial contracts

[latent.py](../../python/quantlibtorch/latent.py) exposes three functions:

- `state_grid(axes)` returns `[N,d]` coordinates in QuantLib layout order:
  coordinate zero varies fastest, `N=product(n_i)`. Axes are fixed nonuniform
  sequences passed to QuantLib's `Predefined1dMesher` and `FdmMesherComposite`.
- `diffusion_rollback` accepts drift `[T,N,d]`, diffusion loading `[T,N,d,w]`,
  optional killing `[T,N]`, and terminal state `[N]`. It returns the rolled-back
  state `[N]`. There is no hardcoded spatial dimension in the operator or API;
  current tests cover one through four dimensions.
- `fd_latent` prices vanilla calls/puts for `Z=(log S,Y_1,...,Y_m)`. It accepts
  latent drift `[T,N,m]`, loading `[T,N,1+m,w]`, and a differentiable initial
  coordinate vector `[1+m]`. One and two latent states are supported by the
  existing 2D and 3D financial solver adapters. Initial coordinates must be
  strictly inside all fixed axes; QuantLib's 3D multicubic interpolation rejects
  its upper endpoint. Use `torch.log(spot)` when constructing the initial vector
  to retain the chain rule to an active spot tensor.

Noise loadings can be rectangular or rank deficient. The native adapter forms
`C=L L^T` with a tensor operation, retaining the graph to every loading. It
constructs `d` directional maps and `d(d-1)/2` original
`SecondOrderMixedDerivativeOp` stencils. The generator is

\[
  \sum_i b_i D_i + \tfrac12\sum_i C_{ii}D_{ii}
       + \sum_{i<j}C_{ij}D_{ij} - kI.
\]

In particular, the mixed coefficient is `C_ij`, without an extra factor of
one half. Killing is split equally across directional maps. There is no
production global matrix, coefficient Jacobian, or solution Jacobian.
Full-grid tensor evaluation avoids per-node calls into a Python network.
Coefficients are supplied directly on mesh nodes; no spatial coefficient
interpolation is hidden in this interface.

One time row denotes a time-independent field. Multiple rows have fixed time
coordinates covering `[0,horizon]`. At each original QuantLib `setTime(t1,t2)`,
the adapter samples **drift, covariance and killing at the interval midpoint**
using linear interpolation. Interpolating covariance, rather than interpolating
and squaring the loading, both specifies the exact discrete model and preserves
PSD between samples. Sampling a continuous neural model and increasing its
time resolution is a separate approximation from refining the rollback steps.
Neither axes nor sample/rollback times are differentiable in this milestone.

For equity pricing the first loading row contains relative stock diffusion
exposures; the remaining rows contain absolute latent exposures. The adapter
imposes `b_logS=r-q-C_00/2` and `k=r`. It never accepts a learned stock drift.
Rates are QuantLib interval forward rates from flat or dated log-linear
discount curves using Actual/365 Fixed. European, American and Bermudan
exercise use `vanillaComposite`; `FdmLogInnerValue` supplies payoff averaging.
The existing solvers retain their theta snapshot and interpolation. Every
financial object has an explicit reference date; the new path does not change
the global evaluation date.

The financial stock endpoints follow `FdmHestonEquityPart`: remove the Ito
variance correction from the log-stock drift at the two outer stock nodes.
This accompanies QuantLib's zero second-derivative edge rows and preserves its
linear-in-spot endpoint convention. The general PDE endpoint does **not** make
this financial modification. Both paths retain QuantLib's one-sided first
derivatives and original mixed stencils at edges and corners. They do not infer
reflecting, absorbing, inflow or degenerate boundary conditions for arbitrary
latent processes. In the Heston checks, the variance grid includes zero and
the supplied diffusion vanishes there, matching the original Heston operator.

### Differentiation and independent validation

Full mode records the original numerical arithmetic. Adjoint mode reuses
Phase 20's directional equation VJPs: transpose solves and local coefficient
contractions compose through the unchanged ADI stages. It does not
differentiate elimination iterations or construct large solution Jacobians.
Operator application, coefficient sampling, covariance formation, exercise
and interpolation remain graph connected. The equation adjoint is explicitly
first order; higher derivatives require full mode. Saved states still grow
with the rollback trajectory; this is not checkpointed trajectory reversal.

[test_latent_pde.py](../../tests/test_latent_pde.py) uses independent polynomial
moment stencils and dense global stage equations on nonuniform 1D/2D/3D/4D
meshes. Douglas, Hundsdorfer, modified Craig-Sneyd and Craig-Sneyd cover
noncommuting, time-varying coefficients and arbitrary signed output seeds.
Every drift, loading, killing and terminal-state VJP is checked. Exact
manufactured solutions based on all products `x_i*x_j` check every mixed pair,
including corners and time-varying rank-one PSD covariance. Additional checks
cover strided inputs, no mutation, scalar execution, finite differences,
full-mode curvature, derivative restrictions and invalid contracts.

[test_fd_latent.py](../../tests/test_fd_latent.py) compares the original stock
`FdmHestonOp` with this general adapter on identical fixed meshes, preserving
Heston's endpoint closure. It tests all four schemes, dated curves and all
three exercise types. Central stock-double bumps cover `kappa`, long-run
variance, volatility of variance, correlation, initial log-spot and initial
variance. Joint-state/time networks in 2D and 3D have independently checked
parameter-direction and initial-state sensitivities. The stock oracle is
[stock_latent_reference.py](../../tests/stock_latent_reference.py).

The final native build passes **1,742 tests**, including all original 1,640
and **102 new checks** (73 PDE and 29 financial). The only warnings in the
Python suite are the existing PyTorch JIT deprecations. The targeted suites
also pass. CPU measurements and retained application evidence are recorded
separately from these numerical assertions.

### Recorded numerical and CPU evidence

The implementation checkpoint is `2cd9e5f`. The
[selected evidence](../benchmarks/neural_sde_phase21.json) includes source and
binary SHA-256 hashes, the clean QuantLib pin, all raw samples and runtime
settings. The [preservation manifest](../benchmarks/neural_sde_phase21_preservation.json)
records the source archive, regenerated overlays and retained test/build logs.
The [complete test log](../benchmarks/neural_sde_phase21_tests.txt) records the
1,742-test result. The archive verifies 109 source/document files and 85
reproducibly regenerated overlay files. All 37 pre-existing C++ source/header
files and the QuantLib translation-unit manifest are unchanged from Phase 20.

Heston parameters are `kappa=1.5, theta=.04, sigma=.3, rho=-.6, v0=.045`, with
`S=K=100, r=.03, q=.01, T=1`. The log-stock domain is `[log(20),log(400)]`
and the variance domain `[0,.5]` uses squared-uniform variance coordinates.
The analytic stock-QuantLib price is **8.79471410825148**.

| Grid / steps | General coefficient price | Absolute analytic error | Difference from original QL FD |
|---|---:|---:|---:|
| 21 x 13 / 15 | 8.81459487060604 | 1.9881e-2 | 7.11e-15 |
| 41 x 25 / 30 | 8.79435582268665 | 3.5829e-4 | 1.24e-14 |
| 81 x 49 / 60 | 8.79464497664142 | 6.9132e-5 | 1.07e-14 |

The smaller fixed-mesh Heston VJP record agrees with six stock-double central
bumps within **2.64e-7 absolute**. These differences include bump error; the
unit tests independently cover all schemes, curves and exercise variants.
Refinement here establishes one application case, not universal accuracy of
arbitrary latent diffusions.

Five Adam updates in the synthetic neural example reduce the two-price mean
squared error from **0.964399 to 9.79590e-5** for one latent factor, and from
**0.771033 to 0.0168119** for two. Full/adjoint values and learning histories
agree to floating-point tolerance. Initial drift/diffusion output-gradient
norms are **0.03324 / 5.00107** (one factor) and **0.02066 / 3.94417** (two).
Initial latent-state loss gradients are **0.95401** and
**[-0.57567, 0.10428]**, respectively. Initial states are measured inputs, not
updated by this example's optimizer.

CPU timing uses sequential fresh processes, one intra/inter-op thread,
Windows HighQoS, affinity to logical CPU 8, two warmups and five samples.
Each total includes the network, one option price, squared loss and backward.
Cold process-memory and graph-hook measurements run in separate workers.
The selected run starts after the complete test suite finishes. An earlier
[pilot](../benchmarks/neural_sde_phase21_pilot.json) overlapped test execution and
is retained with its [driver](../benchmarks/neural_sde_phase21_pilot_source.zip);
its timings are not the selected measurements.

| Latent factors / mesh / steps | Full total ms | Adjoint total ms | Full / adjoint cold peak growth MB |
|---|---:|---:|---:|
| 1 / 9 x 5 / 4 | 152.40 | 44.76 | 50.3 / 26.3 |
| 2 / 9 x 5 x 5 / 4 | 1488.02 | 537.76 | 286.5 / 104.7 |
| 1 / 21 x 13 / 8 | 1659.19 | 418.21 | 344.6 / 80.0 |

Observed full/adjoint total ratios are **3.40x, 2.77x and 3.97x**. These compare
the two differentiation strategies for the same new workload, not against
Phase 20's different basket application. The three selected benchmark prices
are identical across modes; their 232 parameter/initial-state gradients differ
by at most **2.67e-15**. The pilot has the same validation differences.
Peak growth is the difference in
process high-water marks in decimal MB, including graph/runtime overhead.
At 21 x 13, full/adjoint graph nodes are **223,198 / 46,970** and saved-tensor
events **118,931 / 1,623**. Counted saved bytes are only **3.48 / 3.15 MB**;
they do not describe total memory consumption. The remaining scalar operator
and spline graph is still material, especially in 3D.

Reproduce using the pinned environment:

```text
.venv/Scripts/python.exe scripts/build.py --jobs 2
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe examples/neural_latent_sde.py --mode adjoint
.venv/Scripts/python.exe examples/neural_latent_sde.py --mode full --latent-dimensions 2
.venv/Scripts/python.exe scripts/validate_neural_sde.py --windows-qos high --affinity 8
```

Omit the Windows QoS/affinity options on other systems. The library itself
does not change process affinity, thread count or power policy.

### Neural model scope and extension limits

[neural_latent_sde.py](../../examples/neural_latent_sde.py) trains bounded smooth
joint-state/time networks against two synthetic option prices, for one or two
latent factors. It reports separate drift/diffusion output gradients and the
initial-state gradient on a fixed mesh. This is an end-to-end learning example,
not a demonstration of parameter identifiability or market calibration.

For the continuous neural model, bounded stock loadings imply a deterministic
bound on the finite-horizon integrated stock variance, hence the Novikov
exponential-integrability condition. Together with nonexplosive, sufficiently
regular state dynamics, this gives the true martingale property of
`exp(-integral(r-q)dt)*S`. The example's finite-weight tanh networks give
globally Lipschitz bounded coefficients in log-stock/latent coordinates.
Arbitrary mesh arrays only guarantee pointwise PSD; they do not establish
well-posedness, positivity/invariance of a latent domain, or martingality of an
unspecified continuous model. Factor rotations also make unrestricted loading
parameterizations nonidentifiable without additional modeling restrictions.

The remaining extension constraints are concrete:

1. **Numerical stability and boundaries.** PSD covariance does not make a
   centered discretization monotone or every ADI parameter stable for arbitrary
   variable coefficients and dimensions. The original QuantLib scheme
   parameters are retained. Pure diffusion Fourier stability results depend on
   dimension, mixed terms and the ADI parameter; they are not a theorem for
   every learned finite-domain model. See
   [in 't Hout and Mishra, stability analysis](https://arxiv.org/abs/1205.1163).
   Domain enlargement and mesh/time refinement remain essential. General
   latent-boundary contracts and stiff/degenerate model tests are next work.
2. **Financial interpolation.** `FdmNdimSolver<N>` is a compile-time template;
   exposing higher-dimensional financial interpolation requires additional
   explicit instantiations and tests, not new ADI algorithms. The operator,
   mesh layout and directional adjoints already accept more dimensions.
3. **Memory and dimension.** Grid nodes grow as `product(n_i)`. Input loadings
   occupy `T*N*d*w` doubles; covariance occupies `T*N*d*d`, and mixed maps scale
   as `9*N*d(d-1)/2`. Full and local-adjoint trajectories add time-dependent
   storage. Dense tensor-product PDEs remain practical only at low dimension.
4. **Performance.** Covariance and sampling are tensor operations, directional
   assembly uses existing vector kernels, and indexed mixed application uses
   Phase 20's tensor path. Original `NinePointLinearOp::mult` and 3D spline
   construction still materialize scalar entries. Their tensor ownership and
   coefficient reuse are candidates for measured optimization. Batching needs
   explicit leading-dimension contracts, mesh/event compatibility and solver
   dispatch; a CUDA tensor at the current scalar bridge is insufficient.
5. **Solver scope.** No Krylov/damping equation adjoint, higher-order equation
   adjoint, trajectory checkpointing, custom boundary callback or cash-dividend
   interface is added. The APIs are CPU float64 and unbatched. No GPU pricing
   or universal `torch.func`/`torch.compile` claim is made.

The only additional upstream compatibility overlay comes from including
`fdmndimsolver.hpp` in the existing generator's header closure: its `std::min`
call needs the same ADL treatment as the other scalar-compatible solvers.
This is recorded in `math-adl.patch`, separate from numerical hooks. The pinned
QuantLib checkout and previous structured-adjoint implementation are unchanged.

## Phase 20: structured adjoints and a real multidimensional QuantLib engine

Following approval of the architectural review, this milestone extends Approach A
to QuantLib's actual `Fd2dBlackScholesVanillaEngine`. It preserves the optimized
1D engine and adds two-asset local-volatility basket pricing, mixed derivatives,
and four existing ADI schemes. The pre-implementation worktree was clean after
commit `5b0de25e50246abdd3cd939fca1ae7a6678589d9`, which preserves the complete
Phase 19 implementation, tests, benchmark history, snapshots, and review.

### Numerical contracts and integration

[structured_ops.hpp](../../cpp/include/quantlibtorch/structured_ops.hpp) and
[structured_ops.cpp](../../cpp/structured_ops.cpp) provide two concrete contracts:

- **Directional solution equation.** QuantLib's `TripleBandLinearOp` determines
  a permutation \(P\) from the mesh layout and direction. The adapter assembles
  the compact bands of \(\widetilde A=a(PLP^\top)+bI\) in elimination order
  outside its custom autograd forward. Forward calls the original pinned
  QuantLib recurrence through a passive operator copy. Backward solves
  \(\widetilde A^\top\widetilde\lambda=P\bar x\), contracts the three band
  gradients, and returns \(P^\top\widetilde\lambda\) to the RHS. Original bands
  and both scaling coefficients retain their complete graph dependencies.
- **Indexed stencil.** An operator is represented by coefficient and neighbor
  tensors of shape `[K,N]`, with
  \(y_i=\sum_k c_{ki}u_{j_{ki}}\). Application, transpose accumulation, and
  explicit coefficient/state pullbacks use functional tensor operations.
  Repeated neighbor destinations accumulate, including folded boundaries.
  The NinePoint adapter preserves QuantLib's summation order. This elementary
  contract supports higher derivatives; the directional equation adjoint is
  explicitly first-order reverse only.

The directional equation above denotes the matrix of the **actual splitting
recurrence**, which differs from reflected operator application at unused
endpoints. QuantLib sweeps the entire permuted array without resetting at every
line. Independent lines arise from zero line-boundary bands; only the global
first lower and last upper coefficients are unconditionally ignored. The new
adjoint preserves even arbitrary interline coupling in low-level probes instead
of silently masking it. Tests include nonsymmetric 2D and 3D layouts.
No global PDE matrix or solution Jacobian is assembled in production.

The [overlay generator](../../scripts/prepare_quantlib.py) records these extensions
separately in `structured-operators.patch`. The original 1D adjoint dispatch
and optimized step paths remain. Multidimensional solves now select the
directional adapter; NinePoint application can select the indexed stencil.
Scalar application remains an independent execution path. Source compatibility,
profiling, and numerical hooks stay in distinct generated patches.

The [source closure](../../cmake/quantlib_sources.cmake) adds six pinned QuantLib
translation units: the 2D Black-Scholes operator, nine-point and mixed-derivative
operators, two 2D solver units, and the basket engine. Every unit uses
`QL_REAL=TorchReal`; no stock QuantLib library is linked. Upstream meshes,
payoff averaging, exercise, event stopping times, ADI stages and bicubic
interpolation execute in QuantLib. The stock double build is used only in
isolated validation processes.

### Public financial API and supported sensitivities

[fd_basket](../../python/quantlibtorch/fd_basket.py) takes two original local-volatility
tensors and a scalar correlation tensor. Each volatility is either scalar or
a time/log-spot surface for its own asset, with the existing interpolation and
flat-extrapolation convention. Independent fixed Black volatilities control
the two meshes; they do not replace the learned local coefficients.

The API exposes min/max basket calls and puts, European/American/Bermudan
exercise, a shared risk-free curve, and separate dividend-yield curves.
Dates and log-linear discount curves retain the existing Actual/365 Fixed
conventions. Correlation is strictly inside \((-1,1)\). Market inputs, axes,
mesh settings and dates remain fixed; only the two surfaces and correlation
are differentiable. The output is scalar NPV.

Douglas, Craig-Sneyd, modified Craig-Sneyd and Hundsdorfer run their original
QuantLib stage compositions in `full` and `adjoint` modes. No new monolithic
scheme backward was needed. In particular, Hundsdorfer's correction sweep
depends on the first sweep's state, whereas Craig-Sneyd/MCS use the original
state. Local solve VJPs compose these dependencies correctly.

The 2D operator's mixed part includes the original discount correction:
both directional components include \(-rI\), so QuantLib adds \(+rI\) alongside
the correlation stencil. This implementation retains that behavior. The engine
also retains its theta snapshot and extra Greek calculations even when the
bridge returns only NPV; their cost is included in measurements.

### Independent validation

[test_structured_ops.py](../../tests/test_structured_ops.py) checks every band,
RHS and scale VJP against small dense equations in the original layout.
Cases include all axes of rectangular 2D and 3D meshes, nonzero interline
bands, zero scales, strided/negative views, operator lifetime, normalized
primal/adjoint residuals, pivot failures and explicit derivative restrictions.
Stencil tests check transpose identities, duplicate destinations, coefficient
pullbacks, higher derivatives and forward AD.

[pde_probe.cpp](../../cpp/pde_probe.cpp) is a validation adapter built from actual
QuantLib first/second/mixed derivative operators and `FdmBackwardSolver`. It
accepts six spatial coefficient fields
\[
L=b_xD_x+b_yD_y+a_xD_{xx}+a_yD_{yy}+cD_{xy}-kI
\]
and a terminal state. It is not a replacement time-stepping solver.
[test_pde_adjoint.py](../../tests/test_pde_adjoint.py) constructs independent
derivative weights by polynomial moment fitting, forms small dense reference
operators, and solves the mathematical ADI stage equations. Nonuniform meshes,
noncommuting directional operators, variable mixed coefficients, signed seeds,
all coefficient/state VJPs, full HVPs and neural-parameter differences are
covered. The manufactured case \(u(x,y,\tau)=xy+\tau c\) supplies an exact
mixed-derivative value and sensitivity check including edges and corners.

[test_fd_basket.py](../../tests/test_fd_basket.py) compares financial values and
both volatility/correlation gradients with an isolated, pinned stock QuantLib
1.40 engine. Its [oracle](../../tests/stock_basket_reference.py) keeps mesh
volatilities fixed during local-volatility bumps. Other cases check spatial
surface and network parameter differences, exercise/curves, input ownership,
evaluation-date restoration, and full-mode HVPs. Independent review checked the
operator/permutation contracts and all four dense stage formulas against the
pinned source.

The final complete suite passes **1,640 tests**, including **168 new checks**:
60 structured-operator, 66 manufactured PDE, and 42 financial basket cases.
The initial 166-check targeted run and 1,638-test full run also passed; two
residual/dual-seed tests were subsequently added and validated.
The registered [CTest Python-suite target](../benchmarks/pde_phase20_ctest.txt)
also passes (1/1).

The [numerical evidence](../benchmarks/pde_phase20_numerical_validation.json)
retains checked values/VJPs, references, tolerances, source/binary hashes and
the clean QuantLib pin:

| Independent reference | Cases | Maximum value/state difference | Maximum VJP difference |
|---|---:|---:|---:|
| Permuted directional dense equations | 28 | 6.67e-16 | 2.23e-15 |
| Dense ADI stages, all four schemes | 32 | 8.89e-16 | 1.12e-15 |
| Manufactured mixed-derivative evolution | 8 | 1.12e-16 | 1.69e-16 |
| Stock QuantLib basket / central coefficient bumps | 8 | 0 | 5.05e-9 |

Maximum normalized primal/adjoint directional residuals are
**4.59e-17 / 3.51e-17**. Stock finite differences include truncation and
roundoff error, so agreement need not reach exact dense-VJP precision.

### Performance, memory and preservation

[The benchmark driver](../../scripts/benchmark_pde_milestone.py) uses sequential,
fresh workers with one intra/inter-op thread, Windows HighQoS and logical CPU
8. A shared PyTorch network produces both volatility surfaces and correlation,
followed by a squared basket-pricing loss. Totals include network evaluation,
pricing, backward and cleanup. Timing, graph hooks, profiling and cold process
memory run separately. The library does not change affinity or QoS.

| Hundsdorfer workload | Full total ms | Adjoint total ms | Full / adjoint graph nodes | Full / adjoint cold peak growth MB |
|---|---:|---:|---:|---:|
| 3 steps, 7 × 5 nodes | 122.35 | 55.00 | 15,649 / 5,712 | 41.8 / 27.4 |
| 6 steps, 15 × 11 nodes | 957.81 | 366.08 | 122,178 / 39,538 | 201.9 / 78.6 |
| 10 steps, 31 × 21 nodes | 6,425.29 | 2,253.54 | 751,046 / 236,264 | 1,153.3 / 359.9 |

The [small/medium report](../benchmarks/pde_phase20_basket_pilot.json) uses three
warmups and seven samples; the
[larger report](../benchmarks/pde_phase20_basket_scaling.json) uses two warmups
and five samples. Observed total ratios are **2.22x, 2.62x and 2.85x** for
these workloads. Memory uses decimal MB and subtracts the pre-price process
high-water mark; it includes allocator, graph metadata and other native/runtime
effects. It is not tensor-only memory.

At 31 × 21, saved-tensor hooks count **476,586 / 132,990 events** and
**11.34 / 10.30 MB** including repeated saves. Tensor-byte counts alone
therefore do not describe this scalar integration's memory cost. Even the
adjoint's 236,264-node graph and roughly 360 MB process growth leave substantial
scope for improved coefficient representation and trajectory management.

The [7 × 5 profile](../benchmarks/pde_phase20_basket_profile.json) records
**108,476 / 50,276 ATen calls** for full/adjoint execution. Adjoint execution
still has 280 scalar local-volatility calls, 1,289 array-pack calls and 48
spline coefficient operations. Its 16 directional adjoint backwards occupy
about **1.75 ms inclusive** within an instrumented **65.42 ms backward**.
These nested profiler times are not uninstrumented timing budgets. They
indicate substantial cost outside the directional solve boundary, motivating
measurement of scalar assembly, graph bookkeeping and interpolation before
further solver optimization.

The [final 1D controls](../benchmarks/pde_phase20_vanilla_controls_final.json)
load a frozen Phase 19 package/binary and the current build in separate workers,
with 32 warmups and 15 samples. All **282 matched baseline/current samples
have identical prices and gradients**. At 30/61, surface/neural full totals
are **6.62/7.16 → 6.98/7.25 ms**, adjoint **5.98/6.11 → 6.10/6.05 ms**, and
inference **3.70/3.58 → 3.81/3.83 ms**. An
[earlier control run](../benchmarks/pde_phase20_vanilla_controls.json) has small
mixed changes too. Its metadata mistakenly labeled the unchanged Douglas
control as Hundsdorfer; the final driver/report corrects the label. The
[original script](../benchmarks/pde_phase20_pilot_benchmark_source.zip) and
observations remain preserved. That correction changed no numerical or timing
operation. These controls support retention of the existing performance range
with ordinary host variability.

The [preservation manifest](../benchmarks/pde_phase20_preservation.json) verifies
the [source archive](../benchmarks/pde_phase20_source.zip), numerical reports,
benchmark workers, test logs and reproducibly regenerated overlay.
The archive verifies 96 source files and 84 regenerated overlay files; its
SHA-256 is `5f582f7508e5852f6d0cb7d3c2b1f09d03e6e974057298db20fe06168a06f20f`.
**33 existing numerical kernel/header files remain byte-identical** to
Phase 19; the bridge retains its 1D pricing body and adds the basket endpoint.
Combined benchmark validation covers **632 price comparisons** and
**444 gradient comparisons**, with maximum differences
**3.56e-15 / 1.78e-14** across modes. No benchmark establishes batched or GPU
financial pricing; those require further ownership, layout and execution work.

### Limits and next steps

This milestone establishes additional real QuantLib numerical/financial
configurations with correct tested VJPs. It is not a universal residual solver
interface. The current directional adapter uses QuantLib's no-pivot recurrence
and compact transpose solver, retaining their pivot domains; it does not add
GMRES/BiCGstab, sparse direct solvers, or nonlinear equation adjoints.

Multidimensional damping is explicitly rejected: QuantLib's implicit-Euler
damping enters a Krylov solve outside this equation-adjoint contract. Custom
boundary callbacks, active meshes/times, cash dividends in the basket adapter,
and higher-order equation adjoints are also outside this milestone. Exercise
and interpolation sensitivities retain the branch qualifications in the review.

The manufactured six-field adapter is time-independent and internal. The
financial adapter supports time-dependent surfaces for each asset separately;
it does not yet expose arbitrary jointly state-dependent multi-factor drift
and covariance networks. Both pricing APIs remain CPU float64 and unbatched.
Indexed stencil operations are a useful tensor boundary, but their existence
does not establish complete-engine GPU efficiency.

Follow-on work should use the measured new bottlenecks and application needs:
coefficient ownership/materialization across 2D assembly, trajectory storage
and checkpointing, broader coefficient adapters, or a validated iterative
equation/transpose-solve boundary. Preserve the working scheme compositions
and specialized backends as these contracts expand.

## Architectural review: general differentiable PDE solvers (2026-11-01)

**Historical review scope:** preserve the validated implementation and assess numerical
extensibility before further optimization. This review supersedes the prospective
optimization targets in the historical phase reports below. Implementation was
paused pending feedback during the review. The user subsequently approved the
Phase 20 implementation documented above.

**Assessment:** the existing first-order `adjoint` mode already supplies the
equation-based discrete sensitivities required for the supported local-volatility
inverse problem. Local adjoints compose through the entire rollback without a
global solution Jacobian. The main gap is the breadth of the numerical interfaces:
the public pricing bridge, compiled engine closure, directional solve hooks, and
fast step/assembly contracts are specialized to a one-dimensional vanilla engine.
Retain the current kernels and QuantLib integration, and develop explicit
operator/solve/stage differentiation contracts around them after review.

The evidence is the actual source at the preserved Phase 19 revision, its existing
numerical tests, and archived measurements. The native build and validation already
in progress when priorities changed have completed: **1,472 tests passed**, plus
**319 feature-disabled checks**. The [source archive](../benchmarks/fd_phase19_source.zip)
has SHA-256
`7e5106894378c1d34c6e8807b81b483ce18fd3156dc72ecb3682f9d76ed17ed7`.
The [validation manifest](../benchmarks/fd_phase19_validation.json) verifies 86 source
files, the native binary, and 78 reproducibly regenerated overlay files against
clean QuantLib commit `06edddbdb85f006c43a449b5fc846a00c67477f3` (1.40).
All 564 matched ordinary price/VJP samples and 1,136 matched curvature samples
(including eight profiled results) are identical across the latest flag change.
Historical archives remain preserved. These checks establish the tested scope;
they do not establish whole-library differentiability.

### 1. What the current modes differentiate

Both modes differentiate a **discretized computation**, rather than a continuous
PDE adjoint. They differ in the treatment of numerical solution operations.

| Boundary | `adjoint` behavior | `full` behavior |
|---|---|---|
| Tridiagonal and splitting solves | Implicit VJP of the linear equation; elimination runs without an autograd tape | AD of the executed elimination arithmetic, or its fused custom reverse recurrence |
| Supported Douglas / implicit / explicit steps | One local VJP of the implemented step equation | Original composition or optimized composition of differentiable tensor operations |
| Local-volatility sampling and operator assembly | Graph-connected interpolation, squaring, and analytic assembly VJPs | Same dependencies; fused elementary operations support higher derivatives |
| Operator application | Explicit linear-map VJP, including coefficient derivatives and transpose scatter | Elementary arithmetic AD / fused equivalent |
| Exercise, dividend jumps, final interpolation | Remaining autograd composition and executed branches; spline linear solves can use an implicit VJP | Remaining autograd composition and executed branches |

In [SplitAdjoint and TridiagonalAdjoint](../../cpp/adjoint.cpp), the forward saves
compact bands and the solution \(x\). For \(Ax=b\), backward solves
\[
A^\top\lambda=\bar x,\qquad
\bar b=\lambda,\quad
\bar d_i=-\lambda_i x_i,\quad
\bar l_i=-\lambda_{i+1}x_i,\quad
\bar r_i=-\lambda_i x_{i+1}.
\]
There is no reverse traversal of Thomas elimination inside these boundaries.
`adjoint_split` constructs \(A=aL+bI\) from the original tensors **outside** the
custom forward, so the returned band gradients propagate to \(L,a,b\).
The standalone [Python solve adjoint](../../python/quantlibtorch/solver.py) follows
the same construction.

[StepAdjoint / AssembledStepAdjoint](../../cpp/step_adjoint.cpp) go one level higher.
Their local residual is
\[
R(x,u,c)=(I-\beta L_{\rm split}(c))x
              -(I+\alpha L_{\rm apply}(c))u=0.
\]
For Douglas, \(\beta=\theta\Delta t\) and
\(\alpha=(1-\theta)\Delta t\); implicit Euler has \(\alpha=0\);
the internal explicit step has \(\beta=0,\alpha=\theta\Delta t\).
Backward solves the transpose of
\(I-\beta L_{\rm split}\), returns
\((I+\alpha L_{\rm apply})^\top\lambda\) to the old state, and contracts
coefficient perturbations against the old and new states. The distinction
between the two operators is intentional: application reflects endpoint
neighbors, whereas splitting ignores the two unused outer bands. Independent
dense-reference tests include nonzero outer bands.

Crank-Nicolson uses QuantLib's composition of explicit and implicit stages.
An explicit stage needs a direct VJP, not an implicit solve. Sampling,
assembly, interpolation, and events still require their own chain rules;
an implicit solve alone cannot recover their dependencies.

In contrast, [ThomasArithmetic](../../cpp/thomas_arithmetic.cpp) in `full` mode
returns and saves pivots, elimination factors, transformed right-hand sides,
and solutions. Its backward reverses those recurrences, including gradients
of intermediate outputs needed by higher derivatives.
[PCR stages](../../cpp/pcr_stage.cpp) similarly differentiate elimination stages.
Fusing these operations avoids recording every scalar forward operation, but
their derivative semantics remain algorithmic. For an admissible direct solve,
algorithmic AD and equation differentiation agree in exact arithmetic.
Floating-point order, pivot guards, and selected branches still matter.
`TorchReal` comparisons inspect primal values: full AD differentiates the
executed branch, not changes in branch decisions or integer iteration counts.

The API phrase “not a full-trajectory adjoint” describes the absence of a single
custom operation owning the entire trajectory. **It does not mean that
sensitivities stop at a time-step boundary.** Stack all actual stage residuals
and events in rollback execution order. Their Jacobian is block triangular.
Autograd's reverse composition of local adjoints performs the corresponding
block transpose substitution, accumulating every path to shared coefficients.
It therefore gives the requested global discrete VJP on the supported smooth
branch, without assembling that global matrix. This is the same structural
relationship illustrated in the
[dolfin-adjoint derivation](https://www.dolfin-adjoint.org/en/latest/documentation/maths/4-adjoint.html).
Direct coefficient terms in a loss or regularizer remain ordinary PyTorch paths.

### 2. VJP work and saved state

No production pricing path constructs \(D_\sigma u_h\), a dense space-time
Jacobian, or a matrix inverse. The conceptual \(-\lambda x^\top\) matrix gradient is
evaluated only at the stored band entries. Dense matrices occur in small
independent tests, not in the pricing VJP.

| Operation | Saved data used by backward |
|---|---|
| Split / tridiagonal adjoint | Three compact bands and the solution |
| Fused step adjoint | Original three-row band tensor (or three rows), old state, new state; passive time-step settings |
| Diffusion assembly | Original variance, first/second spatial derivative bases, rate and yield |
| Band application | Original bands, input state, and neighbor indices |
| Sampling / events / interpolation | Original coefficient rows and graph dependencies, interpolation data and selected-branch dependencies; normal autograd saves |
| Full Thomas | Original bands/RHS plus pivots, factors, transformed RHS and solution |
| Full PCR | Bands/RHS at each elimination stage, with additional derivative graphs when requested |

For \(S\) executed stages and \(N\) spatial nodes, current 1D adjoint solve/step
storage and arithmetic scale as \(O(SN)\), plus the coefficient/network graph,
sampling caches, and other pricing operations. Damping and events affect \(S\).
This is not an \(O(SNP)\) solution Jacobian for \(P\) coefficients.
Storage still grows with the trajectory: there is no general trajectory
checkpoint/recompute policy. Transpose solves currently recompute their
elimination rather than using a general reusable factorization interface.
The full PCR stage history has an additional logarithmic number of elimination
levels; higher derivatives build further graphs.

Measured 30/61 surface graph counts after Phase 18 are **288 nodes in adjoint**
and **389 in full**; saved-tensor hooks count **427 / 728 saves** and
**258,366 / 402,774 bytes**, respectively. These are repeated-save counts,
not unique allocations or peak live memory. Phase 19 removes temporary,
unrequested full-mode assembly VJPs: it reduces gradient-construction calls by
660, but does not shrink the retained gradient graph. This supports keeping
efficient local boundaries while addressing trajectory memory separately.

### 3. Dependence on particular numerical methods

The **mathematics** requires a valid discrete linearization, its transpose
action/solve, and coefficient pullbacks. It does not require Thomas, PCR,
Douglas, or Black-Scholes coefficients. A different solver for the same
well-solved equation can use the same implicit VJP.

The **software** currently embeds several narrower assumptions:

- The public [pricing API](../../python/quantlibtorch/fd.py) and
  [native bridge](../../cpp/fd_bridge.cpp) construct `FdBlackScholesVanillaEngine`
  and `TensorLocalVol`. Only sampled local volatility is a differentiable
  pricing input. Mesh settings, axes, dates, curves, spot, strike, and dividends
  are fixed values. The output is a scalar NPV; a general PDE solution-field
  interface or user-supplied operator is not exposed.
- `try_step_adjoint` requires the exact `FdmBlackScholesOp` type, one dimension,
  direction zero, passive step settings, and an empty boundary-condition set.
  Its full-mode acceleration has similar guards. Subclasses and boundary
  callbacks take the original step path; tests cover selected 1D fallbacks.
- The [generated splitting hook](../../scripts/prepare_quantlib.py) explicitly
  throws `"adjoint currently requires one dimension"` on a multidimensional
  mesher. This is a hard unsupported case, not a generic fallback.
- Assembly uses the Black-Scholes form
  \(\tfrac12vD_{xx}+(r-q-\tfrac12v)D_x-rI\), with private operator caches and
  original-band ownership. Another operator needs an appropriate assembly
  dependency contract; reusing this formula would be incorrect.
- Thomas and PCR have solver-specific pivot domains. `auto` also imposes
  scaling/diagonal-dominance/residual guards. Matrix nonsingularity alone does
  not guarantee acceptance by these implementations or their transpose solves.

The exact-type guards are appropriate for protecting specialized fast paths.
The architectural gap is the lack of a broader validated interface beneath
them. Existing caches, packed arrays, vector kernels, band handoffs, and native
arithmetic are useful optional implementations of such an interface.

### 4. QuantLib methods that can actually be selected

| Method or capability | Current status | Additional work needed |
|---|---|---|
| Douglas, Crank-Nicolson, implicit Euler in `fd_vanilla` | Public and tested in `full` and first-order `adjoint`, for the 1D Black-Scholes/local-volatility operator | Preserve current domain and nonsmoothness qualifications |
| European, American, Bermudan vanilla exercise; dated curves and cash dividends | Public and tested with the above engine; coefficient sensitivities only | Other financial inputs would need tensor APIs and complete dependency coverage |
| Upstream tridiagonal/Thomas solve | Public standalone full/adjoint solve; used by pricing | Alternative matrix structures/backends need explicit interfaces |
| Fused Thomas / tensor PCR | `auto` selects eligible full-mode implementations; explicit pricing PCR is full-only | Equation adjoints are mathematically compatible with PCR, but that combination is not exposed |
| Explicit Euler | Compiled and tested as an internal stage, including its local VJP | Public standalone scheme selection and end-to-end validation |
| Craig-Sneyd, modified Craig-Sneyd, Hundsdorfer, method of lines, TR-BDF2 | Sources or template instantiations are in the vanilla dependency closure; not publicly selectable or validated as pricing configurations here | Exposure, actual stage/control-flow audit, and numerical validation; existing primitive VJPs may be reusable |
| GMRES / BiCGstab | QuantLib implementations are compiled; upstream multidimensional implicit Euler can call them | No public backend selection or equation-level adjoint registration; transpose actions/solves, tolerances and validation are needed |
| Heston / other 2D or 3D engines, mixed-derivative operators | Not in the supported bridge/compiled engine closure; multidimensional splitting adjoint rejects them | Extend source closure, operator/layout and financial adapters, then validate |
| Batched CPU/CUDA tensor primitives | Available separately for supported application/assembly/PCR operations | They do not establish batched or GPU execution of the financial engine |

The [source closure](../../cmake/quantlib_sources.cmake) includes several schemes
because [FdmBackwardSolver](../../third_party/QuantLib/ql/methods/finitedifferences/solvers/fdmbackwardsolver.cpp)
references them. Compilation is not a supported differentiation API. The
Python selector exposes exactly three schemes and no arbitrary
`FdmSchemeDesc`, solver object, or operator. Running an additional 1D scheme
would also not validate its multidimensional splitting or mixed-derivative terms.

The [upstream implicit Euler implementation](../../third_party/QuantLib/ql/methods/finitedifferences/schemes/impliciteulerscheme.cpp)
uses `solve_splitting` when the operator has one direction, and BiCGstab/GMRES
otherwise. There is no generic custom VJP around that Krylov solve. A compatible
`TorchReal` execution in full mode could differentiate its executed iterations,
but that has not been established as a supported multidimensional pricing path.
An implicit derivative of a converged equation is also different from the
derivative of a fixed number of unconverged iterations.

Integration still uses actual QuantLib instruments, processes, curves,
mesh construction, rollback, event ordering and interpolation. The
[build](../../CMakeLists.txt) compiles the selected pinned sources with `TorchReal`
and links LibTorch, not a stock double-based QuantLib library. Stock QuantLib
is an isolated reference process in tests. The current bridge fixes conventions
such as Actual/365 Fixed and its calendar choices; it does not expose the whole
QuantLib object model or accept arbitrary stock Python QuantLib engine objects.
Extending financial coverage should continue through this compatible build.

### 5. Recommended differentiation architecture

**Add structured residual and linearization capabilities, while retaining local
adjoint composition and specialized backends.** A generic function named `F`
does not make an opaque engine differentiable. The present
[FdmLinearOpComposite interface](../../third_party/QuantLib/ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp)
provides application, directional splitting, and preconditioning, but no required
transpose application or coefficient VJP. Even `toMatrixDecomp` defaults to
failure. A design that requires every operator to assemble a matrix would
unnecessarily narrow QuantLib's existing choices.

The following are proposed contracts, not implemented APIs:

| Layer | Required information | Preserved freedom |
|---|---|---|
| Coefficient/operator adapter | Immutable graph-connected coefficient ownership; application and coefficient pullback; transpose action; boundary/source dependencies | Analytic stencil contractions, sparse operations, or AD of a local residual |
| Implicit solve adapter | Primal solve, linearization at its result, transpose solve, residual/termination status, saved context | Thomas, PCR, banded/sparse direct solvers, or matrix-free Krylov methods |
| Scheme/stage adapter | Actual stage equations and dependencies, time evaluations, history, boundary and event ordering | Primitive composition, a fused step VJP, or a validated trajectory adjoint |
| PyTorch boundary | Incoming seed to coefficient/input VJPs; explicit first/higher/forward-AD capability | Upstream neural networks and direct loss/regularizer paths remain PyTorch |

For an implicit block \(R(z,h,c)=0\), an adapter can provide actions
\(R_zv\), \(R_z^\top w\), and
\((R_h^\top w,R_c^\top w)\), plus a solve for \(R_z^\top\lambda=\bar z\).
It returns \(-R_h^\top\lambda\) and \(-R_c^\top\lambda\).
None requires an assembled solution Jacobian. For a linear block these become
the current transpose solve and local stencil contractions. For an explicit
block, a normal VJP is sufficient. The separation of forward solvers from
residual-based differentiation has a precedent in
[Blondel et al., Efficient and Modular Implicit Differentiation](https://arxiv.org/abs/2105.15183).
Using that principle here is a recommendation, not a proposal to adopt another
library or replace QuantLib.

Different schemes must supply different dependency structures:

- A theta step can use a residual involving its actual old/new operators and
  boundary terms. Coefficient evaluation times and endpoint elimination are
  part of that residual.
- Multidimensional Douglas first computes
  \(y_0=u+\Delta t L_{\rm total}u\), then directional stages
  \((I-\theta\Delta t L_i)y_i-y_{i-1}+\theta\Delta t L_i u=0\),
  subject to the actual boundary transformations. Its adjoint reverses these
  stages. Craig-Sneyd/Hundsdorfer corrections and mixed terms need their actual
  dependencies too. Replacing this with one unsplit implicit-Euler or theta
  residual would differentiate a different numerical method.
- Multistep methods need history dependencies; TR-BDF2 needs its stages and
  history; Runge-Kutta methods need their stage couplings. An adaptive method
  also needs an explicit sensitivity policy for accepted time grids, rejection
  and error control. Differentiating with a frozen/replayed accepted grid must
  be stated, not confused with a derivative through discontinuous step choices.
- A nonlinear solve needs the derivative of its actual converged residual,
  including state-dependent coefficients. It must not use merely the last
  frozen-coefficient/Picard matrix unless that is the intended approximation.
- An obstacle/complementarity solve needs a nonsmooth sensitivity contract.
  Treating it as an ordinary nonsingular smooth equation is insufficient.

Not every scheme needs a bespoke monolithic backward. Existing explicit
operations and newly supported directional solves may already compose the
correct adjoint. Fusing a stage or trajectory should remain an optional
optimization supported by equivalence tests. PETSc TSAdjoint demonstrates
scheme-level discrete differentiation, higher-order products, and checkpointing;
this is an architectural precedent, not evidence that every QuantLib scheme is
already covered. [Zhang et al.](https://arxiv.org/abs/1912.07696)

For iterative backends, report primal and adjoint residuals and make tolerances
part of the contract. Root sensitivity is accurate only to the extent justified
by convergence and conditioning. A forward preconditioner is not automatically
a correct transpose preconditioner. Avoid differentiating iterations for an
equation-mode VJP, but preserve an explicitly named algorithmic mode when the
truncated algorithm itself is the intended function.

The current first-order restriction is an **implementation contract**, not an
IFT restriction. Higher-order implicit differentiation is possible through
differentiable adjoint solves and the necessary mixed residual derivatives.
Current `full` mode remains the available tested route for those derivatives.
No immediate change of the public default mode is proposed.

### Nonsmooth sensitivities and mathematical limitations

QuantLib's [American projection](../../third_party/QuantLib/ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.cpp)
replaces continuation only when the intrinsic value is strictly greater.
Bermudan projection uses the same comparison at exercise dates. With the current
fixed payoff, the state derivative is one on the continuation branch and zero
on the exercise branch; equality retains the continuation branch. There need
not be a unique classical derivative at a switching point. This is a piecewise
derivative of the executed discrete rollback/projection schedule, not a smooth
free-boundary derivative or an adjoint of a separately converged complementarity
problem.

Spline limiters and other value-based comparisons also select branches.
Gradient and HVP comparisons apply away from unstable switches; a central
difference crossing a switch need not match the chosen branch derivative.
New complementarity solvers would require an active-set or generalized
derivative policy and appropriate validation. Smoothing changes the numerical
problem and should be an explicit modeling choice.

The fixed mesh and coefficient-sampling coordinates define the current
sensitivity problem. Training mesh locations, event dates, or adaptive
decisions would require additional dependencies and sometimes nonsmooth
analysis. Ill-conditioning, singular linearizations, poor PDE resolution,
and inverse-problem nonidentifiability are mathematical/numerical limitations;
a generic software interface cannot remove them.

### 6. Neural coefficients and multidimensional applications

**Neural local volatility already works on the fixed sampled surface.**
[The neural-parameter test](../../tests/test_fd_engine.py) evaluates a PyTorch
network, maps its output to positive volatility, prices with the real QuantLib
engine, and checks the pricing-loss parameter gradient against an independent
parameter-space directional difference in both modes. The returned coefficient
VJP propagates through interpolation and squaring to that network. A native
neural-network implementation inside QuantLib is not required.

The current model is the **interpolated sampled network**, not direct evaluation
of an arbitrary callable network at every internal PDE point. More capable
inverse problems need a deliberate sampling/quadrature contract, shared
coefficient graphs across instruments, and measured training memory/runtime.
Multi-price losses can already compose scalar calls, but there is no batched
pricing engine; the bridge also serializes its QuantLib evaluation-date state
with a mutex. Mesh/domain refinement, regularization, and parameter
identifiability need validation beyond VJP correctness.

For a two-factor or more general Markov diffusion, the backward pricing
operator contains drift components, covariance terms (including cross
derivatives), discount/killing terms, and possibly sources. Coefficients
parameterized nonlinearly by a network do not by themselves make the pricing
PDE nonlinear in its solution. Required extensions are:

1. Compile and expose the appropriate QuantLib engine, mesher, operator and
   event dependency closure with `TorchReal`. A Heston engine/operator is a
   useful concrete 2D target, but does not by itself implement arbitrary neural
   drift/diffusion fields.
2. Generalize directional line extraction and transpose scatter to the actual
   layout and boundary rules. Add mixed-derivative application/coefficient
   VJPs and validated ADI stage composition. Replace the hard 1D rejection only
   once these paths exist; deleting the guard is not support.
3. Supply graph-connected drift and covariance adapters, with an appropriate
   positive-semidefinite diffusion parameterization and treatment of
   degeneracy/boundaries. Preserve the chosen risk-neutral measure, drift and
   numeraire conventions. Learning physical dynamics and a market price of
   risk is a separate modeling extension.
4. Preserve existing QuantLib financial inputs and event ordering as the
   engine coverage grows. New trainable curve/payoff/boundary inputs require
   explicit derivative paths; primal interoperability alone is insufficient.
5. Measure checkpointing, coefficient reuse, line batching and device placement
   on representative calibration losses. CPU/CUDA tensor primitives do not
   make scalar financial objects or the complete engine GPU-efficient.

Spatial grid size grows multiplicatively with the number of factors. A general
adjoint reduces derivative cost but does not remove that PDE scaling barrier.
If a later application requires Monte Carlo or another numerical family, it
will need an appropriate pathwise/estimator differentiation contract and
separate validation. The proposed operator/solve contracts should not be sold
as automatic support for all Neural SDE pricing methods.

### 7. Engineering judgment, evidence, and proposed next work

**Recent work is specialized, but remains useful.** Native band/step arithmetic,
PCR and Thomas recurrences, spatial caches and original-band handoffs reduce
overhead on a real QuantLib pricing path. Their guards and reference paths
preserve tested behavior. There is no mathematical dead end in composing these
local adjoints. However, continuing to optimize full Thomas higher-derivative
graphs as the main priority would not resolve the user's central gaps:
operator generality, multidimensional schemes, iterative equation adjoints,
financial exposure, and trajectory memory.

Representative evidence:

| Evidence | Observation | What it establishes |
|---|---|---|
| [Step adjoint tests](../../tests/test_step_adjoint.py) | Independent dense equations, every band/state VJP, reflected endpoints; three schemes × three exercise types, coincident events, nonuniform curves, directional differences | Correct local equation adjoints and their tested composition |
| [Band solve tests](../../tests/test_fd_adjoint.py) | Arbitrary nonsymmetric bands and scaling coefficients, dense references, finite differences, pivot failures | The solve rule is not derived only for a particular volatility surface |
| [Engine tests](../../tests/test_fd_engine.py) | Isolated stock QuantLib prices/parallel vegas, surface and neural-parameter differences, refinement and higher-derivative contracts | Financial integration and end-to-end coefficient sensitivity in the exposed scope |
| [Full step tests](../../tests/test_full_tensor_step.py), [PCR tests](../../tests/test_parallel_solve.py) | Original-path comparisons, scheme/event coverage, VJPs and curvature | Retain algorithmic AD as an independent implementation reference |
| Phase 18 [timing](../benchmarks/fd_phase18_selected_timing.json) and [scaling](../benchmarks/fd_phase18_selected_scaling.json) | Selected 30/61 neural adjoint total 7.44 → 6.08 ms; 120/241 surface total essentially unchanged | Gains are workload-specific; local adjoints are already practical |
| [Current Phase 19 controls](../benchmarks/fd_phase19_selected_controls.json) | 30/61 surface full/adjoint 7.09 / 5.72 ms; neural loss 7.46 / 6.19 ms | Equation adjoints already service the relevant first-order training workload |
| Phase 19 curvature profiles and repeats below | 660 fewer gradient-construction tensor calls; mixed selected wall-time changes, including a neural 30/61 regression | Real eliminated work does not justify a universal performance claim |

The full suite passed on the preserved native build. Tests and benchmarks were
not expanded into unsupported multidimensional configurations during this
review. Agreement between two implementations alone is weaker evidence than
independent equations and finite differences; the existing tests provide both
for the supported boundaries. None proves all possible meshes, boundary
conditions, pivot patterns or nonsmooth switches correct.

**Recommended sequence, subject to feedback:**

1. Record a capability contract separating equation versus algorithm
   derivatives, first/higher/forward order, supported operators, boundaries,
   layouts, devices, batching and solver domains. An unsupported path should
   either fail clearly or use a specifically validated fallback.
2. Introduce a small structured linear-solve/linearization adapter around the
   existing implementations. Validate two forward backends for the same
   equations using the same coefficient VJP contract. Preserve the current
   specialized paths and pinned QuantLib provenance.
3. Exercise that contract on a genuinely different operator: a 2D manufactured
   problem with mixed derivatives and the actual ADI stages, followed by a
   real QuantLib multidimensional financial engine. Another 1D Black-Scholes
   microbenchmark would not establish the needed flexibility.
4. Validate transpose identities, primal/adjoint residuals, small dense VJPs,
   parameter-space differences/Taylor tests, boundary/event derivatives,
   solver-tolerance behavior and saved-state ownership. Compare integrated
   financial prices with isolated stock QuantLib.
5. Profile representative multi-contract neural calibration and trajectory
   memory before choosing checkpointing, additional fusion, higher-order
   implicit derivatives, batching or GPU work.

This is an incremental extension of Approach A. A residual interface is expected
to improve extensibility; it is not a measured performance improvement.
Dispatch and representation conversion could add overhead, so it should retain
the existing efficient band operations and allow scheme-specific fusion.
A monolithic global adjoint may later help scheduling or checkpointing, but is
not needed to obtain correct coefficient VJPs now. The review's feedback gate
was subsequently satisfied by the user's explicit approval of Phase 20.

## Phase 19: requested full-mode assembly gradients

`QLT_DIFFUSION_SELECTED_GRADS=1` evaluates only assembly VJP outputs
requested by the current autograd task during full-mode graph-building backward.
The preceding implementation always evaluated all five input gradients. A
financial pricing call usually requires only variance gradients; spatial bases
and rates are passive. `AutogradContext::needs_input_grad` also accounts for a
caller requesting only a subset of otherwise active inputs. The benchmark level
`phase19` selects this default (`diffusion_grads` is an alias); `phase18`
retains all five outputs. Zero restores the preceding behavior.

Every original input remains saved. A requested VJP uses every original operand
on which it depends, even when that operand's own VJP is not requested. This is
necessary for later cross-derivatives. Incoming forward-mode VJP weights retain
their tangents, including calls without `create_graph`. Only unused returned
gradients and their temporary expressions are skipped. The elementary formulas,
native ordinary backward, first-order adjoints, forward arithmetic, numerical
solver selection, and elimination-state graph are unchanged. No coefficient is
detached or modified in place, and no QuantLib overlay change is required.

The opt-in native build passes **1,472 tests**, including **101 new checks**,
plus **319 targeted checks**, **319 feature-disabled checks**, and **203 checks**
with original assembly forward arithmetic selected. New coverage includes all
31 nonempty input-activity subsets through third order, all 31 subsets with
incoming dual weights both with and without graph recording, and a request for
one VJP followed by cross-derivatives to all other active inputs. Finite first
and second differences cover additional partially active combinations. Existing
alias, view, transform, financial, solver-state, and fallback checks remain.

`scripts/benchmark_fd_curvature.py --profile` now records forward,
gradient-construction, and HVP scopes, then separately inspects the retained
graphs and saved-tensor events. Instrumentation runs after all timed samples.
Both the profiler and saved-tensor-hook runs must reproduce the timed price,
gradient, and HVP exactly. The [baseline](../benchmarks/fd_phase19_baseline_curvature.json)
uses the unchanged Phase 18 native sources and binary. Its
[source snapshot](../benchmarks/fd_phase19_baseline_source.zip),
[commands](../benchmarks/fd_phase19_baseline_commands.json), and
[validation](../benchmarks/fd_phase19_baseline_validation.json) record the sole
baseline source change: optional benchmark instrumentation. Baseline archive
SHA-256: `1032892c484a2f883a94054168296e1275b076704463be128ab2686d110a1d1a`.

In the baseline surface 30/61 Thomas workload, assembly graph backward accounts
for **990 of 6,731** gradient-construction ATen calls. The profile measures about
**2.37 ms** inclusive in that scope. Nested profiler scopes and instrumentation
overhead prevent treating this as an additive uninstrumented time budget.

### Pilot measurements and mechanism

Fresh workers use one intra/inter-op thread, Windows HighQoS, logical CPU 8,
32 warmups, and 15 timed samples. All **36** pricing/curvature workers verify
process mask 256 and disabled execution-speed throttling. Complete curvature
totals include construction, forward, gradient recording, HVP, and cleanup;
neural cases also include network evaluation and a squared pricing loss.

| Default-Thomas HVP workload | Steps / nodes | Phase 18 ms | Requested gradients ms |
|---|---:|---:|---:|
| Surface | 10 / 31 | 15.91 | 15.10 |
| Neural loss | 10 / 31 | 19.25 | 15.99 |
| Surface | 30 / 61 | 36.58 | 33.69 |
| Neural loss | 30 / 61 | 37.61 | 33.77 |
| Surface | 60 / 121 | 66.32 | 62.31 |
| Neural loss | 60 / 121 | 70.96 | 61.77 |

The [30/61 off](../benchmarks/fd_phase19_selected_grads_curvature0.json),
[30/61 on](../benchmarks/fd_phase19_selected_grads_curvature1.json),
[scaling off](../benchmarks/fd_phase19_selected_grads_scaling0.json), and
[scaling on](../benchmarks/fd_phase19_selected_grads_scaling1.json) reports
measure **1.05-1.20x** total speedups for default Thomas. At 30/61,
surface/neural gradient-construction medians fall **10.31/10.93 to 9.22/9.07
ms**. PCR totals are mixed: 30/61 surface/neural changes **123.14/124.99 to
121.87/120.87 ms**, while 60/121 changes **247.21/278.80 to 255.49/242.74 ms**.
No uniform PCR speedup is claimed.

Ordinary first-order [controls](../benchmarks/fd_phase19_selected_grads_controls.json)
keep their preceding computational paths. Surface/neural full-mode totals
change **6.92/6.93 to 6.88/6.80 ms**, adjoint **5.72/5.96 to 5.88/6.00 ms**,
and inference **3.64/3.68 to 3.92/3.71 ms**. Their variation limits causal
interpretation of small timing differences. Affinity and HighQoS do not fix
host frequency or competing load; the library does not set either policy.

The 30/61 profiles remove exactly **660 gradient-construction ATen calls**:
surface/neural Thomas falls **6,731/6,802 to 6,071/6,142**, and PCR
**40,685/40,756 to 40,025/40,096**. Assembly's contribution falls **990 to
330 calls**. Forward and HVP operation counts are unchanged. The retained
gradient graphs remain **3,430/3,531 nodes** for Thomas and **14,261/14,362**
for PCR; the removed work built temporary gradients that autograd discarded.
For Thomas, gradient-construction saved-tensor events fall **2,502/2,526 to
2,370/2,394**, and counted bytes **1,322,675/1,327,002 to 1,193,843/1,198,170**.
PCR likewise removes 132 events and 128,832 counted bytes. These hook counts
include repeated and transient saves; they do not establish reduced peak live
memory. Forward saved-tensor counts and bytes are unchanged.

The pilot [source archive](../benchmarks/fd_phase19_selected_grads_source.zip),
[commands](../benchmarks/fd_phase19_selected_grads_commands.json), and
[validation](../benchmarks/fd_phase19_selected_grads_validation.json) verify
**86 source files** and **78 regenerated overlay files**. All **1,136** matched
curvature samples, including eight profiled numerical results, have identical
prices, gradients, and HVPs. All **564** matched ordinary price/VJP samples
are also identical. Combined historical/current curvature validation covers
**1,692 timed/warmup samples**, with maximum HVP difference **2.56e-13** across
solver configurations. Ordinary fixed-reference validation covers **1,720**
comparisons, with maximum price difference **3.56e-15** and VJP difference
**1.78e-14**. Pilot archive SHA-256:
`89d7cf3fe8eacdda425224f3ceac6077c99a00c9f74df3ddb59b0951152a82a2`.

### Selected default repeat and preservation

The selected default build passes **1,472 tests** and **319 feature-disabled
checks**. Its repeat reverses the pilot's order, running selected-gradient
workers first. Runtime controls and sample counts remain the same.

| Default-Thomas HVP workload | Steps / nodes | All gradients ms | Requested gradients ms |
|---|---:|---:|---:|
| Surface | 10 / 31 | 16.40 | 17.74 |
| Neural loss | 10 / 31 | 17.72 | 16.32 |
| Surface | 30 / 61 | 38.44 | 36.84 |
| Neural loss | 30 / 61 | 35.04 | 38.29 |
| Surface | 60 / 121 | 67.98 | 64.12 |
| Neural loss | 60 / 121 | 68.45 | 66.70 |

The [30/61 off](../benchmarks/fd_phase19_selected_curvature0.json),
[30/61 on](../benchmarks/fd_phase19_selected_curvature1.json),
[scaling off](../benchmarks/fd_phase19_selected_scaling0.json), and
[scaling on](../benchmarks/fd_phase19_selected_scaling1.json) reports show mixed
wall-time changes. The 30/61 neural total regresses despite the confirmed
reduction in gradient-construction operations; its unchanged forward/HVP
components are also slower. Surface/neural gradient construction changes
**10.88/9.99 to 9.58/10.05 ms**. PCR 30/61 totals change
**120.53/121.30 to 112.53/125.34 ms**. These results do not establish a uniform
speedup for either solver.

Ordinary [controls](../benchmarks/fd_phase19_selected_controls.json), whose
computational paths are unchanged by this flag, measure surface/neural full
totals **7.25/7.34 to 7.09/7.46 ms**, adjoint **5.95/6.30 to 5.72/6.19 ms**,
and inference **3.96/3.84 to 3.68/4.26 ms**. They again demonstrate host
variation even with affinity and HighQoS. Profiles reproduce the pilot's exact
660-call reduction, unchanged retained graphs, and 132 fewer transient/repeated
saves. Eliminating unused work is established; a universal wall-time benefit
or reduced peak memory is not.

The selected [source archive](../benchmarks/fd_phase19_source.zip),
[commands](../benchmarks/fd_phase19_commands.json), and
[validation](../benchmarks/fd_phase19_validation.json) verify **86 source files**,
**78 overlay files**, and **36** worker runtime policies. The only source
differences from the pilot archive select the new default in
`cpp/adjoint.cpp` and the benchmark level. All **564** matched ordinary samples
and **1,136** matched curvature samples (including eight profile results) are
identical. Combined curvature validation covers **2,820** warmup/timed samples,
with maximum price/gradient/HVP differences
**1.78e-14 / 1.36e-13 / 2.56e-13** across configurations. Ordinary
fixed-reference validation covers **2,680** comparisons, with maximum price/VJP
differences **3.56e-15 / 1.78e-14**. Selected source SHA-256:
`7e5106894378c1d34c6e8807b81b483ce18fd3156dc72ecb3682f9d76ed17ed7`.

The full-mode Thomas reverse graph was a prospective optimization target before
the architectural review: its 30/61 gradient construction still has **2,380**
direct tensor calls inside the vector reverse scope and **408** inside nested
native forwards; the complete HVP has **14,721** calls. A vector-RHS
specialization was considered but **has not been implemented and is deferred**.
The user's priority is now the general differentiation assessment above.

## Phase 18: original bands in first-order steps

`QLT_ADJOINT_ASSEMBLED_BANDS=1` extends the original assembly-matrix handoff
to the existing first-order step adjoint and inference path. Zero restores
the preceding three-row inputs. The benchmark level `phase18` enables it;
`phase17` disables it. The full-mode handoff remains independently controlled
by `QLT_ASSEMBLED_BANDS`. No new QuantLib overlay patch is needed.

The private operator owner follows Phase 17's invalidation contract: clear it
before every `setTime` and assembly attempt, then retain only the new successful
result. Original Array coefficients remain materialized. Each adjoint saves
the original matrix, state, and solution; operator copies, later coefficient
rebinding, exceptions, and owner destruction cannot replace these saved tensors.
A fresh no-grad assembly is valid for inference, with no stale coefficient
gradient inherited from a preceding active assembly.

Native forward and ordinary backward read a contiguous logical matrix without
temporary row views. Backward returns one fresh matrix gradient and one state
gradient. The existing row-input implementation shares the same arithmetic
helpers. Original Douglas expression order, reflected application endpoints,
unused outer splitting bands, scaled forward pivot checks, transpose-solve
near-zero pivot checks, and finite-output/VJP checks are preserved. The generic
backward remains available for unsupported incoming weights. First-order mode
continues to reject higher derivatives explicitly; this change adds no new
mixed-AD guarantee to that mode. Full-mode solver selection, elimination states,
higher derivatives, and incoming dual handling remain unchanged.

Eligibility retains the existing exact one-dimensional Black-Scholes operator,
CPU float64, fixed settings, empty boundary set, and supported first-order
kernel modes. Native step arithmetic must be enabled. The matrix route supports
the existing Douglas, implicit, and explicit first-order steps, including
no-grad and inference calls. The public financial API remains CPU-only and
unbatched.

The opt-in build passes **1,371 tests**, including **91 new checks**, plus
**403 targeted checks**, **403 feature-disabled checks**, **179 checks each**
with native step arithmetic or adjoint assembly disabled, and **262 checks**
with the independent full-mode owner disabled. Coverage includes exact original
forward bits, independent dense-solve VJPs and finite differences, extreme
coefficient/time scales, views and aliases, partial activity, incoming-weight
views, pivot/nonfinite failures, first-order rejection, and pricing-state
restoration. Ownership tests run backward in reverse order after the owner has
been destroyed. Existing full-mode, curvature, event, curve, callback, and
mixed-derivative checks remain in the suite.

### Selected default repeat

The selected default rebuild passes all **1,371 tests** and **403
feature-disabled checks**. It changes only the feature default, corresponding
test expectation, and benchmark alias from the archived pilot. Numerical
expressions are unchanged. Workers run with the feature enabled first and
disabled second, reversing the pilot order, with the same CPU-8 HighQoS,
one-thread, 32-warmup, 15-sample controls.
All **40** selected pricing/curvature workers verify the requested affinity and
HighQoS settings.

| Adjoint workload | Steps / nodes | Phase 17 ms | Phase 18 ms |
|---|---:|---:|---:|
| Surface | 30 / 61 | 7.08 | 7.00 |
| Neural loss | 30 / 61 | 7.44 | 6.08 |
| Surface | 60 / 121 | 12.69 | 11.59 |
| Surface | 120 / 241 | 26.51 | 26.52 |
| Neural Bermudan/dividend loss | 30 / 61 | 10.05 | 9.33 |
| Constant volatility | 30 / 61 | 6.62 | 5.69 |

Selected [timing](../benchmarks/fd_phase18_selected_timing.json),
[scaling](../benchmarks/fd_phase18_selected_scaling.json),
[event](../benchmarks/fd_phase18_selected_events.json), and
[constant](../benchmarks/fd_phase18_selected_constant.json) reports support
**1.22x** for neural loss, **1.09x** for 60/121 surface pricing, and **1.16x**
for constant volatility. The 120/241 total is effectively unchanged in this
repeat. Its backward median falls **7.21 to 5.61 ms**, while forward rises
**19.30 to 20.52 ms**; separately measured medians do not necessarily sum to
the median total. All six adjoint backward medians improve, but total gains
vary with workload and run. The pilot's larger surface gain is retained below.

Inference also varies: surface/neural 30/61 changes **3.90/4.20 to 3.91/3.70
ms**, and surface 120/241 **17.61 to 17.72 ms**. Unchanged full-mode
[controls](../benchmarks/fd_phase18_selected_controls.json) measure surface/neural
30/61 **7.27/7.29 to 7.03/7.68 ms**, and 60/121 **14.23/14.81 to 13.90/14.67
ms**. The repeated controls reinforce the limits of attributing small timing
differences to this first-order change.

Unchanged curvature controls ([off](../benchmarks/fd_phase18_selected_curvature0.json),
[on](../benchmarks/fd_phase18_selected_curvature1.json)) measure default-Thomas
surface/neural totals **36.10/36.04 to 38.79/37.97 ms**, and PCR
**108.29/113.36 to 135.90/123.59 ms**. This variation is retained, with no
full-mode curvature speed benefit attributed to the new first-order handoff.

The selected [source archive](../benchmarks/fd_phase18_source.zip),
[commands](../benchmarks/fd_phase18_commands.json), and
[validation](../benchmarks/fd_phase18_validation.json) retain **7,102**
fixed-reference price/VJP comparisons, including every warmup/timed sample,
with maximum price difference **1.25e-14** and VJP difference **2.34e-13**.
All **1,504** selected matched price/VJP samples are identical. Combined
curvature validation covers **1,128 samples**, with maximum HVP difference
**1.64e-13** and directional finite-difference error **1.57e-8**. All **85**
source files and **78** regenerated overlay files match; the pinned QuantLib
checkout is clean. Archive SHA-256:
`fb5db18e51ac1775f90f7b008dc6300b145c177764805b1008757ce92aedf63d`.

### Pilot measurements and mechanism

Fresh workers use one intra/inter-op thread, Windows HighQoS, logical CPU 8,
32 warmups, and 15 timed samples. All **52** pilot pricing/curvature workers
verify process mask 256 and disabled execution-speed throttling. Totals include
construction, forward, VJP, and cleanup; neural cases also include network
evaluation and pricing loss. The library does not set affinity or power policy.

| Adjoint workload | Steps / nodes | Phase 17 ms | Matrix handoff ms |
|---|---:|---:|---:|
| Surface | 30 / 61 | 6.37 | 5.78 |
| Neural loss | 30 / 61 | 6.85 | 5.95 |
| Surface | 60 / 121 | 11.72 | 11.20 |
| Surface | 120 / 241 | 26.34 | 22.71 |
| Neural Bermudan/dividend loss | 30 / 61 | 9.61 | 9.04 |
| Constant volatility | 30 / 61 | 7.67 | 5.73 |

The [timing](../benchmarks/fd_phase18_adjoint_bands_timing.json),
[scaling](../benchmarks/fd_phase18_adjoint_bands_scaling.json),
[event](../benchmarks/fd_phase18_adjoint_bands_events.json), and
[constant](../benchmarks/fd_phase18_adjoint_bands_constant.json) reports show
adjoint gains of **1.05-1.34x**. Inference results are mixed: surface 30/61
improves **3.92 to 3.62 ms**, while neural 30/61 changes **3.78 to 3.99 ms**.
Unchanged full-mode [controls](../benchmarks/fd_phase18_adjoint_bands_controls.json)
also vary: surface/neural 30/61 changes **6.69/7.18 to 7.48/7.65 ms**. Affinity
does not eliminate host frequency/load changes; no uniform inference gain or
full-mode speed effect is claimed.

Unchanged full-mode curvature controls ([off](../benchmarks/fd_phase18_adjoint_bands_curvature0.json),
[on](../benchmarks/fd_phase18_adjoint_bands_curvature1.json)) measure default-Thomas
surface/neural 30/61 totals **34.73/35.31 to 35.21/35.48 ms**. PCR controls vary
more, including surface **115.27 to 150.50 ms**. These controls do not establish
a curvature performance change from a first-order-only optimization.

The [profile](../benchmarks/fd_phase18_adjoint_bands_profile.json) keeps forward
ATen calls at **1,917**, while adjoint backward calls fall **2,624 to 1,700**,
removing **924 calls**. Array packs fall **192 to 93** for adjoint and **191
to 92** for inference; unpacks remain **173**. At 30/61 and 60/121, the
[graph report](../benchmarks/fd_phase18_adjoint_bands_structure.json) falls
**387 to 288** and **627 to 438** nodes. Row-selection nodes fall **167 to
68** and **317 to 128**. Saved-tensor counts fall **493 to 427** and **853 to
727**; byte counts including repeated saves remain **258,366 / 919,566**.
Those counts are not unique storage. Cold peak RSS growth falls only about
**0.26 / 0.15 MiB** in this repeat, so no substantial memory benefit is claimed.
Instrumented assembly/step-forward scopes remain about **1.69 / 0.77 ms**
inclusive for adjoint; nested scope times are not additive wall-clock costs.

The pilot [source archive](../benchmarks/fd_phase18_adjoint_bands_source.zip),
[commands](../benchmarks/fd_phase18_adjoint_bands_commands.json), and
[validation](../benchmarks/fd_phase18_adjoint_bands_validation.json) retain
**4,606** fixed-reference price/VJP comparisons, with maximum price difference
**1.25e-14** and VJP difference **2.34e-13**. All **1,508** matched numeric
price/VJP samples are identical. Combined validation with the preceding selected
curvature reports covers **752 samples**, with maximum HVP difference
**1.64e-13**. All **85** source files and **78** regenerated overlay files
match. Archive SHA-256:
`7fb05f3f293acd079b3a9eb98ae45c540984a67c71a59ba639fb03163c3fb0b7`.
The [baseline profile](../benchmarks/fd_phase18_baseline_profile.json),
[baseline graph](../benchmarks/fd_phase18_baseline_structure.json), and
[baseline validation](../benchmarks/fd_phase18_baseline_validation.json) preserve
the initial unchanged Phase 17 binary and source comparison.

Remaining measured costs include assembly/materialization and the remaining
row-selection nodes. Full-mode single-RHS solve views and unnecessary gradients
for passive assembly operands are candidates for separate investigation. Any
follow-up must preserve original inputs, solver states, independent VJPs, and
higher/mixed derivatives; no architectural pivot is implied.

## Phase 17: retain original assembled bands

`QLT_ASSEMBLED_BANDS=1` retains the original graph-connected assembly tensor
in its private `FdmBlackScholesOp` owner and passes it directly to supported
full-mode step preparation. Zero restores the preceding three-row inputs.
The benchmark level `phase17` selects the default; `phase16` disables it.
This removes three Array packs and three row-selection gradient edges per
eligible step. Original Array coefficients remain materialized for other
operations and fallback paths.

The owner is cleared at the start of every `setTime`, before rates, sampling,
or any possible exception. The assembly hook also clears the supplied owner
before an attempt and populates it only after successful supported assembly.
Alternative routes therefore cannot reuse a preceding active tensor. Operator
copies retain their assembled values; each preparation operation saves its
own original matrix and state, so later reassembly or owner destruction cannot
replace the tensors needed by backward. The storage/lifetime changes are
isolated in generated `assembled-bands.patch`; compatibility patches and
numerical substitutions remain separate. The pinned QuantLib sources and
license attribution are preserved.

The preparation boundary returns one matrix gradient. Native forward and
ordinary backward read one contiguous logical matrix, resolving views without
changing the original tensor. They write fresh outputs and gradient buffers.
Higher derivatives and incoming dual VJP weights reconstruct graph-connected
row views from the saved original matrix and use the existing functional
formulas. No coefficient is detached, no input is updated in place, and no
storage/base-pointer inference is used. The elementary arithmetic, actual
neighbor indices, auto-domain/residual checks, and complete Thomas/PCR
elimination-state graph are unchanged.

Eligibility retains Phase 16's exact one-dimensional operator, CPU float64,
gradient recording, `kernels='auto'`, fixed settings, and empty boundary set.
Assembly arithmetic and native step preparation must be enabled. Explicit
steps, small or rejected systems, active settings, callbacks, subclasses,
forward duals, and transformed inputs keep their existing handling. First-order
adjoints and inference keep their preceding paths. The public financial API
remains CPU-only and unbatched.

The matrix opt-in and selected default builds pass **1,280 tests**, including
**93 new checks**. The opt-in revision also passes **409 targeted checks**,
**409 owner-disabled checks**, **194 assembly-arithmetic-disabled checks**,
**409 preparation-disabled checks**, and **272 checks each** with Thomas
arithmetic or native operator forward disabled. The selected rebuild passes
all **409 owner-disabled checks**. Coverage includes exact original forward
bits, every output VJP and undefined weight, finite differences through second
order, derivatives through third order, all elimination-output dependencies,
views/aliases, incoming duals, forward AD, transforms, dtype/device fallbacks,
operator copies, coefficient rebinding, failed/fallback/no-grad assembly,
reverse-order backward after owner destruction, and rejected huge-RHS or
weak-dominance systems. Existing financial event, curve, callback, and curvature
checks run in the full suite. Probe transform/CUDA checks do not establish
financial tracing or GPU-pricing support.

### Selected measurements and reproducibility

Fresh workers use one intra/inter-op thread, Windows HighQoS, logical CPU 8,
32 warmups, and 15 timed samples. All **32** selected pricing/curvature
workers record mask 256, processor 8 at observation, and disabled execution-speed
throttling. Pricing totals include construction, forward, VJP, and cleanup;
neural workloads also include network evaluation and pricing loss. The selected
repeat runs `phase16` before `phase17`, reversing the matrix opt-in order.
The additional `phase15` timing controls disable both recent changes in the
same binary and runtime configuration.

| Full-mode workload | Steps / nodes | Phase 15 ms | Phase 16 ms | Phase 17 ms |
|---|---:|---:|---:|---:|
| Surface | 30 / 61 | 9.29 | 8.91 | 7.94 |
| Neural loss | 30 / 61 | 10.04 | 8.63 | 7.35 |
| Surface | 60 / 121 | 18.87 | 14.31 | 13.75 |
| Surface | 120 / 241 | 40.70 | 31.12 | 28.62 |
| Neural Bermudan/dividend loss | 30 / 61 | Not measured | 11.82 | 10.94 |
| Constant volatility | 30 / 61 | Not measured | 7.57 | 6.81 |

The [timing](../benchmarks/fd_phase17_selected_timing.json),
[scaling](../benchmarks/fd_phase17_selected_scaling.json),
[event](../benchmarks/fd_phase17_selected_events.json), and
[constant](../benchmarks/fd_phase17_selected_constant.json) reports support
selected full-mode speedups of **1.04-1.17x** over Phase 16. Relative to
Phase 15 in this run, neural loss improves **1.37x** and 120/241 surface
pricing **1.42x**. These are measured cases, not universal ratios.

Unchanged [adjoint/inference controls](../benchmarks/fd_phase17_selected_controls.json)
change **6.66/3.77 to 6.89/4.16 ms** for surface pricing and **7.17/3.92 to
7.03/3.91 ms** for neural pricing. Their variation, including the roughly 10%
surface-inference shift, shows that affinity does not fix frequency or load.
No adjoint/inference gain is attributed to this full-mode change. The 60/121
full-mode gain is smaller than in the matrix opt-in repeat; both sets of raw
samples are retained.

Complete default-Thomas 30/61 surface/neural Hessian-vector workloads improve
**38.51/38.48 to 36.13/36.52 ms** ([off](../benchmarks/fd_phase17_selected_curvature0.json),
[on](../benchmarks/fd_phase17_selected_curvature1.json)). Automatic-PCR controls
with Thomas disabled change **119.08/128.93 to 118.99/121.50 ms**. PCR and
smaller-grid results vary across the pilots below, so no uniform curvature
speedup is claimed. These totals include forward, gradient recording, HVP,
and cleanup.

The selected [source archive](../benchmarks/fd_phase17_source.zip),
[commands](../benchmarks/fd_phase17_commands.json), and
[validation](../benchmarks/fd_phase17_validation.json) retain **7,300**
fixed-reference price/VJP comparisons across selected and historical reports,
with maximum differences **3.56e-15 / 1.53e-13**. All **940** selected matched
owner-on/off price/VJP samples are identical. Combined curvature validation
covers **2,256** samples, with maximum price/gradient/HVP differences
**1.24e-14 / 6.04e-14 / 1.64e-13** and maximum central-difference HVP discrepancy
**1.63e-8**. All **84** expected source files and **78** regenerated overlay
files match; pinned QuantLib remains clean. Archive SHA-256:
`a511f791d964905726a556d5110d985694671e9eaddca6eac3198f0b73eb9905`.
Only the feature default, matching test expectation, and benchmark alias
differ from the matrix opt-in archive; numerical code is unchanged.

### Matrix opt-in measurements and retained prototype

The matrix opt-in uses the same worker policy, with the new path before
`phase16`. Entries here are disabled to enabled. Its
[surface/neural 30/61 totals](../benchmarks/fd_phase17_matrix_timing.json)
improve **7.46/7.83 to 6.69/7.22 ms**;
[60/121 and 120/241 surface totals](../benchmarks/fd_phase17_matrix_scaling.json)
improve **14.16/30.52 to 12.39/27.32 ms**. Neural
[Bermudan/dividend loss](../benchmarks/fd_phase17_matrix_events.json) changes
**11.20 to 10.65 ms**, and [constant volatility](../benchmarks/fd_phase17_matrix_constant.json)
**7.06 to 6.30 ms**. Unchanged [controls](../benchmarks/fd_phase17_matrix_controls.json)
are retained. Default-Thomas 30/61 surface/neural HVP totals improve
**37.02/38.01 to 33.63/36.93 ms**, while smaller-grid and PCR changes are
mixed ([off](../benchmarks/fd_phase17_matrix_curvature0.json),
[on](../benchmarks/fd_phase17_matrix_curvature1.json)).

The [profile](../benchmarks/fd_phase17_matrix_profile.json) confirms full-mode
30/61 forward ATen calls remain **1,844**, while backward calls fall **3,183
to 2,358**, eliminating **825 calls**. Array packs fall **190 to 91**;
unpacks remain **173**. Domain and residual checks remain at **34 each**.
The [graph report](../benchmarks/fd_phase17_matrix_structure.json) falls **488 to
389** nodes at 30/61 and **818 to 629** at 60/121. Row-selection nodes fall
**167 to 68** and **317 to 128** respectively. Saved-tensor counts fall
**794 to 728** and **1,424 to 1,298**, while byte counts including repeated
saves remain **402,774 / 1,467,414**. Those counts are not unique live storage.
Cold peak RSS growth falls about **0.09/0.79 MiB** in this repeat; the earlier
prototype includes a small increase at 60/121, so no general memory gain is
claimed. Instrumented assembly/full-step scopes remain about **1.83/1.25 ms**
inclusive; nested scope times are not additive wall-clock costs.

The matrix [source archive](../benchmarks/fd_phase17_matrix_source.zip),
[commands](../benchmarks/fd_phase17_matrix_commands.json), and
[validation](../benchmarks/fd_phase17_matrix_validation.json) retain **5,188**
fixed-reference comparisons, **944** identical matched numeric price/VJP
samples, and **1,880** combined curvature samples. All **84** source files
and **78** overlays match. Archive SHA-256:
`014dc1d7733613843d2e2500b6dd5b51a4a48fd7c0fe54bdf4e168b0537dbce4`.

The first prototype retained the original matrix but created temporary row
views inside native preparation. It passed **1,278 tests** and reduced the
graph equally, yet added **198 forward ATen calls**; backward calls were
**2,556**, compared with **2,358** after removing those temporary views.
Its [timing](../benchmarks/fd_phase17_bands_timing.json) shows surface/neural
30/61 totals **7.86/8.71 to 7.99/8.08 ms**. The
[scaling](../benchmarks/fd_phase17_bands_scaling.json),
[event](../benchmarks/fd_phase17_bands_events.json),
[constant](../benchmarks/fd_phase17_bands_constant.json),
[control](../benchmarks/fd_phase17_bands_controls.json),
[graph](../benchmarks/fd_phase17_bands_structure.json), and
[profile](../benchmarks/fd_phase17_bands_profile.json) reports retain its modest
or mixed effects. Its [off](../benchmarks/fd_phase17_bands_curvature0.json) and
[on](../benchmarks/fd_phase17_bands_curvature1.json) curvature reports also vary.
This prototype was revised before selecting a default.

The prototype [source archive](../benchmarks/fd_phase17_bands_source.zip),
[commands](../benchmarks/fd_phase17_bands_commands.json), and
[validation](../benchmarks/fd_phase17_bands_validation.json) retain **3,452**
fixed-reference comparisons and **1,128** combined curvature samples.
The matched-record counter includes 950 entries, of which 944 contain numeric
price/VJP samples; all those numeric samples are identical. All **84** source
files and **78** overlays match. Archive SHA-256:
`f53b1efdda12d2d2eb038caac0c093378f3ae53339cc19324fa6d69b50cab9de`.

Full-mode materialization still performs 173 Array unpacks per 30/61 price,
and single-RHS solve shape operations remain on its graph. The first-order
adjoint step still packs and differentiates three separate band rows. Next
investigate an explicit original-matrix handoff for that path, retaining its
transpose-pivot checks and first-order-only contract, and profile the remaining
full-mode conversions before choosing further changes.

## Phase 16: elementary step preparation

`QLT_STEP_PREPARATION=1` prepares the scaled bands and right-hand side of
supported full-mode Douglas and implicit-Euler steps inside one elementary
CPU autograd operation. Zero restores Phase 15's tensor expressions. The
benchmark level `phase16` selects the default; `phase15` disables it.

The new operation retains the pinned QuantLib expression order: band scaling
uses `(-theta)*dt`, and Douglas forms `(state + dt*applied) -
(theta*dt)*applied` with separate products and add/subtract operations.
Application uses the operator's actual neighbor indices, including reflected
endpoints. Its translation unit disables floating-point contraction and
reassociation. Original bands and state remain saved; native forward and
ordinary backward write fresh buffers. Higher derivatives and incoming dual
VJP weights use functional formulas on those original inputs. The existing
auto-domain checks, residual test, and full Thomas/PCR elimination-state graph
remain outside this boundary. It introduces no new solve or implicit VJP.
Attribution to the pinned QuantLib step and triple-band sources is retained in
`cpp/step_preparation.cpp`, with the license in `LICENSES/QuantLib.txt`.

Eligibility inherits the full tensor-step hook: ordinary CPU float64 tensors,
gradient recording, `kernels='auto'`, exact one-dimensional
`FdmBlackScholesOp`, passive settings, and no boundary callbacks. Native
elementary forward arithmetic must also be enabled. Explicit steps, active
settings, other operators, boundary callbacks, forward duals, and transformed
inputs keep their existing dispatch. A rejected solve leaves state untouched
for the original step. Financial objects/events, first-order adjoints,
inference, and the scalar adapter keep their existing paths. There is no new
QuantLib overlay patch.

Both opt-in and selected builds pass **1,187 tests**, including **101 new
checks**. The opt-in build also passes **293 targeted checks**, **293 with
preparation disabled**, and three separate **179-check** configurations with
Thomas arithmetic, native operator forward, or full tensor steps disabled.
The selected default rebuild passes **294 preparation-disabled checks**,
including the earlier operator-pricing dispatch check. Coverage includes
exact preceding-expression forward bits, all four output VJPs, undefined
output weights, independent differences, derivatives through third order,
all Thomas intermediate-output dependencies, actual/permuted neighbors,
strided/negative/shared/expanded inputs, partial activity, input versions,
incoming dual weights, forward AD, gradient/JVP/function transforms, fake
tracing, dtype/device fallbacks, financial events, rejected domains, and
Hessian-vector products. Numerical-probe fallbacks do not establish financial
GPU execution or tracing support.

### Selected measurements and runtime controls

The final repeat pins each fresh worker to logical CPU 8 on this host, uses
Windows HighQoS and one intra/inter-op thread, and records 32 warmups plus 15
timed samples. It runs `phase16` before `phase15`, reversing the opt-in order.
All 28 pricing/curvature workers record process affinity mask 256, processor
8 at observation, and disabled execution-speed throttling. Affinity is an
optional benchmark setting; the pricing library does not change it. Totals
include construction, forward, VJP, and cleanup; neural workloads include
network evaluation and pricing loss. Entries below are **disabled to enabled**.

| Full-mode workload | Steps / nodes | Median ms | Ratio |
|---|---:|---:|---:|
| Surface | 30 / 61 | 11.67 to 7.73 | 1.51x |
| Neural loss | 30 / 61 | 9.53 to 7.21 | 1.32x |
| Surface | 60 / 121 | 17.76 to 13.67 | 1.30x |
| Surface | 120 / 241 | 43.78 to 28.98 | 1.51x |
| Neural Bermudan/dividend loss | 30 / 61 | 12.91 to 10.97 | 1.18x |
| Constant volatility | 30 / 61 | 8.84 to 6.81 | 1.30x |

The [timing](../benchmarks/fd_phase16_affinity_timing.json),
[scaling](../benchmarks/fd_phase16_affinity_scaling.json),
[event](../benchmarks/fd_phase16_affinity_events.json), and
[constant](../benchmarks/fd_phase16_affinity_constant.json) reports retain all
samples. Affinity does not fix frequency or competing load: the 30/61 surface
samples show within-run variation, so the larger ratios should not be treated
as universal gains. The neural 30/61 and surface 60/121 improvements are also
supported by the unrestricted pilot below.

Unchanged [adjoint/inference controls](../benchmarks/fd_phase16_affinity_controls.json)
stay close: surface totals change **6.56/3.49 to 6.65/3.46 ms**, and neural
totals **6.53/3.56 to 6.33/3.58 ms**. Complete default-Thomas 30/61
surface/neural Hessian-vector workloads improve **36.94/39.65 to
34.46/36.39 ms** ([off](../benchmarks/fd_phase16_affinity_curvature0.json),
[on](../benchmarks/fd_phase16_affinity_curvature1.json)). Automatic-PCR controls
with Thomas disabled change **117.22/117.99 to 115.80/114.96 ms**; across all
runs their changes are small or mixed, so no general PCR curvature gain is
claimed. Curvature totals include forward, gradient recording, HVP, and cleanup.

An earlier unrestricted selected repeat retains improved full-mode timings
([timing](../benchmarks/fd_phase16_selected_timing.json),
[scaling](../benchmarks/fd_phase16_selected_scaling.json),
[events](../benchmarks/fd_phase16_selected_events.json),
[constant](../benchmarks/fd_phase16_selected_constant.json)), but later
[controls](../benchmarks/fd_phase16_selected_controls.json) change substantially
on unchanged paths: the surface adjoint measures 9.93 versus 6.44 ms.
Its [off](../benchmarks/fd_phase16_selected_curvature0.json) and
[on](../benchmarks/fd_phase16_selected_curvature1.json) curvature comparisons
are consequently confounded by host timing regimes. They remain archived.
Follow-up diagnostics use the unchanged surface-adjoint workload: last-48
medians are **6.13 ms** with [unrestricted HighQoS](../benchmarks/fd_phase16_runtime_highqos.json),
**6.15 ms** on [CPU 8](../benchmarks/fd_phase16_runtime_cpu8.json), and **13.17 ms**
on [CPU 19](../benchmarks/fd_phase16_runtime_cpu19.json). Prices/VJPs are identical
through all 80 samples per run. These observations motivate the explicit
affinity repeat; they do not identify core architecture or calibrated clock
frequency. Runtime diagnostics observe scheduling outside each timed price.

The final [source archive](../benchmarks/fd_phase16_source.zip),
[commands](../benchmarks/fd_phase16_commands.json), and
[validation](../benchmarks/fd_phase16_validation.json) retain **5,660**
fixed-reference price/VJP comparisons, with maximum differences **3.56e-15 /
1.53e-13**. The **940** matched final preparation-on/off samples have identical
prices and maximum VJP difference **1.53e-13**. Combined curvature validation
covers **1,504** samples with maximum price/gradient/HVP differences
**1.24e-14 / 6.04e-14 / 1.43e-13**. All **83** expected source files and **77**
regenerated overlay files match; pinned QuantLib remains clean. Archive SHA-256:
`cfb46cf13b8e3b40588c9d75abfcf0b0d0dcd36761820f1f7478c8da4c4c5548`.

The [initial selected source](../benchmarks/fd_phase16_initial_source.zip),
[commands](../benchmarks/fd_phase16_initial_commands.json), and
[validation](../benchmarks/fd_phase16_initial_validation.json) preserve the first
repeat. Its archive SHA-256 is
`557a546ffc1b6f9a6df0a363504c8afa703dc59fd801a7d1228663adda3f4968`.
The final archive has identical native sources and binary; subsequent changes
only add optional benchmark affinity controls. Relative to the pilot, numerical
code is unchanged apart from enabling the feature by default, with corresponding
test expectations and benchmark alias/control changes.

### Opt-in evidence and remaining work

The pilot uses unrestricted HighQoS workers with the same thread/warmup/sample
policy. Full-mode [surface/neural 30/61 totals](../benchmarks/fd_phase16_preparation_timing.json)
improve **14.95/15.26 to 11.95/12.27 ms**. Surface
[60/121 and 120/241 totals](../benchmarks/fd_phase16_preparation_scaling.json)
improve **30.59/62.51 to 23.64/49.96 ms**. Neural
[Bermudan/dividend loss](../benchmarks/fd_phase16_preparation_events.json)
changes **21.53 to 17.52 ms**, and
[constant volatility](../benchmarks/fd_phase16_preparation_constant.json)
**14.57 to 11.30 ms**. Unchanged
[adjoint/inference controls](../benchmarks/fd_phase16_preparation_controls.json)
are retained. Default-Thomas 30/61 surface/neural HVP totals improve
**64.63/65.92 to 58.44/58.57 ms**; smaller-grid and PCR results vary
([off](../benchmarks/fd_phase16_preparation_curvature0.json),
[on](../benchmarks/fd_phase16_preparation_curvature1.json)).

The [profile](../benchmarks/fd_phase16_preparation_profile.json) records full-mode
30/61 forward/backward ATen calls falling **2,129/3,982 to 1,844/3,183**.
The 33 full-step scopes change **3.73 to 2.04 ms** inclusive, while domain
and residual checks remain at **34 each**. Array packs/unpacks remain at
**190/173**. These nested instrumented times are not additive wall-clock costs.
The [graph report](../benchmarks/fd_phase16_preparation_structure.json) falls
**807 to 488** nodes at 30/61 and **1,437 to 818** at 60/121. Saved-tensor
counts rise by 12 at both grids; byte counts including repeated saves change
**398,870 to 402,774** and **1,459,670 to 1,467,414** respectively. They are
not unique live storage. Cold peak RSS growth falls about **0.23/0.65 MiB**;
this modest observation does not establish general memory savings.

The opt-in [source archive](../benchmarks/fd_phase16_preparation_source.zip),
[commands](../benchmarks/fd_phase16_preparation_commands.json), and
[validation](../benchmarks/fd_phase16_preparation_validation.json) retain
**4,508** fixed-reference comparisons including historical Phase 15 reports,
**950** matched records (944 with numeric price/VJP samples), and **752**
curvature samples.
All **83** source files and **77** overlays match. Archive SHA-256:
`5483271974512436a2e736aef21f86d3c882396a0b3a9093f317d22a8d06eef4`.

Remaining assembly costs about **3.04 ms** inclusive in the instrumented
30/61 profile. The 488-node graph still contains **167 row-selection nodes**,
and array packing/materialization counts have not fallen. Next investigate
retaining the original assembled band tensor across the operator/step boundary,
with explicit ownership and invalidation, to reduce those conversions and
gradient edges. Preserve original input dependencies, fallback behavior,
financial callbacks, and fresh graphs between prices. The financial API remains
CPU-only and unbatched.

## Phase 15: elementary time-variance arithmetic

`QLT_TIME_VARIANCE_ARITHMETIC=1` enables the existing local-volatility time
interpolation and square inside one elementary CPU autograd operation. Zero
restores the preceding tensor expression. The benchmark level `phase15`
selects the new default; `phase14` disables it.

The operation computes `sigma = (1-weight)*low + weight*high`, then
`sigma*sigma`, preserving the two products, addition, and square in their
original order. Its translation unit disables floating-point contraction and
reassociation. It saves the original interpolation rows and writes only fresh
outputs. Ordinary backward evaluates the original elementary VJP with native
arithmetic. Higher derivatives and incoming forward-mode VJP weights use
functional tensor formulas reconstructed from those original rows. No
coefficient is silently detached, and this introduces no implicit-adjoint
approximation or new cache.

Eligibility is limited to ordinary CPU float64 rows. Forward duals,
transformed/wrapped tensors, other dtypes, and other devices retain the
original tensor expression. The existing sampling hook still requires passive
times and coordinates; time clamps and single-time surfaces retain their
existing square/cache path. QuantLib objects, rollback events, overwrite
policy, scalar adapter, and first-order-only financial adjoint contract keep
their existing handling. There is no new QuantLib overlay patch.

Both the opt-in and selected builds pass **1,086 tests**, including **74 new
checks**. The opt-in build passes **216 targeted checks**, **216 checks with
the new operation disabled**, and **74 with spatial caching disabled**. The
selected default rebuild also passes all **216 disabled-operation checks**.
Coverage includes exact original forward bits, both input VJPs, strided and
negative views, repeated/shared/expanded inputs, finite differences through
second order, derivatives through third order, incoming dual weights, forward
AD, gradient/JVP/function transforms, fake tracing of the numerical probe,
CPU/CUDA dtype fallbacks, inference, input versions, extreme/nonfinite values,
fresh and overlapping financial graphs, curves, exercise/dividend events, and
financial Hessian-vector products. These numerical-probe fallbacks do not add
a financial tracing or GPU-pricing guarantee.

### Selected measurements and reproducibility

The selected run measures `phase15` before `phase14`, reversing the pilot
order. Fresh HighQoS workers use one intra/inter-op thread, 32 warmups, and
15 timed samples. Price/VJP totals include construction, forward, backward,
and cleanup; neural workloads also include network evaluation and the pricing
loss. Inference omits backward. All entries below are **disabled to enabled**.

| Workload | Steps / nodes | Full ms | Adjoint ms | Inference ms |
|---|---:|---:|---:|---:|
| Surface | 30 / 61 | 16.29 to 15.54 | 11.49 to 10.46 | 6.26 to 6.32 |
| Neural loss | 30 / 61 | 16.20 to 15.95 | 11.68 to 11.16 | 6.17 to 6.29 |
| Surface | 60 / 121 | 30.06 to 30.48 | 21.95 to 19.49 | 12.68 to 13.05 |
| Surface | 120 / 241 | 64.16 to 62.97 | 45.20 to 42.30 | 29.33 to 28.83 |
| Neural Bermudan/dividend loss | 30 / 61 | 22.39 to 21.85 | 17.67 to 16.64 | Not measured |

The [timing](../benchmarks/fd_phase15_selected_timing.json),
[scaling](../benchmarks/fd_phase15_selected_scaling.json), and
[event](../benchmarks/fd_phase15_selected_events.json) reports support adjoint
gains of about **1.05-1.13x** across these workloads. Full-mode gains are
smaller, with a slight slowdown at 60/121. Inference results vary by grid and
run. Constant-volatility [controls](../benchmarks/fd_phase15_selected_constant.json)
do not invoke the new operation: full/adjoint/inference totals vary from
**14.28/9.65/6.09 to 14.99/9.76/5.77 ms**. No uniform full-mode or inference
speedup is claimed.

The selected curvature repeat
([enabled](../benchmarks/fd_phase15_selected_curvature1.json),
[disabled](../benchmarks/fd_phase15_selected_curvature0.json)) measures complete
30/61 default-Thomas surface/neural Hessian-vector workloads at
**64.80/66.53 to 63.78/66.62 ms**. Automatic-PCR controls with Thomas disabled
change **193.77/196.67 to 198.86/205.59 ms**, reversing the pilot's direction.
The pilot also has mixed default-Thomas results, so no consistent curvature
gain is claimed. These totals include forward, gradient recording, HVP, and
cleanup.

The selected [source archive](../benchmarks/fd_phase15_source.zip),
[commands](../benchmarks/fd_phase15_commands.json), and
[validation](../benchmarks/fd_phase15_validation.json) retain **5,572**
fixed-reference price/VJP comparisons across pilot and selected reports, with
maximum differences **1.24e-14 / 2.04e-13**. All **1,598** selected matched
sampling-on/off samples have identical prices and VJPs. Combined curvature
validation covers **1,128** raw samples, with maximum price/gradient/HVP
differences **1.24e-14 / 6.04e-14 / 1.31e-13** and maximum central-difference
HVP discrepancy **1.63e-8**. All **80** expected source files and **77**
regenerated overlay files match; the pinned QuantLib checkout is clean. The
selected archive SHA-256 is
`1ea439a7ce72da0e6eda17ea629151d6999f737e0f009fba449992d85bdf678f`.
Only the feature default, matching test expectation, and benchmark alias
differ from the opt-in source; numerical code is unchanged.

### Opt-in evidence and remaining work

The pilot uses the same measurement policy, with `phase14` before
`time_variance`. Its [timing](../benchmarks/fd_phase15_time_variance_timing.json),
[scaling](../benchmarks/fd_phase15_time_variance_scaling.json), and
[event](../benchmarks/fd_phase15_time_variance_events.json) reports retain every
sample. Entries are again disabled to enabled.

| Workload | Steps / nodes | Full ms | Adjoint ms | Inference ms |
|---|---:|---:|---:|---:|
| Surface | 30 / 61 | 16.53 to 15.59 | 11.04 to 10.42 | 6.49 to 6.23 |
| Neural loss | 30 / 61 | 16.77 to 16.02 | 11.66 to 10.48 | 6.64 to 5.94 |
| Surface | 60 / 121 | 31.34 to 29.66 | 22.02 to 19.90 | 12.51 to 12.52 |
| Surface | 120 / 241 | 66.50 to 61.90 | 45.05 to 42.47 | 29.77 to 28.46 |
| Neural Bermudan/dividend loss | 30 / 61 | 22.20 to 21.50 | 17.28 to 16.11 | Not measured |

The [profile](../benchmarks/fd_phase15_time_variance_profile.json) confirms **99
fewer forward and 99 fewer backward ATen calls** at 30/61: full mode changes
**2,228/4,081 to 2,129/3,982**, and adjoint mode **2,016/2,723 to 1,917/2,624**.
The full-mode sampling scope changes **1.44 to 0.67 ms**, and coefficient
assembly including sampling changes **4.05 to 2.97 ms**. These are nested
instrumented times and must not be added or substituted for wall-clock totals.

The [graph measurements](../benchmarks/fd_phase15_time_variance_structure.json)
fall **906 to 807** nodes in full mode and **486 to 387** in adjoint mode at
30/61. At 60/121 they fall **1,626 to 1,437** and **816 to 627** respectively.
Saved-tensor counts and byte counts including repeated saves are unchanged;
they are not unique live storage. Cold peak RSS changes range from about
**-0.27 to +0.12 MiB**, so no general memory reduction is claimed. The
[constant controls](../benchmarks/fd_phase15_time_variance_constant.json) retain
the same execution path.

The pilot curvature reports
([off](../benchmarks/fd_phase15_time_variance_curvature0.json),
[on](../benchmarks/fd_phase15_time_variance_curvature1.json)) measure default-Thomas
surface/neural HVP totals **29.75/29.47 to 29.34/30.65 ms** at 10/31 and
**61.48/66.52 to 63.66/65.17 ms** at 30/61. The 30/61 automatic-PCR controls
change **194.05/205.22 to 193.24/196.77 ms**. Across **752** raw samples,
maximum HVP difference is **1.31e-13**.

The opt-in [source archive](../benchmarks/fd_phase15_time_variance_source.zip),
[commands](../benchmarks/fd_phase15_time_variance_commands.json), and
[validation](../benchmarks/fd_phase15_time_variance_validation.json) retain
**4,324** fixed-reference comparisons including historical Phase 14 controls,
with maximum price/VJP differences **1.24e-14 / 2.04e-13**. All **1,618**
matched sampling-on/off price/VJP samples are identical. All **80** expected
source files and **77** regenerated overlay files match; pinned QuantLib is
clean. The opt-in archive SHA-256 is
`8633f1d687bfcd56dbccf08c1f8a8878dc5888011e1b7a442a66d26f31a655db`.

The remaining full-mode profile places about **4.23 ms** in the tensor-step
scope and **2.97 ms** in assembly, including **0.67 ms** in sampling. Array
packs/unpacks remain at **190/173** per 30/61 price. Next investigate remaining
step graph operations and coefficient materialization, retaining the original
elimination-state dependencies, mixed AD, financial callbacks, and fallbacks.
The financial API remains CPU-only and unbatched.

## Phase 14: full-mode tensor steps

`QLT_FULL_TENSOR_STEP=1` enables a tensor composition of supported full-mode
steps; zero restores the preceding array-based step. The benchmark level
`phase14` selects this path and `phase13` disables it. The composition calls
the existing full-AD elementary application and guarded solve kernels. It
introduces no new VJP formula or implicit-adjoint shortcut. Original inputs
and the selected solver's graph-connected elimination states retain higher
derivatives.

The pinned one-dimensional `FdmBlackScholesOp` has identical `apply` and
`apply_direction(0)` operations within a step. Douglas now computes that
application once, retaining the expression `(state + dt*applied) -
(theta*dt)*applied` with its separate products and add/subtract ordering.
Implicit and explicit steps use their corresponding original expressions.
Intermediate values remain tensors, avoiding repeated Array packing and
unpacking. The composition is attributed to QuantLib v1.40's Douglas,
implicit-Euler, and explicit-Euler sources in `cpp/step_adjoint.cpp`; the
QuantLib license remains in `LICENSES/QuantLib.txt`.

Eligibility remains narrow: CPU float64, gradient recording enabled,
`kernels='auto'`, exact `FdmBlackScholesOp`, direction zero in one dimension,
fixed passive settings, and no boundary callbacks. `setTime` executes in the
original scheme before this hook. Events remain in the original rollback.
Active settings, other operator types, boundary callbacks, forward duals, and
other kernel modes keep the existing paths. The full-AD application switch
must also be enabled. Domain and residual checks still select or reject the
same solve; a rejected candidate leaves the state untouched for the original
step. The first-order adjoint and inference paths are unchanged.

Both the opt-in and selected default builds pass **1,012 tests**, including
**78 new checks**, plus
**229 targeted checks**, **229 step-disabled fallback checks**, and **78
kernel-fallback checks** with native application forward and Thomas arithmetic
disabled. The selected rebuild also passes all **229 step-disabled checks**.
Coverage includes all band/state VJPs against original schemes and
dense equations, exact preceding-path forward bits, reflected endpoints,
views/shared inputs, first/second differences, derivatives through third
order, incoming VJP duals, forward AD, gradient/JVP transforms, rejected
domains, boundary callbacks, active settings, subclasses, and financial
curves with coincident exercise/dividend events and curvature.

### Selected measurements and reproducibility

The selected run measures `phase14` before `phase13`, reversing the opt-in
order. Its [timing](../benchmarks/fd_phase14_selected_timing.json),
[scaling](../benchmarks/fd_phase14_selected_scaling.json), and
[event](../benchmarks/fd_phase14_selected_events.json) reports use the same
HighQoS, one-thread, 32-warmup, 15-sample policy and end-to-end boundaries.

| Full-mode workload | Steps / nodes | Phase 13 ms | Selected ms | Speedup |
|---|---:|---:|---:|---:|
| Surface | 30 / 61 | 18.42 | 16.38 | 1.12x |
| Neural loss | 30 / 61 | 18.99 | 16.15 | 1.18x |
| Surface | 60 / 121 | 36.62 | 31.87 | 1.15x |
| Surface | 120 / 241 | 83.57 | 65.36 | 1.28x |
| Neural Bermudan/dividend loss | 30 / 61 | 24.58 | 21.70 | 1.13x |

The selected curvature repeat
([enabled](../benchmarks/fd_phase14_selected_curvature1.json),
[disabled](../benchmarks/fd_phase14_selected_curvature0.json)) improves complete
30/61 surface/neural default-Thomas Hessian-vector workloads from
**72.79/72.98 to 64.93/65.45 ms**, about **1.12x** each. Automatic-PCR controls
with Thomas disabled measure surface **193.07 to 192.96 ms** and neural
**227.15 to 191.05 ms**. The neural PCR result changes direction from the pilot,
so no consistent PCR curvature gain is claimed. Adjoint/inference
[controls](../benchmarks/fd_phase14_selected_controls.json) remain around
11.6-11.9 ms and 6.1-6.4 ms, with no speedup claim for those unchanged paths.

The selected [source archive](../benchmarks/fd_phase14_source.zip),
[commands](../benchmarks/fd_phase14_commands.json), and
[validation](../benchmarks/fd_phase14_validation.json) retain **3,070**
fixed-reference price/VJP comparisons across the pilot and selected reports,
with maximum differences **3.55e-15 / 1.33e-13**. The **846** selected step
on/off samples have identical prices and maximum VJP difference **1.33e-13**.
Combined curvature validation covers **1,128** raw samples across both
revisions, with maximum price/gradient/HVP differences
**1.24e-14 / 8.88e-14 / 1.78e-13** and maximum central-difference HVP discrepancy
**1.63e-8**. All **77** expected source files and **77** regenerated overlay
files match; the pinned QuantLib checkout is clean. The selected archive
SHA-256 is `71374a7ba596415f08e3aba7d2a50d42a12ed94a2df51be32643bbc478287714`.
Only the feature default, its test expectation, and the benchmark alias differ
from the opt-in source; numerical code is unchanged.

### Opt-in evidence

Fresh HighQoS workers use one intra/inter-op thread, 32 warmups, and 15 timed
samples. Full-mode totals include construction, forward, VJP, and cleanup;
neural cases also include network evaluation and the pricing-loss VJP.

| Full-mode workload | Steps / nodes | Phase 13 ms | Tensor step ms | Speedup |
|---|---:|---:|---:|---:|
| Surface | 30 / 61 | 18.37 | 16.09 | 1.14x |
| Neural loss | 30 / 61 | 18.72 | 16.49 | 1.13x |
| Surface | 60 / 121 | 37.28 | 31.17 | 1.20x |
| Surface | 120 / 241 | 87.78 | 67.54 | 1.30x |
| Neural Bermudan/dividend loss | 30 / 61 | 24.62 | 23.36 | 1.05x |

The [timing](../benchmarks/fd_phase14_tensor_step_timing.json),
[scaling](../benchmarks/fd_phase14_tensor_step_scaling.json), and
[event](../benchmarks/fd_phase14_tensor_step_events.json) reports retain every
sample. Adjoint/inference [controls](../benchmarks/fd_phase14_tensor_step_controls.json)
range from 11.15-12.03 ms and 6.23-6.53 ms respectively; this change makes no
speedup claim for those paths.

The [profile](../benchmarks/fd_phase14_tensor_step_profile.json) confirms that
30/61 application calls fall **62 to 31**, packs **624 to 190**, and unpacks
**359 to 173**. Domain and residual checks each remain at **34**, including
the interpolation solve. Forward/backward ATen calls fall **2,453/4,298 to
2,228/4,081**. The [graph measurements](../benchmarks/fd_phase14_tensor_step_structure.json)
fall **937 to 906** nodes at 30/61 and **1,687 to 1,626** at 60/121. Saved tensor
counts fall **1,067 to 782** and **1,967 to 1,412**; byte counts including
repeated saves fall **490,430 to 398,870** and **1,815,470 to 1,459,670**.
Those byte counts are not unique live storage. Cold peak RSS growth falls by
only **0.71/0.48 MiB** in these runs, so memory claims remain limited to these
measurements.

Complete default-Thomas Hessian-vector workloads improve from **32.09/35.54
to 29.25/31.27 ms** for 10/31 surface/neural cases and **72.82/73.60 to
63.94/64.25 ms** at 30/61. The paired curvature reports
([off](../benchmarks/fd_phase14_tensor_step_curvature0.json),
[on](../benchmarks/fd_phase14_tensor_step_curvature1.json)) also exercise automatic
PCR with Thomas disabled. Those controls are mixed: 30/61 surface changes
203.97 to 195.60 ms, but neural changes 207.33 to 217.64 ms. No uniform PCR
curvature gain is claimed. Explicit `kernels='pcr'` retains its original step.
Across **752** raw curvature samples, maximum price/gradient/HVP differences
are **1.24e-14 / 5.68e-14 / 1.78e-13**, and the largest central-difference HVP
discrepancy remains **1.63e-8**.

The opt-in [source archive](../benchmarks/fd_phase14_tensor_step_source.zip),
[commands](../benchmarks/fd_phase14_tensor_step_commands.json), and
[validation](../benchmarks/fd_phase14_tensor_step_validation.json) retain **3,934**
fixed-reference price/VJP comparisons including historical controls, with
maximum differences **1.24e-14 / 3.03e-13**. Directly matched step on/off
validation contains **856** samples with identical prices and a maximum VJP
difference of **1.33e-13**. All **77** expected source files match the archive,
all **77** regenerated overlay files match, and pinned QuantLib remains clean.
There is no new overlay patch: the existing guarded scheme hook dispatches
the full-mode composition. The opt-in archive SHA-256 is
`40184d90da68f22c94840a7dfd54342b58f99d098b8670817a4ac75ac9a86860`.

The remaining profile still places about **3.55 ms** in coefficient assembly,
including **1.23 ms** in sampling, and **3.64 ms** inside the full tensor-step
scope. These are nested instrumented times, not additive wall-clock costs.
Next investigate coefficient materialization, sampling, and the remaining
step operations while preserving full elimination-state derivatives and
original fallbacks. The financial API remains CPU-only and unbatched.

## Phase 13: reuse of immutable spatial bases

`FdmBlackScholesOp` constructs its first- and second-derivative operators once
and declares both members `const`. Each operator now owns a small cache of
their packed tensors. `QLT_BASIS_CACHE=1` enables reuse by default;
`QLT_BASIS_CACHE=0` restores packing on every assembly. The benchmark level
`phase13` selects this path, while `phase11` retains the preceding configuration
including the unconditional Phase 12 mixed-AD correction.

Only bases consisting entirely of passive C++ scalar constants enter the
cache. Tensor-valued bases retain their original packing and graph connections,
even when `requires_grad=False`. Output aliases and transformed inputs bypass
reuse. Inference-mode tensors are repacked if the owner later executes outside
inference mode. The cache has the same lifetime as its immutable operators;
the financial API creates new owners for every price and releases them on
exceptions. Autograd saves its own tensor references, so reverse execution
remains valid after the operator is destroyed. There are no global pointer
keys, coefficient detaches, or in-place tensor changes.

The generated `basis-cache.patch` keeps storage and reuse changes separate
from compatibility and numerical patches. The assembly expression, upstream
mesh and financial objects, solve predicates, events, and derivative contracts
are unchanged. Both the opt-in and selected default builds pass **934 tests**,
including **23 new checks**
for fresh graphs, changing meshes/markets, operator lifetime, engine failures,
active/aliased/view bases, derivatives through third order, incoming VJP duals,
forward AD, and transforms. The targeted and cache-disabled runs each pass
**120 tests**, also repeated successfully after selecting the default.

### Selected measurements and reproducibility

The selected run measures `phase13` before `phase11`, reversing the opt-in
order. The [timing](../benchmarks/fd_phase13_selected_timing.json),
[scaling](../benchmarks/fd_phase13_selected_scaling.json), and
[event](../benchmarks/fd_phase13_selected_events.json) reports retain all samples.
The worker policy and timing boundaries are the same as the opt-in experiment.

| Workload | Steps / nodes | Cache off ms | Selected ms | Speedup |
|---|---:|---:|---:|---:|
| Surface full AD | 30 / 61 | 21.06 | 18.76 | 1.12x |
| Surface adjoint | 30 / 61 | 12.08 | 11.94 | 1.01x |
| Surface inference | 30 / 61 | 6.66 | 6.41 | 1.04x |
| Neural full AD | 30 / 61 | 18.87 | 19.13 | 0.99x |
| Neural adjoint | 30 / 61 | 12.35 | 11.98 | 1.03x |
| Neural inference | 30 / 61 | 7.06 | 6.53 | 1.08x |
| Surface full AD | 60 / 121 | 40.15 | 37.00 | 1.09x |
| Surface adjoint | 60 / 121 | 22.74 | 21.48 | 1.06x |
| Surface inference | 60 / 121 | 14.11 | 12.25 | 1.15x |
| Surface full AD | 120 / 241 | 87.17 | 83.89 | 1.04x |
| Surface adjoint | 120 / 241 | 51.28 | 47.25 | 1.09x |
| Surface inference | 120 / 241 | 35.78 | 30.00 | 1.19x |
| Neural Bermudan/dividend full AD | 30 / 61 | 25.60 | 24.42 | 1.05x |
| Neural Bermudan/dividend adjoint | 30 / 61 | 18.20 | 17.06 | 1.07x |
| Neural Bermudan/dividend inference | 30 / 61 | 9.35 | 8.06 | 1.16x |

The neural full-mode total is slightly slower in this repeat despite a lower
forward median (10.64 to 10.33 ms). Together with the earlier timing variation,
this rules out a claim of uniform full-mode acceleration. Adjoint gains at
30/61 are also modest. The strongest repeated benefit is larger-grid inference;
neither curvature nor general memory savings are claimed.

The selected [source archive](../benchmarks/fd_phase13_source.zip),
[commands](../benchmarks/fd_phase13_commands.json), and
[validation](../benchmarks/fd_phase13_validation.json) retain **4,806**
fixed-reference price/VJP comparisons across the opt-in and selected reports,
with maximum differences **1.24e-14 / 3.03e-13**. The **1,410** selected
cache-on/off samples have identical prices and VJPs. All **76** expected source
files and **77** regenerated overlay files match, and pinned QuantLib remains
clean. The selected archive SHA-256 is
`2ed614edacfc41e20e29bedde2f83f5bb8f4a5f3970ea0849262fb9184eed36b`.
Only the cache default, its test expectation, and the benchmark alias differ
from the archived opt-in source; numerical code is unchanged.

### Opt-in experiment and remaining costs

Fresh HighQoS workers use one intra/inter-op thread, 32 warmups, and 15 timed
samples. Every price/VJP total includes construction, forward, backward, and
cleanup. Inference includes construction, forward, and cleanup. The initial
[timing experiment](../benchmarks/fd_phase13_basis_timing.json) measures 30/61
surface full-mode **20.18 to 18.41 ms**, adjoint **12.33 to 11.39 ms**, and
inference **6.92 to 6.30 ms**. Neural timings change from **21.28/12.47/6.88 ms**
to **18.77/12.02/6.71 ms**, respectively. The
[scaling experiment](../benchmarks/fd_phase13_basis_scaling.json) measures 120/241
surface full-mode **88.66 to 83.53 ms**, adjoint **50.82 to 46.21 ms**, and
inference **34.05 to 29.49 ms**. Gains are workload dependent: the
[Bermudan/dividend adjoint](../benchmarks/fd_phase13_basis_events.json) changes
only **18.03 to 17.91 ms**, while inference changes **8.86 to 7.82 ms**.

The [profile](../benchmarks/fd_phase13_basis_profile.json) confirms **one basis
packing operation and 32 reuses**, replacing 33 packs at 30/61. Forward ATen
calls fall **2,837 to 2,453** in full mode and **2,400 to 2,016** in adjoint and
inference modes; backward counts remain unchanged. Instrumented assembly
falls **5.13 to 3.25 ms** in full mode, **4.69 to 3.21 ms** in adjoint mode, and
**3.75 to 2.87 ms** in inference. These nested scope times are not additive
or substitutes for uninstrumented end-to-end timings.

The [structure measurements](../benchmarks/fd_phase13_basis_structure.json) keep
full-mode graphs at **937/1,687** nodes and adjoint graphs at **486/816** for
30/61 and 60/121. Repeated-save byte counts also stay unchanged: those counts
count the same cached tensor on each save. Cold peak RSS changes range from
**+0.22 to -0.55 MiB** across these cases; no general memory-saving claim is
made from these small differences.

The paired curvature controls
([cache off](../benchmarks/fd_phase13_basis_curvature_pair0.json),
[cache on](../benchmarks/fd_phase13_basis_curvature_pair1.json)) retain complete
30/61 surface/neural Hessian-vector workloads near **72 ms** with Thomas and
**192-200 ms** with PCR. The earlier independent vector-only controls
([off](../benchmarks/fd_phase13_basis_curvature0.json),
[on](../benchmarks/fd_phase13_basis_curvature1.json)) range from **69.94 to
72.46 ms**. These measurements establish no distinct curvature speedup.
Across **564** raw curvature samples, maximum price/gradient/HVP differences
are **1.24e-14 / 5.68e-14 / 7.11e-14**; the largest central-difference HVP
discrepancy remains **1.63e-8**.

The opt-in [source archive](../benchmarks/fd_phase13_basis_source.zip),
[commands](../benchmarks/fd_phase13_basis_commands.json), and
[validation](../benchmarks/fd_phase13_basis_validation.json) retain **3,750**
fixed-reference price/VJP comparisons including historical controls. Maximum
differences are **1.24e-14 / 3.03e-13**. The **1,432** directly matched cache
on/off samples have identical prices and VJPs. All **76** expected source files
match the archive and all **77** regenerated overlay files match the build;
the pinned QuantLib checkout is clean. The opt-in archive SHA-256 is
`5b3a252bf7961d24150d9bb244f5fc1036cd2e20c9f842728f68f70f8598acbb`.

An additional `QLT_PASSIVE_CONSTANTS=0` diagnostic passes **22 checks** and
fails fake tracing. The same failure reproduces with the cache disabled and
with the existing assembly fake-tracing test, unchanged from Phase 12. With
passive constants disabled, the mesher's scalar comparisons attempt to extract
fake tensor values. This diagnostic combination remains unsupported; ordinary
fake tracing passes in the full suite. The
[negative controls](../benchmarks/fd_phase13_fake_tensor_constants_control.json)
and failed diagnostic log are retained rather than counted as passing checks.

The next measured targets identified here were array/operator bookkeeping and
the full-mode solve boundary: after caching, instrumented full-mode band solves
still cost **3.46 ms** at 30/61, while assembly costs **3.25 ms**, including
**1.11 ms** of coefficient sampling. Phase 14 addresses the repeated step
conversions; coefficient sampling and further gradient work remain targets
before broader financial-object reuse. The pricing
API remains CPU-only and unbatched; this phase makes no GPU claim.

## Phase 12: incoming forward tangents through full-mode VJPs

An additional mixed-AD check exposed a gap in the earlier native full-mode
assembly, application, and PCR-stage backward paths. With a primal forward
computed normally, supplying forward-mode dual VJP weights to
`torch.autograd.grad(..., create_graph=False)` lost their tangents in native
buffer reads. Ordinary primal gradients remained correct, and the same cases
with `create_graph=True` already used differentiable tensor formulas.

The new independent tests reproduce **18 failures and 12 passes** on the
selected Phase 11 binary. The [negative-control record](../benchmarks/fd_phase12_backward_duals_before.json)
preserves the binary/source/test/log hashes and failing case names. Coverage
includes each of four PCR-stage output adjoints, complete solves against dense
references, full operator boundaries against forced original operations, and
both auto/PCR financial paths against the original scalar engine.

The correction routes incoming dual weights through the existing tensor
backward formulas even when reverse graph recording is disabled. Thomas
arithmetic already handled this case. Ordinary native VJPs, first-order-only
adjoint contracts, forward pricing arithmetic, and the Phase 11 optimizations
are unchanged. All **30 mixed-AD tests** now pass, along with the **911-test
full suite** and **10 configured fallback tests**. The new test file is
unchanged between the failing and passing runs.

The [timing controls](../benchmarks/fd_phase12_backward_duals_timing.json) retain
Phase 11 performance: surface/neural adjoints take **12.30/12.38 ms**, inference
**6.95/7.10 ms**, and full-mode price/VJPs **18.91/19.26 ms** at 30/61.
The [curvature controls](../benchmarks/fd_phase12_backward_duals_curvature.json)
measure complete surface/neural Hessian-vector workloads at **72.37/71.99 ms**
with the selected Thomas path and **210.69/201.90 ms** with PCR. These are
regression controls, not a new performance claim for the correctness fix.
Workers retain the HighQoS, one-thread, 32-warmup, 15-sample policy.

The [source archive](../benchmarks/fd_phase12_source.zip),
[commands](../benchmarks/fd_phase12_commands.json), and
[validation](../benchmarks/fd_phase12_validation.json) preserve **2,580**
fixed-reference price/VJP comparisons against the selected Phase 11 reports,
with maximum differences **3.55e-15 / 6.39e-14**. Cross-revision curvature
validation includes **752** raw samples from this revision and selected
Phase 10, with maximum Hessian-vector difference **1.78e-13**. The largest
current central-difference discrepancy is 1.63e-8. All **75** expected source
files match the archive, all **76** regenerated overlay files match the build,
and the pinned upstream checkout is clean. The archive SHA-256 is
`440cf87616dc2404e2c9b2d2d208081a14599f48f5e2993172ac4c85d1b32322`.

The next performance target identified here was operator packing. The pinned
`FdmBlackScholesOp` declares its first- and second-derivative spatial operators
`const`, constructing them once while repeatedly packing them during assembly.
This motivated Phase 13's reuse within one price, with explicit lifetime/graph
checks and fresh state across prices, before broader engine caching.

## Phase 11: native first-order assembly and steps

The selected build enables `QLT_ADJOINT_ARITHMETIC_FORWARD=1` and
`QLT_STEP_NATIVE_ARITHMETIC=1` by default. Setting either switch to zero restores
the corresponding original forward/backward implementation. Supported steps
remain one-dimensional, with fixed settings and empty boundary callbacks;
other settings, operator types, and boundaries retain the existing fallbacks.
All original coefficient/state tensors remain saved, and only fresh outputs
and scratch buffers are written. The first-order-only adjoint contract and
full-mode graph-connected higher derivatives are preserved. No stock QuantLib
library is linked and no new overlay patch is introduced.

The selected rebuild passes **881 tests** with both switches unset,
**139 fallback tests** with both disabled, and **108 tests** with only the
new step path enabled. A new near-singular regression verifies that a finite
forward result can still require rejecting the transpose solve: its nonzero
pivot is about 2.47e-32, inside QuantLib's `close(pivot, 0)` threshold. Exact
upstream forward comparisons cover multiple sizes/scales, reflected endpoints,
readonly views/aliases, all coefficient/state VJPs, and original failure cases.

Fresh HighQoS workers use one intra/inter-op thread, 32 warmups, and 15 timed
samples. The selected run measures Phase 11 before Phase 10, reversing the
pilot order. Adjoint timings include forward, loss/VJP, and cleanup; inference
includes forward and cleanup.

| Workload | Steps / nodes | Phase 10 ms | Phase 11 ms | Speedup |
|---|---:|---:|---:|---:|
| Constant volatility adjoint | 30 / 61 | 23.09 | 10.42 | 2.22× |
| Surface adjoint | 30 / 61 | 23.76 | 11.98 | 1.98× |
| Neural loss adjoint | 30 / 61 | 24.35 | 12.73 | 1.91× |
| Surface inference | 30 / 61 | 15.83 | 6.75 | 2.35× |
| Neural inference | 30 / 61 | 15.65 | 7.26 | 2.16× |
| Surface adjoint | 60 / 121 | 63.79 | 24.14 | 2.64× |
| Surface inference | 60 / 121 | 44.49 | 14.55 | 3.06× |
| Surface adjoint | 120 / 241 | 207.79 | 50.07 | 4.15× |
| Surface inference | 120 / 241 | 148.88 | 34.36 | 4.33× |
| Neural Bermudan adjoint, coincident exercise/dividend | 30 / 61 | 29.27 | 18.06 | 1.62× |
| Neural Bermudan inference | 30 / 61 | 17.58 | 8.53 | 2.06× |

The selected [timing](../benchmarks/fd_phase11_selected_timing.json),
[scaling](../benchmarks/fd_phase11_selected_scaling.json), and
[event](../benchmarks/fd_phase11_selected_events.json) reports retain every sample.
Full-mode [controls](../benchmarks/fd_phase11_selected_controls.json) remain near
their prior times: surface 19.38 → 19.23 ms and neural 19.77 → 19.15 ms. These
paths do not receive the Phase 11 arithmetic changes; no full-mode speedup is
claimed. The library does not change process QoS.

The combined opt-in [profile](../benchmarks/fd_phase11_step_profile.json) measures
adjoint diffusion forward **5.10 → 0.365 ms**, step forward **6.58 → 1.30 ms**,
and step backward **2.93 → 0.754 ms**. Operator assembly still costs 4.53 ms
including sampling and packing. These nested instrumented scopes are not
uninstrumented timing gains. Forward ATen calls change 2,466 → 2,400 and
backward calls 2,459 → 2,723: native scratch tensors add visible ATen allocation
calls while eliminating more expensive scalar-adapter work. The
[structure report](../benchmarks/fd_phase11_step_structure.json) keeps graph counts
at 486/816 for 30/61 and 60/121. Cold peak RSS changes by +0.12/-0.16 MiB;
this phase makes no memory-saving claim.

The selected [source archive](../benchmarks/fd_phase11_source.zip),
[commands](../benchmarks/fd_phase11_commands.json), and
[validation](../benchmarks/fd_phase11_validation.json) preserve **6,532**
fixed-reference price/VJP comparisons across both experiments and selection,
including every raw warmup/timed sample. Maximum price/VJP differences are
**7.11e-15 / 1.35e-13**. All **74** expected source files match the selected
archive; all **76** freshly generated overlay files match the build. QuantLib
remains clean at v1.40 commit `06edddbdb85f006c43a449b5fc846a00c67477f3`.
The selected archive SHA-256 is
`527620bcd606465ede622f9f64f2baa530d1029e786509931ee0d9af9c1dcbcd`.

### Experiments and preserved arithmetic

The initial `QLT_ADJOINT_ARITHMETIC_FORWARD=1` experiment reuses the
verified native diffusion assembly and stencil forward arithmetic inside
the first-order adjoint boundaries. It avoids constructing passive operator
copies there. Original input tensors remain saved, and the existing VJPs and
first-order-only contract are unchanged. Full-mode higher derivatives keep
their selected Phase 10 path. This experiment initially kept the switch at
zero by default while measurement and verification proceeded.

The assembly experiment passes **830 tests**, **163 targeted tests**, and
**51 explicit fallback tests**, including independent all-input VJPs,
readonly views/aliases, first-order contract rejection, and exact forward
comparisons at multiple sizes/scales. The diffusion test probe now accepts an
explicit mode zero to force the original QuantLib mult/axpyb reference,
independently of both full-mode and adjoint optimizations. Financial dispatch
and inference are checked against the scalar original engine. Fresh HighQoS
workers with 32 warmups and 15 samples measure 30/61 surface adjoint
**22.88 → 19.28 ms**, neural adjoint **24.32 → 19.91 ms**, and surface inference
**14.70 → 11.35 ms**. At 120/241, surface adjoint improves
**204.35 → 150.96 ms** and inference **151.22 → 99.05 ms**. Full-mode controls
remain near their previous times. Graph counts are unchanged at 486/816 for
30/61 and 60/121; cold peak RSS increases by 0.26/0.57 MiB, so this version
makes no memory-saving claim.

The [assembly source archive](../benchmarks/fd_phase11_arithmetic_source.zip),
[commands](../benchmarks/fd_phase11_arithmetic_commands.json), and
[validation](../benchmarks/fd_phase11_arithmetic_validation.json) retain **2,300**
fixed-reference pricing/VJP comparisons, including all raw samples. Maximum
price/VJP differences are 7.11e-15 / 1.35e-13. All 71 expected source files and
76 regenerated overlay files match, and the pinned upstream checkout is clean.
Raw [timing](../benchmarks/fd_phase11_arithmetic_timing.json),
[scaling](../benchmarks/fd_phase11_arithmetic_scaling.json),
[event](../benchmarks/fd_phase11_arithmetic_events.json),
[structure](../benchmarks/fd_phase11_arithmetic_structure.json), and
[profile](../benchmarks/fd_phase11_arithmetic_profile.json) reports remain separate.

The second experiment, `QLT_STEP_NATIVE_ARITHMETIC=1`, executes the existing
one-dimensional, fixed-setting, empty-boundary step directly on native CPU
values inside the first-order boundary. It preserves the original Douglas,
implicit Euler, and explicit Euler expression ordering, reflected application
endpoints, scaled splitting recurrence, and transpose-solve VJP. The forward
retains the original inverse-pivot first check and direct-pivot later checks;
backward retains the compact solver's `close(pivot, 0)` rule. The native helper
uses local strict floating-point compilation and carries pinned upstream
source attribution/license notices. All inputs remain saved; only fresh
outputs and scratch buffers are written. Unsupported settings, boundaries,
operator types, and transform inputs retain the existing paths.

The combined opt-in version passes 880 tests plus 107 fallback tests. Its
[source archive](../benchmarks/fd_phase11_step_source.zip),
[commands](../benchmarks/fd_phase11_step_commands.json), and
[validation](../benchmarks/fd_phase11_step_validation.json) preserve 4,420 of the
price/VJP comparisons above. Its [timing](../benchmarks/fd_phase11_step_timing.json)
report separately measures the baseline, assembly-only, and combined versions;
[controls](../benchmarks/fd_phase11_step_controls.json),
[scaling](../benchmarks/fd_phase11_step_scaling.json), and
[events](../benchmarks/fd_phase11_step_events.json) remain archived. Selection
adds the tiny-pivot regression and an unknown-step-kind guard, and avoids
redundant readable conversions before native backward. Numerical formulas are
unchanged; historical source differences are explicitly labelled.

Remaining measured costs include array/operator packing, coefficient sampling,
financial-object construction, and autograd bookkeeping. The financial API
remains a single-contract CPU API. Complete contract-batch measurements are
still needed before making batching or GPU pricing claims.

## Phase 10: differentiated CPU elimination

The selected `kernels="auto"` path enables `QLT_THOMAS_ARITHMETIC=1` by default.
After the same domain check, it solves the already assembled compact bands
with the division-based recurrence from QuantLib
v1.40 `TridiagonalOperator::solveFor`. The same residual check and original
QuantLib fallback remain. Unsupported native cases retain PCR; explicit PCR,
the vectorized/scalar original solvers, and first-order adjoints are unchanged.
Both Thomas switches default to one. The selected rebuild passes **810 tests**
with both environment variables unset, plus **233 tests** with both disabled.
Setting `QLT_THOMAS_ARITHMETIC=0` restores the previous guarded PCR selection;
setting `QLT_THOMAS_VECTOR_BACKWARD=0` retains the serial higher-derivative
fallback without disabling the native ordinary forward/backward.

`cpp/thomas_arithmetic.cpp` preserves the upstream forward expression order,
copyright attribution, and license. The compact recurrence differs from the
inverse-pivot ordering in `TripleBandLinearOp::solve_splitting`, just as the
existing optimized PCR path does; financial equivalence is checked separately.
No QuantLib overlay change or stock QuantLib linkage is introduced.

Forward elimination stores pivots, upper factors, and intermediate RHS values,
then performs back substitution. Ordinary backward explicitly reverses those
operations, including factor, pivot, coefficient, and RHS dependencies. It does
not use the implicit transpose-solve VJP. The revised operation returns its
pivots, factors, and intermediate RHS values as auxiliary outputs, preserving
their gradient edges as well as the original coefficient/RHS tensors. With
`QLT_THOMAS_VECTOR_BACKWARD=1`, three differentiable bidiagonal recurrences
reverse back substitution, RHS elimination, and pivot/factor elimination.
Their VJPs include **all four output adjoints**, which is necessary for the
recursive higher-derivative chain. Forward duals, function transforms, and fake
tensors use functional recurrence in the private probe; the financial dispatch
retains its previous PCR path for these cases. Only fresh output/scratch
buffers are mutated. Broadcasting happens outside the custom operation, so
autograd reduces shared-band gradients correctly.

The first prototype reconstructed the saved state and recorded each reverse
scalar recurrence separately for higher derivatives. Although correct, the
complete 30/61 surface forward/gradient/Hessian-vector product took
**1,115 ms versus 204 ms** with Phase 9. That version is rejected as the default
and preserved in the [initial source archive](../benchmarks/fd_phase10_thomas_source.zip),
[curvature report](../benchmarks/fd_phase10_thomas_curvature.json),
[commands](../benchmarks/fd_phase10_thomas_commands.json), and
[validation](../benchmarks/fd_phase10_thomas_validation.json). The functional
reconstruction remains an explicit fallback with the vector-backward switch
disabled. It must not be mistaken for the revised higher-derivative path.

The private `_thomas_arithmetic` probe supports broadcast batches and multiple
RHS columns on CPU, with tensor fallback elsewhere. Its checked mode requires
finite inputs/output and finite nonzero pivots. It does not reproduce the
original compact solver's absolute `close(pivot, 0)` threshold outside the
financial auto domain; the original public API and fallback retain that policy.
The auto domain's scale/dominance bounds exclude those tiny-pivot cases.

The revised opt-in build passes **810 tests**, plus **84 tests** with the
serial higher-derivative fallback. Coverage includes independent dense values
and all VJPs, bitwise compact-upstream forward agreement, partial-input
gradcheck/gradgradcheck, auxiliary-output VJPs, mixed third derivatives and
incoming weights, aliases, empty/shared/cross-broadcast batches, readonly
views, extreme scales, forward AD (including incoming VJP tangents), transforms,
fake tracing, CUDA fallback, actual financial dispatch, Hessians, and unchanged
auto-domain rejection/original-solver fallback tests.

Fresh HighQoS workers, one intra/inter-op thread, 32 warmups, and 15 samples give
the following selected medians. This run measures Phase 10 before Phase 9,
reversing the opt-in pilot's order. Pricing includes forward, VJP, and cleanup;
curvature includes forward, graph-building gradient, Hessian-vector product,
and cleanup. Every curvature sample is checked across strategies, and each
worker also compares its Hessian-vector product with central differences of
the first gradient.

| Full-AD workload | Steps / nodes | Phase 9 ms | Phase 10 ms | Speedup |
|---|---:|---:|---:|---:|
| Constant volatility price/VJP | 30 / 61 | 23.12 | 17.93 | 1.29× |
| Surface price/VJP | 30 / 61 | 24.84 | 19.27 | 1.29× |
| Neural pricing loss | 30 / 61 | 26.00 | 20.29 | 1.28× |
| Surface price/VJP | 60 / 121 | 51.93 | 38.62 | 1.34× |
| Surface price/VJP | 120 / 241 | 118.98 | 88.97 | 1.34× |
| Neural Bermudan loss, coincident exercise/dividend | 30 / 61 | 30.84 | 26.40 | 1.17× |
| Surface Hessian-vector workload | 10 / 31 | 84.18 | 31.92 | 2.64× |
| Neural Hessian-vector workload | 10 / 31 | 74.82 | 32.12 | 2.33× |
| Surface Hessian-vector workload | 30 / 61 | 202.92 | 73.01 | 2.78× |
| Neural Hessian-vector workload | 30 / 61 | 206.82 | 72.76 | 2.84× |

See the selected [timing](../benchmarks/fd_phase10_selected_timing.json),
[scaling](../benchmarks/fd_phase10_selected_scaling.json),
[events](../benchmarks/fd_phase10_selected_events.json), and
[curvature](../benchmarks/fd_phase10_selected_curvature.json) reports. The earlier
[opt-in timing/control](../benchmarks/fd_phase10_vector_timing.json),
[scaling](../benchmarks/fd_phase10_vector_scaling.json),
[event](../benchmarks/fd_phase10_vector_events.json), and
[curvature](../benchmarks/fd_phase10_vector_curvature.json) reports remain separate.
First-order adjoint controls remain around 23–24 ms and use their original
implementation. Timing variation changes individual ratios between runs;
both orderings show gains on every paired optimized workload.

The opt-in [structure report](../benchmarks/fd_phase10_vector_structure.json)
reduces graph nodes **1,241 → 937** at 30/61 and **2,325 → 1,687** at 60/121.
Cold-process peak RSS growth changes from **20.65 → 19.30 MiB** and
**26.73 → 22.45 MiB**, respectively. The
[profile](../benchmarks/fd_phase10_vector_profile.json) reduces forward ATen calls
**4,095 → 2,837** and backward calls **6,870 → 4,298**. The total instrumented
native Thomas forward scope is 0.265 ms over 34 solves; its surrounding band
solve scope still costs 2.89 ms. Operator assembly costs 4.50 ms, including
1.15 ms for coefficient sampling. These nested profiler durations are not
uninstrumented wall-time savings and must not be added together.

Standalone CPU probe measurements use the same HighQoS/warmup/sample policy,
unchecked device-resident inputs, and include forward, all-input VJP, and
cleanup. Independent dense value/VJP validation accompanies every shape.
The [independent-band](../benchmarks/fd_phase10_vector_batch_independent.json) and
[shared-band](../benchmarks/fd_phase10_vector_batch_shared.json) reports give:

| Batch / nodes / RHS | Bands | PCR ms | Native Thomas ms |
|---|---|---:|---:|
| 1 / 61 / 1 | independent | 0.250 | 0.094 |
| 256 / 257 / 1 | independent | 27.84 | 5.59 |
| 256 / 257 / 4 | independent | 31.17 | 6.09 |
| 256 / 257 / 1 | shared | 7.43 | 5.77 |
| 256 / 257 / 4 | shared | 29.67 | 8.53 |

The private probe is an investigation tool; the financial API remains one
contract on CPU, and public `tridiagonal_solve_pcr` retains PCR. Shared CPU
inputs may be materialized inside the explicit boundary, with autograd
performing the outer broadcast reductions. The earlier Phase 10
[CUDA diagnostic](../benchmarks/fd_phase10_thomas_cuda.json), on the archived
initial prototype revision, measures independent 256/257/4 PCR at 9.68 ms
resident and 11.94 ms including transfers. It does not establish a GPU gain
over the stronger CPU baseline. Historical Phase 8 GPU ratios must not be
applied to this revision, and no GPU financial-pricing claim is made.

The selected [source archive](../benchmarks/fd_phase10_source.zip),
[commands](../benchmarks/fd_phase10_commands.json), and
[validation](../benchmarks/fd_phase10_validation.json) preserve **3,844**
fixed-reference price/VJP comparisons across both prototypes and selection,
including every raw warmup/timed sample. Maximum price/VJP differences are
**2.31e-14 / 2.06e-13**. The selected curvature report contains **376** paired
raw comparisons with maximum Hessian-vector difference **1.78e-13**; its
largest central-difference discrepancy is **1.63e-8**. Each of the two archived
curvature reports adds another 376 comparisons. The validation also retains
29 independent tensor checks across the archived CPU/CUDA reports. All **70**
expected source files match the selected archive; all **76** regenerated
overlay files match the build. QuantLib remains clean at pinned v1.40 commit
`06edddbdb85f006c43a449b5fc846a00c67477f3`. The selected archive SHA-256 is
`6ec1eb61b1b011b9aa5ece9903694d751f78c78c3eb16520868d86aba0b350f9`.

The revised opt-in [source archive](../benchmarks/fd_phase10_vector_source.zip),
[commands](../benchmarks/fd_phase10_vector_commands.json), and
[validation](../benchmarks/fd_phase10_vector_validation.json) preserve 2,694
of those pricing/VJP comparisons. Structure/profile/batch measurements above
use this revision; selection changes defaults, comments, dispatch-test
expectations, and documentation with identical numerical arithmetic. Historical
source mismatches are explicitly labelled in the selected validation.

Remaining work is to reduce array/operator bookkeeping and autograd overhead, investigate the
passive-copy costs still present in first-order adjoint mode, and measure
complete contract batches before making batching or GPU pricing claims.

The next investigation starts from a fresh
[adjoint/inference profile](../benchmarks/fd_phase11_baseline_profile.json) on the
selected Phase 10 build, with [validation](../benchmarks/fd_phase11_baseline_validation.json)
against the same source archive. At 30/61, adjoint-mode operator assembly costs
9.17 ms instrumented, including 4.82 ms inside diffusion-adjoint forward.
Step-adjoint forward/backward scopes cost 5.73/3.21 ms. Inference still spends
4.56 ms in diffusion-adjoint forward and 5.71 ms in step-adjoint forward.
Inspection shows passive band/array copies and original QuantLib calls inside
these boundaries. The next reversible experiment is to reuse the verified
direct assembly arithmetic inside the first-order boundary, preserving its
derivative contract, then measure whether native step arithmetic is warranted.
These are profiling targets, not claimed uninstrumented savings.

## Phase 9: solver checks and full-AD operator arithmetic

The selected build enables native CPU solver-selection predicates and elementary
assembly/application arithmetic by default. The financial engine, mesh, event
ordering, conservative PCR domain, residual thresholds, and original-solver
fallbacks are preserved. Full mode differentiates the executed elementary
operations, including higher derivatives; it does not substitute the separate
first-order solve adjoint. No stock double-based QuantLib library is linked.

| Switch (default 1) | Setting to 0 |
|---|---|
| `QLT_NATIVE_SOLVE_CHECKS` | Use the original tensor admissibility/residual predicates |
| `QLT_ASSEMBLY_ARITHMETIC` | Use the original full-AD tensor assembly expressions |
| `QLT_OPERATOR_ARITHMETIC` | Use the original full-AD tensor band application |
| `QLT_ARITHMETIC_NATIVE_FORWARD` | Execute QuantLib's original forward routines inside the elementary boundaries, retaining their VJPs |

The selected rebuild passes **726 tests** with these environment variables
unset, plus **213 tests** with all four explicitly disabled. Tests cover
independent values/VJPs, numerical first/second differences, mixed third
derivatives and incoming weights, coefficient/RHS aliases, reflected endpoints,
readonly views, forward AD, transforms/tracing, financial events, and stock
QuantLib references. The first-order adjoint contract remains unchanged.

### Matched pricing measurements

Fresh HighQoS workers use one intra/inter-op thread, 32 warmups, and 15 timed
samples. Times include forward, backward/loss VJP, and cleanup. The selected
run reverses worker order relative to the opt-in pilot, running Phase 9 before
Phase 8. The library does not change process QoS.

| Full-AD workload | Steps / nodes | Phase 8 ms | Phase 9 ms | Speedup |
|---|---:|---:|---:|---:|
| Constant volatility price/VJP | 30 / 61 | 42.16 | 22.55 | 1.87× |
| Surface price/VJP | 30 / 61 | 41.09 | 24.53 | 1.68× |
| Neural pricing loss, 17 parameters | 30 / 61 | 43.15 | 26.68 | 1.62× |
| Surface price/VJP | 60 / 121 | 86.99 | 51.33 | 1.69× |
| Neural Bermudan loss, coincident exercise/dividend | 30 / 61 | 50.52 | 32.09 | 1.57× |

The [selected timing](../benchmarks/fd_phase9_selected_timing.json),
[scaling](../benchmarks/fd_phase9_selected_scaling.json), and
[event](../benchmarks/fd_phase9_selected_events.json) reports retain all samples.
The same binary also measures the Phase 7 surface/neural baselines at
89.80/90.21 ms, giving directly measured 3.66×/3.38× improvements to Phase 9.
The opt-in [controls](../benchmarks/fd_phase9_native_arithmetic_controls.json)
retain similar first-order adjoint and inference times: surface adjoint
23.44 → 22.67 ms and inference 15.51 → 15.74 ms. These paths do not receive
the full-AD arithmetic optimization; no inference speedup is claimed.

The archived opt-in [structure report](../benchmarks/fd_phase9_native_arithmetic_structure.json)
reduces graph nodes **2,319 → 1,241** at 30/61 and **4,423 → 2,325** at 60/121.
Cold-process peak RSS growth changes from **23.15 → 21.45 MiB** and
**28.84 → 26.99 MiB** respectively. The
[profile](../benchmarks/fd_phase9_native_arithmetic_profile.json) reduces forward
ATen calls **11,497 → 4,095** and backward calls **10,108 → 6,870**. Structural
measurements use the archived opt-in revision; the selected revision changes
defaults and dispatch-test expectations, with identical arithmetic.

The [selected source archive](../benchmarks/fd_phase9_source.zip),
[command manifest](../benchmarks/fd_phase9_commands.json), and
[validation](../benchmarks/fd_phase9_validation.json) preserve the result.
Validation includes **7,590 fixed-reference price/VJP comparisons**, including
every raw warmup/timed value across the three Phase 9 investigations and final
selection. Maximum price/VJP differences are **1.42e-14 / 8.88e-14**. All **66**
expected source files match the selected archive; all **76** fresh overlay
files match the build. QuantLib remains clean at pinned v1.40 commit
`06edddbdb85f006c43a449b5fc846a00c67477f3`. Earlier snapshots remain separate and
their source mismatches are explicitly marked historical.

### Experiments and derivative contracts

The first experiment fuses the existing conservative PCR admissibility and
residual predicates in `cpp/solve_checks.cpp`. `QLT_NATIVE_SOLVE_CHECKS=1`
enabled it while the investigation initially kept the default zero. These are
implementation-selection booleans only. The chosen solve still uses the
original graph-connected coefficient tensors. CPU float64 vectors, including
strided, expanded, and lazy-negative views, use native reads; unsupported
wrappers/transforms and devices retain the original tensor predicates.

Bounds, row dominance margin, minimum system size, residual scaling, and
nonfinite comparison behavior match the earlier predicates. The new translation
unit disables floating-point contraction/reassociation (`/fp:strict` on MSVC,
`-fno-fast-math -ffp-contract=off` otherwise), outside the shared precompiled
header. This reproduces the separately rounded ATen operations at decision
boundaries without changing compiler policy for QuantLib or the scalar adapter.
No new QuantLib overlay patch is needed for this experiment.

All **614 tests pass** with the native checks enabled. The 101 new cases cover
exact thresholds and adjacent floating-point values, scaled residual boundaries,
nonfinite arithmetic, random cancellation/scales, readonly tensor views, forward
duals, function-gradient transforms, CUDA fallback, and pricing VJPs/Hessians.
The previous financial, stock-reference, higher-derivative, and event tests also
pass. The [source snapshot](../benchmarks/fd_phase9_checks_source.zip),
[commands](../benchmarks/fd_phase9_checks_commands.json), and
[validation](../benchmarks/fd_phase9_checks_validation.json) preserve this version.

Matched HighQoS workers with 32 warmups and 15 samples measure full-AD surface
pricing **40.93 → 36.76 ms (1.11×)** and neural loss **40.55 → 37.45 ms (1.08×)**
at 30/61, and surface pricing **88.70 → 78.58 ms (1.13×)** at 60/121.
Adjoint controls remain near their earlier times. The
[pilot profile](../benchmarks/fd_phase9_checks_profile.json) reduces forward ATen
calls from 11,497 to 6,499; backward calls stay at 10,108 and graph nodes at
2,319. Instrumented domain/residual time changes from 7.28/5.36 ms to
0.087/0.052 ms; those profiler durations are not uninstrumented timing gains.

The first elementary-operator experiment wraps the original QuantLib assembly
and application forwards, adds a native ordinary VJP, and reconstructs the
elementary chain rule with saved original tensors when recording higher
derivatives. It passes **698 tests**. Its [snapshot](../benchmarks/fd_phase9_arithmetic_source.zip),
[commands](../benchmarks/fd_phase9_arithmetic_commands.json), and
[validation](../benchmarks/fd_phase9_arithmetic_validation.json) retain **2,592**
price/VJP comparisons. Full-AD surface/neural times improve from 43.81/42.83 to
33.06/31.30 ms at 30/61, but the 60/121 surface case takes 79.43 ms versus
76.90 ms with checks alone. Scalar-adapter/operator-copy costs offset the
backward savings at that size. This version is not selected as the default.

The selected forward implementation, initially tested as an opt-in experiment
(`QLT_ARITHMETIC_NATIVE_FORWARD=1`, with
`QLT_ASSEMBLY_ARITHMETIC=1` and `QLT_OPERATOR_ARITHMETIC=1`) evaluates the same
elementary forward arithmetic in fresh tensor buffers. The separate
`cpp/operator_arithmetic.cpp` preserves the pinned QuantLib `setTime`, `mult`,
`axpyb`, and `apply` expression ordering and actual neighbor indices, with
upstream attribution and `LICENSES/QuantLib.txt`. Its compilation also disables
contraction/reassociation locally. This replaces scalar-adapter copies inside
the explicit custom operation; the scalar adapter itself still retains every
coefficient tensor. Saved originals and graph-connected backward expressions
preserve higher derivatives; the full and first-order solve strategies remain
distinct. Duals and active function/fake-tensor transforms use tensor arithmetic.

The expanded targeted suite passes **143 tests**, including bitwise forward
agreement with the upstream routines at scales 1e-150, 1, and 1e150; all band,
RHS, variance, rate, and yield VJPs; mixed third derivatives; aliasing; and
gradient, Jacobian, JVP, vmap, functionalization, and fake-trace fallback. A
preexisting transform failure in vector application was also fixed by lifting
owned constant neighbor indices instead of filling a storage-less tensor
factory result. The opt-in version also passes all **726 tests**, with **2,782**
fixed-reference price/VJP comparisons preserved in its
[source archive](../benchmarks/fd_phase9_native_arithmetic_source.zip),
[commands](../benchmarks/fd_phase9_native_arithmetic_commands.json), and
[validation](../benchmarks/fd_phase9_native_arithmetic_validation.json).

Its independent [ablations](../benchmarks/fd_phase9_native_arithmetic_timing.json)
measure surface full AD at 37.09 ms with native checks alone, 31.61 ms adding
native assembly, 30.64 ms adding native application, and 24.75 ms adding both.
At 60/121, direct arithmetic lowers the wrapped-forward experiment from
76.07 to 52.36 ms. These measurements support selecting both operations with
the direct forward implementation.

The financial engine remains a single-contract CPU API. Unsupported tensor
transforms/devices retain functional paths, and the existing standalone
CPU/CUDA solver coverage does not establish GPU financial pricing. The selected
build emits the existing PyTorch-header `std::optional` C4267 warning when
compiling the strict-predicate translation unit outside the PCH; no project-code
warning or test failure was observed.

Remaining costs include solve recurrence dispatch, coefficient/time-step
bookkeeping, and packing. The profile attributes 7.20 ms to forward band solves
(including 5.10 ms PCR), 4.71 ms to assembly (including 1.32 ms sampling), and
3.12 ms to PCR-stage backward. These are instrumented, nested durations, not
additive wall-clock costs. Next work should compare more work per recurrence
boundary and differentiated original CPU elimination, preserving the full
derivative contract and independent references. The Phase 8 shared single-RHS
dispatch question also remains open; no new batched/GPU performance claim is
made in Phase 9.

## Phase 8: differentiation through fused PCR stages

The implementation executes one elementary PCR elimination stage as a
native CPU operation in `cpp/pcr_stage.cpp`. `QLT_PCR_NATIVE_STAGE=0` disables
it; the default enables it for the measured CPU workloads described below. The financial engine, assembled linear
systems, PCR elimination sequence, conservative auto-selection domain, pivot
checks, and residual fallback are unchanged. No new QuantLib overlay patch is
needed. Phase 7's verified source snapshot and reports remain preserved.

For elimination distance `s`, each row computes `a=-lower/d_left` and
`b=-upper/d_right`, then the same lower/diagonal/upper/RHS updates as the tensor
PCR recurrence. Native forward and ordinary first-order backward fuse these
elementary operations. Backward differentiates the executed recurrence, including
both elimination multipliers and every incoming band/RHS contribution. It does
not use the implicit inverse-matrix VJP of the separate solve adjoint.

Original graph-connected inputs are saved. When recording backward for higher
derivatives, the stage reconstructs its chain rule with functional tensor
expressions. Second and third derivatives therefore remain connected, including
derivatives with respect to incoming VJP weights. Forward duals use the original
tensor recurrence. The scalar adapter does not detach or mutate coefficients;
native loops write only fresh output/gradient buffers. Quotient-rule expressions
avoid explicitly squaring a denominator, which can overflow/underflow at extreme
scales. Lazy-negative views are resolved before native raw-storage reads.

Broadcast expansion stays outside the custom operation, so autograd reduces
VJPs back to shared coefficients. CUDA, forward duals, active function
transforms, and fake/wrapper tensors retain the tensor implementation. In
particular, `torch.func.grad` and `jacrev` reject C++ custom autograd Functions;
the explicit fallback preserves their previously working behavior. Fake-tensor
tracing is checked by executing the resulting graph on real inputs and comparing
every VJP. `vmap` continues to use PyTorch's existing batching fallback for the
public custom operator; a specialized batching rule is not claimed. Transform
tests use unchecked PCR because the existing pivot checks require host booleans.

Single-RHS workloads use native stages only up to **32,768 total rows after
broadcasting**, and only when RHS batches do not duplicate matrix bands. This
is a conservative performance policy, independent of numerical admissibility.
Shared single-RHS matrices and larger single-RHS workloads retain the original
tensor recurrence. Small financial solves and supported multi-RHS batches use
native stages. `QLT_PCR_ROLL=0` and the separate opt-in `QLT_PCR_FUSED=1`
experiment also retain their original tensor formulations.

The selected native build passes **513 tests**. An additional **16 dispatch
tests** pass with stage fusion disabled. Coverage includes independent indexed
stage equations and dense solves, every band/RHS VJP, finite differences,
mixed second/third derivatives, forward AD with and without `no_grad`,
shared/strided/negative views, unused outputs, extreme scales, aliases,
tracing save/load with different runtime shapes, function transforms, dispatch
boundaries, financial events, and unchanged engine/stock references. No
QuantLibAAD or XAD implementation is used.

### Matched pricing measurements

Both configurations use the same selected binary, one intra/inter-op thread,
explicit Windows HighQoS, 32 warmups, and 15 timed samples in separate workers.
Times include forward, VJP/loss backward, and cleanup. The library does not
change caller process QoS. The comparison is Phase 7 versus enabled stage fusion.

| Full-AD workload | Time steps / space nodes | Phase 7 ms | Phase 8 ms | Speedup |
|---|---:|---:|---:|---:|
| Surface price and VJP | 30 / 61 | 93.21 | 42.96 | 2.17× |
| Neural pricing loss, 17 parameters | 30 / 61 | 91.74 | 42.08 | 2.18× |
| Surface price and VJP | 60 / 121 | 198.21 | 85.45 | 2.32× |
| Neural Bermudan loss, coincident exercise/dividend | 30 / 61 | 104.91 | 51.24 | 2.05× |

The raw [pricing](../benchmarks/fd_phase8_selected_timing.json),
[larger-grid](../benchmarks/fd_phase8_selected_scaling.json), and
[event](../benchmarks/fd_phase8_selected_events.json) reports retain every warmup
and timed price/gradient. The earlier [adjoint controls](../benchmarks/fd_phase8_timing.json)
remain close: surface 22.99 → 23.50 ms and neural 23.51 → 24.19 ms. This feature
changes full-AD PCR stages, not the separate adjoint strategy.

The [selected graph/memory workers](../benchmarks/fd_phase8_selected_structure.json)
record 2,319 / 4,423 graph nodes at 30/61 and 60/121, versus 6,768 / 14,346 in
the [archived matched baseline](../benchmarks/fd_phase8_memory.json). Cold process
peak growth changes from 28.73 → 22.27 MiB and 44.24 → 28.62 MiB. These are
process counters including native allocations and allocator retention, not
unique tensor storage. The [selected profile](../benchmarks/fd_phase8_selected_profile.json)
confirms forward/backward ATen calls of 11,497 / 10,108, versus the earlier
27,409 / 27,350.

### Batch selection and rejected regressions

The unrestricted native stage improved B=256, N=257, R=4 CPU solve-plus-VJP
time from 116.73 → 30.91 ms for independent matrices and 96.58 → 28.73 ms for
a shared matrix. Independent-batch cold peak growth fell from 183.40 → 97.33
MiB. These measurements are preserved with the
[pre-dispatch source and validation](../benchmarks/fd_phase8_pre_transform_validation.json).

Single-RHS cases exposed two regressions. Duplicating one matrix across B=256,
N=257 RHS batches changed 7.56 → 36.38 ms and 64.38 → 71.25 MiB. Even with
independent matrices, the larger single-RHS case changed 26.35 → 41.68 ms.
The selected policy retains tensor PCR for both. Independent single-RHS
[crossover measurements](../benchmarks/fd_phase8_crossover_stage1.json) at N=257
show native times 1.75 / 3.58 / 6.82 ms for B=32 / 64 / 128, compared with
[tensor times](../benchmarks/fd_phase8_crossover_stage0.json) 4.82 / 7.39 / 12.41 ms.
The 32,768-row bound is deliberately conservative relative to that host's
measured crossover; it is not a universal hardware optimum. The
[crossover source](../benchmarks/fd_phase8_crossover_source.zip) reconstructs the
pre-cutoff native revision, with every recorded native hash checked in its
[manifest](../benchmarks/fd_phase8_crossover_source.json).

With the selected policy, B=256, N=257, R=4 improves **116.37 → 29.24 ms
(3.98×)** for [independent matrices](../benchmarks/fd_phase8_selected_batch_stage1_independent.json)
and **97.58 → 28.56 ms (3.42×)** for a
[shared matrix](../benchmarks/fd_phase8_selected_batch_stage1_shared.json). The
single-RHS independent case is 27.55 versus 26.76 ms on the same tensor
algorithm. The shared single-RHS case remains somewhat slower: 7.29 versus
8.16 ms, with a reversed-order repeat of 7.59 versus 7.96 ms. Both repeats are
retained; no speedup is claimed for that fallback. Its cold peak growth is
64.80 versus 64.42 MiB, avoiding the unrestricted stage's extra coefficient
storage. Residual dispatch overhead and host variability deserve further
profiling, rather than attributing this difference to a numerical change.

The [selected CUDA comparison](../benchmarks/fd_phase8_selected_cuda.json) measures
9.22 ms resident and 15.42 ms including host transfers for the independent
B=256, N=257, R=4 case, versus the improved CPU's 29.24 ms. These are **3.17× /
1.90×** advantages, substantially smaller than comparisons against the older
CPU implementation. All these kernel timings use unchecked PCR on admissible
systems and include every band/RHS VJP. They do not establish GPU financial
pricing or a universal hardware crossover.

### Reproduction and next work

Historical pilots remain intact, including the initial
[single-system pricing pilot](../benchmarks/fd_phase8_stage_pilot.json), the
[opt-in batch snapshot](../benchmarks/fd_phase8_batch_pilot_source.zip), and the
[enabled-default snapshot before dispatch refinements](../benchmarks/fd_phase8_pre_transform_source.zip).
Their hashes identify their actual source revisions; current defaults must not
be silently attributed to those older measurements. The
[command manifest](../benchmarks/fd_phase8_commands.json),
[selected source snapshot](../benchmarks/fd_phase8_source.zip), and
[final validation](../benchmarks/fd_phase8_validation.json) preserve reproduction
details and source/binary/test/log hashes. The source archive's SHA-256 is
`1e9e553dde1142913ded3760475ab161cb75dfcdb42910fcb60c70c9a060de5b`;
all 59 expected native/build/Python/test/script source files match.

Cross-report validation checks **2,694 fixed-reference price/VJP comparisons**,
including every raw pricing warmup/timed sample. Maximum absolute differences
are 1.42e-14 for prices and 1.24e-13 for gradients. All **47 CPU/CUDA dense
reference cases** pass; 13 use the final selected native source and the rest
are explicitly marked historical. Maximum tensor-VJP difference is 7.11e-15.
The freshly generated overlay matches all 75 build-overlay files. The pinned
QuantLib checkout remains clean at
`06edddbdb85f006c43a449b5fc846a00c67477f3`. No stock double-based QuantLib
library is linked into the TorchReal build.

The remaining full-AD profile still shows substantial operator assembly,
band application, and conservative solver-domain/residual-check overhead.
Next reversible experiments can fuse those elementary expressions or the
implementation-selection predicates while retaining exact financial behavior,
all coefficient dependencies, forward/higher derivatives, and original-solver
fallbacks. Whole-engine batching and GPU pricing remain separate, unproven work.

## Phase 7: spatial reuse and wider native adjoints

Phase 7 keeps the same QuantLib financial engine, mesh, time discretization,
stopping times, exercise/dividend processing, and interpolation. It reduces
autograd overhead inside explicit first-order VJP boundaries and reuses
graph-connected local-volatility samples within each price. Full AD remains
available for higher derivatives. The default `kernels="auto"` is unchanged.
Phase 6 and earlier sections below record historical measurements.

### Graph-connected spatial reuse

The tensor local-volatility term structure caches the spatial interpolation of
each surface time row on the current mesh. Time interpolation still happens at
the original QuantLib sample times. A singleton or edge-held time row can also
reuse its squared variance. Cache entries retain their tensor graph and forward
dual information. They are owned by a single pricing call, and changing the
mesh or gradient-recording state clears them. Separate prices never share
coefficient tensors or cached graphs. `QLT_SPATIAL_CACHE=0` disables this reuse.

### Explicit first-order boundaries and provenance

`operator-adjoint.patch` adds a VJP around band application. Forward re-enters
the original `TripleBandLinearOp::apply`, including its actual neighbour indices
and reflected endpoints. Backward accumulates all band and state sensitivities,
including duplicate endpoint indices. This remains useful for the narrower
`kernels="apply"` path and fallback schemes.

`step-adjoint.patch` adds a wider boundary **after** the original financial
operator's `setTime` and boundary helper's `setTime`. It accepts only the exact
one-dimensional `FdmBlackScholesOp` type, direction zero, empty boundary sets,
and passive time, step size, and theta. Subclasses, custom boundaries, active
settings, other operators, and other dimensions retain the original scheme.
Forward calls the original QuantLib Douglas, implicit Euler, or explicit Euler
scheme with an immutable snapshot of the already assembled matrix. The
snapshot's `setTime` is a no-op; the actual financial operator has already
updated its coefficients outside the boundary. Crank–Nicolson keeps its two
original Euler substeps and their separate coefficient updates.

For a step `A x = B u`, backward solves `A^T lambda = g` with QuantLib's original
tridiagonal algorithm, then returns `B^T lambda` and every band contribution.
The application stencil reflects endpoints while splitting ignores the unused
outer lower/upper entries; their distinct coefficient derivatives are retained.
Step settings enter this boundary only when passive, so no active setting is
silently converted to a constant.

`diffusion-adjoint.patch` adds a local-volatility assembly boundary around
`(r-q-v/2) Dx + (v/2) Dxx - r I`, where `v` is variance. Forward calls the
original `mult` and `axpyb` methods in their original order. Backward returns
sensitivities for variance, both derivative operators, rate, and yield. It packs
all inputs before rebinding output, preserving aliases. Quanto and Black-vol
assembly branches retain their original paths.

These numerical dispatch patches are separate from the compatibility overlay;
the pinned QuantLib checkout remains untouched. Native values are read only
inside the explicit custom-autograd boundaries. The scalar adapter does not
detach coefficients or mutate tensors. Native backward loops write newly
allocated gradient buffers. All three new adjoints reject higher derivatives;
`mode="full"` retains the differentiable original/tensor arithmetic.
`QLT_OPERATOR_ADJOINT`, `QLT_STEP_ADJOINT`, and `QLT_ASSEMBLY_ADJOINT` independently
disable these paths when set to zero. They also respect `QLT_NATIVE_ADJOINT=0`.

The audit found and fixed an existing raw-storage issue in native solve VJPs:
`.contiguous()` does not necessarily materialize a tensor's lazy negative bit.
Raw reads now resolve lazy negative/conjugate views before obtaining a data
pointer. A saved pre-fix regression failed for lazy-negative solve coefficients;
the new tests compare all such coefficient/RHS VJPs with independent dense
solves and also cover lazy-negative incoming gradients.

### Functional PCR fusion

An opt-in PCR experiment combines multiply/add expressions using `addcmul` in
both diagonal and RHS updates. This reduces forward dispatcher work without a custom derivative: full,
higher, and forward AD continue through PyTorch's tensor operations. Fused
arithmetic can change rounding; independent dense equations, VJPs, numerical
differences, and pricing comparisons check the result. It made end-to-end CPU
pricing slower, so `QLT_PCR_FUSED` **defaults to zero**. The other Phase 7 switches
default to one. All switches are immutable after their first native use in a
process.

### Individual ablations

These comparisons use one worker process per configuration and grid, one torch
intra-op/inter-op thread, eight warmups, and nine timed calls. The total includes
construction, forward, VJP, and cleanup. Each row changes only the named feature;
all raw prices and coefficient VJPs are retained alongside the timing samples.
The reports identify their native-source versions, including intermediate
implementations. Historical timings are not multiplied together.

| Change | Mode / grid | Before ms | After ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| [Spatial reuse](../benchmarks/fd_phase7_spatial_ablation.json) | full, 30/61 | 493.03 | 465.06 | 1.06× |
| Spatial reuse | adjoint, 30/61 | 188.14 | 167.18 | 1.13× |
| [Application VJP](../benchmarks/fd_phase7_apply_ablation.json) | adjoint, 30/61 | 168.77 | 145.43 | 1.16× |
| Application VJP | adjoint, 60/121 | 374.72 | 342.52 | 1.09× |
| [Step VJP](../benchmarks/fd_phase7_step_ablation.json) | adjoint, 30/61 | 140.73 | 102.73 | 1.37× |
| Step VJP | adjoint, 60/121 | 334.59 | 245.91 | 1.36× |
| [Assembly VJP](../benchmarks/fd_phase7_assembly_ablation.json) | adjoint, 30/61 | 102.68 | 91.11 | 1.13× |
| Assembly VJP | adjoint, 60/121 | 245.47 | 239.05 | 1.03× |
| [PCR fusion experiment](../benchmarks/fd_phase7_fused_ablation.json) | full, 30/61 | 460.52 | 478.75 | 0.96× |
| PCR fusion experiment | full, 60/121 | 1,043.86 | 1,060.11 | 0.98× |

Assembly fusion reduces backward work but increases forward work in these
samples; its larger-grid total improvement is small. PCR fusion reduces forward
time but increases backward time enough to lose overall. Its
[instrumented profile](../benchmarks/fd_phase7_fused_profile.json) has 35,320 backward
ATen calls, versus 28,736 in the Phase 6 profile. It therefore remains disabled
by default. The intermediate fusion report's `phase7` label refers to the
then-candidate configuration with fusion enabled; its recorded flags disambiguate
it. The final harness uses `phase7` for the selected defaults and `pcr_fused` for
the optional experiment.

### Matched end-to-end evidence

The [30/61 comparison](../benchmarks/fd_phase7_timing.json),
[60/121 comparison](../benchmarks/fd_phase7_scaling.json), and
[coincident-event comparison](../benchmarks/fd_phase7_events.json) use Phase 6
`current` versus the selected `phase7` defaults, both with `kernels="auto"`.
Each configuration/case runs in a fresh process with eight warmups and nine
timed samples. The 60/121 comparison reverses configuration order.

| Workload | Mode | Phase 6 ms | Phase 7 ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 30/61 nonuniform surface | full | 483.04 | 475.58 | 1.02× |
| 30/61 nonuniform surface, extended warmup | adjoint | 185.06 | 90.32 | 2.05× |
| 30/61 neural field and squared loss | full | 491.32 | 463.12 | 1.06× |
| 30/61 neural field and squared loss | adjoint | 185.37 | 91.00 | 2.04× |
| 60/121 nonuniform surface | full | 1,054.17 | 1,016.04 | 1.04× |
| 60/121 nonuniform surface | adjoint | 409.43 | 250.38 | 1.64× |
| 30/61 Bermudan neural loss with dividend | full | 543.30 | 522.77 | 1.04× |
| 30/61 Bermudan neural loss with dividend | adjoint | 240.64 | 125.14 | 1.92× |

The short surface-adjoint run measures 189.67 → 23.85 ms. Its Phase 7 worker
remains in the fast regime for all nine timed samples, while the Phase 6 worker
transitions from 41 ms to roughly 190 ms. That apparent 7.95× ratio compares
different observed regimes and is excluded from the table above. All raw data
is retained. The earlier assembly ablation's selected implementation measures
91.11 ms on the same financial workload. Absolute timings continue to depend
strongly on the host's runtime transitions, investigated below.

A [follow-up with 32 warmups and 15 timed calls](../benchmarks/fd_phase7_sustained.json),
again with configuration order reversed, measures **185.06 → 90.32 ms (2.05×)**
for this surface-adjoint workload. Both workers now reach the sustained slower
regime. This is the comparable surface ratio reported in the table. The shorter
and longer experiments are both preserved rather than overwriting the anomaly.

### Windows process-policy diagnosis

[`scripts/diagnose_fd_runtime.py`](../../scripts/diagnose_fd_runtime.py) records
repeated prices, process CPU time, memory, logical processor observations, and
OS-reported MHz outside timed work. These observations perturb intervals between
calls, so the diagnostic totals are separate from headline benchmarks.

The [default-policy run](../benchmarks/fd_phase7_runtime_default.json) transitions
from approximately 23 ms to 90 ms as the calling thread moves from logical CPUs
8/10 to 18/19. The reported MHz also changes, but this API is not a calibrated
instantaneous frequency measurement. Fixing affinity to
[CPU 8](../benchmarks/fd_phase7_runtime_cpu8.json) improves the last-30-call median
to 49.43 ms; fixing it to [CPU 19](../benchmarks/fd_phase7_runtime_cpu19.json) does
not remove the slower regime. All runs preserve exactly equal prices/VJPs.

A [process-local opt-out of execution-speed throttling](../benchmarks/fd_phase7_runtime_highqos.json)
keeps the last 30 calls near **23.16 ms**, with no affinity restriction. The
process reports control mask 1 and state mask 0 after the call. This follows
Microsoft's documented HighQoS example; Windows can otherwise infer a process's
QoS from system heuristics and use frequency/core selection to trade performance
for efficiency. See [SetProcessInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation).

The controlled result is strong evidence that Windows execution-speed policy
drives much of the host's timing transition. It does not isolate every scheduler
or frequency mechanism. The diagnostic changes only its own process and ends
when the process exits. No machine power plan or library-wide runtime policy is
changed. The benchmark drivers now accept explicit `--windows-qos high`; the
default remains `system`. Every controlled worker records the requested policy
and observed control/state masks. `high` is a Windows-only benchmark option.

A [repeat under system-managed policy](../benchmarks/fd_phase7_runtime_system_repeat.json)
again transitions from roughly 23 to 91 ms, while observed process control/state
masks remain zero. Thus the diagnostic does not establish explicit EcoQoS
opt-in; the policy dependence is supported by the intervention and repeated
timing behaviour.

With this explicit control, a [matched 32-warmup/15-sample comparison](../benchmarks/fd_phase7_highqos_timing.json)
gives the following stable-regime measurements. Both Phase 6 and Phase 7 receive
the same QoS setting; the code comparison therefore does not conflate the
algorithmic change with the process-policy effect.

| Workload | Mode | Phase 6 ms | Phase 7 ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 30/61 nonuniform surface | full | 93.78 | 88.82 | 1.06× |
| 30/61 nonuniform surface | adjoint | 42.32 | 23.54 | 1.80× |
| 30/61 neural field and squared loss | full | 95.82 | 90.05 | 1.06× |
| 30/61 neural field and squared loss | adjoint | 41.32 | 24.38 | 1.69× |
| 60/121 nonuniform surface | full | 206.93 | 196.96 | 1.05× |
| 60/121 nonuniform surface | adjoint | 92.39 | 62.50 | 1.48× |
| 30/61 Bermudan neural loss with dividend | full | 104.84 | 109.88 | 0.95× |
| 30/61 Bermudan neural loss with dividend | adjoint | 48.80 | 29.94 | 1.63× |

The [larger-grid](../benchmarks/fd_phase7_highqos_scaling.json) and
[event](../benchmarks/fd_phase7_highqos_events.json) runs use the same explicit QoS
and sampling protocol. Full AD is about 5% slower in this controlled Bermudan
run; the small full-mode gains are not universal. Adjoint improves in every
matched case here.

A [full-mode event repeat with configuration order reversed](../benchmarks/fd_phase7_highqos_events_repeat.json)
measures 105.69 → 98.70 ms (1.07×), compared with 104.84 → 109.88 ms (0.95×)
in the first controlled event run. Both are retained. This variation is larger
than the modest full-mode effect for that workload, so no reliable event
full-AD speedup is claimed.

These ratios differ from the system-managed comparisons because the mix of
native and tensor work responds differently to the host execution regime. The
raw system-managed and controlled measurements remain distinct. A process QoS
request is an explicit performance/power choice for the caller, not a numerical
optimization silently imposed by the pricing library.

The [separate memory and graph workers](../benchmarks/fd_phase7_memory.json) show
that the wider boundaries substantially shrink the adjoint graph. Memory is a
single cold-process OS high-water measurement per case, including native
allocations and allocator retention; it is not repeated unique tensor storage.

| Grid | Mode | Peak working-set growth MiB, Phase 6 → Phase 7 | NPV graph nodes, Phase 6 → Phase 7 |
| --- | --- | ---: | ---: |
| 30/61 | full | 29.03 → 28.43 | 7,146 → 6,768 |
| 30/61 | adjoint | 20.73 → 18.47 | 2,326 → 486 |
| 60/121 | full | 45.34 → 44.26 | 15,084 → 14,346 |
| 60/121 | adjoint | 24.98 → 19.37 | 4,396 → 816 |

The [selected-default profile](../benchmarks/fd_phase7_profile.json) has 2,466
forward / 2,459 backward ATen calls for adjoint at 30/61, versus Phase 6's
6,230 / 8,992. It samples the grid 33 times but computes only three spatial
rows. Full AD still has 27,409 / 27,350 calls and 34 PCR solves; those solves
account for about 196 ms of instrumented forward time. Profiler timings include
instrumentation overhead and nested scopes and are not end-to-end timings.

The remaining full-AD cost is predominantly tensor elimination and its backward
graph. Adjoint retains assembly/step wrappers, graph-connected time sampling,
interpolation, and financial setup. Spatial reuse and the wider boundaries
address the previous measured bottlenecks; batched financial execution,
trajectory checkpointing, and further reductions in full-AD dispatcher work
remain separate investigations. Single-contract pricing is still CPU-only.

The standalone batched solve comparisons
([original arithmetic](../benchmarks/fd_phase7_batched_pcr_fused0.json),
[optional fusion](../benchmarks/fd_phase7_batched_pcr_fused1.json)) hold PCR's pivot
checks disabled in both configurations and use independently validated strictly
dominant systems. They include forward, every band/RHS VJP, and cleanup, with
one CPU thread and synchronized CUDA timings on the RTX 4060 Laptop GPU.

| Systems × nodes × RHS | Device / transfer | Unfused ms | Fused ms |
| --- | --- | ---: | ---: |
| 1 × 61 × 1 | CPU | 1.45 | 1.70 |
| 1 × 61 × 1 | CUDA resident | 38.04 | 37.21 |
| 1 × 61 × 1 | CUDA roundtrip | 37.16 | 39.48 |
| 256 × 257 × 4 | CPU | 641.36 | 644.24 |
| 256 × 257 × 4 | CUDA resident | 54.43 | 58.46 |
| 256 × 257 × 4 | CUDA roundtrip | 68.13 | 65.39 |

Fusion has no consistent throughput advantage here, reinforcing the decision
to leave it opt-in. The larger unfused workload is 11.8× faster on resident CUDA
and 9.4× including input/result/all-gradient transfers than the same single-thread
CPU tensor PCR. The small workload remains much slower on CUDA. All eight
independent CPU dense-reference comparisons pass (maximum value difference
`1.111e-15`, VJP difference `1.777e-15`). These are numerical-kernel measurements,
not batched financial prices or a comparison against native batched Thomas.

The [controlled-QoS CPU/CUDA repeat](../benchmarks/fd_phase7_highqos_batched_pcr.json)
uses 32 warmups and 15 timed samples in each worker, with the same HighQoS request
on both CPU and CUDA. For 256 × 257 × 4, unchecked CPU is **116.60 ms**, CUDA
resident **12.81 ms**, and CUDA including transfers **13.54 ms**: 9.1× and 8.6×
against this CPU comparator. Checked totals are 124.16 / 11.72 / 13.63 ms. The
small unchecked case is still slower on CUDA (1.53 ms CPU versus 7.81 ms resident).
The eight independent dense-reference checks pass. The overlap and reversals
between checked/unchecked CUDA timings do not establish a benefit from checks;
host/launch variability remains at this scale.

### Verification

The final Phase 7 build passes **417 tests** (`python -m pytest -q`, 103.04 s;
19 existing JIT/profiler warnings). An earlier build with PCR fusion enabled
also passes all 417 tests. New coverage exercises nonuniform/singleton spatial
sampling, strided tensors and shared backing storage, actual row reuse,
graph-enabled and graph-disabled execution, full-mode second derivatives and forward AD,
and every input VJP of application, step, and diffusion-assembly boundaries.
Independent dense reflected and compact matrices distinguish the step's
application and solve boundary rules. All new native VJPs have numerical
difference checks and enforce their first-order contract.

Engine tests cover Douglas, Crank–Nicolson, and implicit Euler across European,
American, and Bermudan exercise, dated nonflat discount/yield curves, and a cash
dividend coincident with exercise. Values and gradients agree with original
full-AD Thomas execution and directional differences. A separate actual
Black–Scholes operator probe verifies that custom boundary callbacks, active
time/step/theta settings, and derived operator classes take the original path
and retain their values/VJPs. Profiler assertions verify both eligible dispatch
and these fallbacks. The numerical engine's broader financial scope remains
limited to the existing API; these hooks do not add a new financial model.

The [final evidence validation](../benchmarks/fd_phase7_validation.json) checks
**3,932 comparisons**, including every stored warmup and timed price/VJP and
300 runtime-diagnostic prices/VJPs, against a fixed reference per financial
case. Maximum price difference is `1.422e-14`; maximum VJP difference is
`8.172e-14`. All 75 files in a freshly generated overlay match the build
overlay byte for byte. The pinned upstream worktree is clean. Sixteen separate
CPU/CUDA dense-reference solve validations pass. Source, binary, build
configuration, test, script, patch, and log hashes are retained.

[`fd_phase7_source.zip`](../benchmarks/fd_phase7_source.zip) preserves the build,
native, Python, test, benchmark, example, and license sources independently of
future edits. The validator verifies 55 build/native/Python/test source files
against the current workspace and records the archive hash. Dependencies,
compiled binaries, reports, and documentation are outside this source snapshot;
the QuantLib commit and PyTorch dependency remain pinned by the included setup.

### Reproduction and next work

[`fd_phase7_commands.json`](../benchmarks/fd_phase7_commands.json) retains the
commands for matched timings, ablations, profiles, memory, longer warmups,
runtime diagnostics, CPU/CUDA solves, build, tests, and final validation. It
notes the two historical fusion reports whose former `phase7` candidate label
now corresponds to `pcr_fused` or an explicit `--pcr-fused 1`. Current `current`
continues to mean Phase 6; `phase7` selects the verified defaults.

```powershell
.venv/Scripts/python.exe scripts/build.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels current,phase7 --kernels auto --grids 30x61 --variants optimized --surfaces surface,neural --modes full,adjoint --tasks timing --warmup 32 --repeats 15 --windows-qos high --no-stock --output build/phase7_reproduction.json
```

The QoS option is specific to Windows; omit it for system-managed workers or
other platforms. Next investigations target full-AD PCR operation/layout costs
and the remaining native-adjoint setup/step overhead, using explicit runtime
controls and independent derivatives. A wider financial batching interface or
whole-engine replacement remains a separate architectural decision.

A [reversible PCR layout probe](../benchmarks/pcr_layout_experiment.json) stored
bands and RHS in one tensor to share neighbour rolls. Dense values and all input
VJPs passed, but it was slower in every measured case: 1 × 61 × 1 CPU
1.51 → 2.48 ms; 256 × 257 × 4 CPU 124.39 → 257.26 ms; resident CUDA
10.03 → 14.21 ms. The Python forward has extra frontend overhead, but backward
also became slower. The self-contained report retains its source and raw
samples. It was not integrated; further work should target arithmetic/AD
dispatch cost rather than this layout alone.

A promising next reversible experiment is a fused elementary PCR-stage
operator, with an independent full/reference path. It must differentiate the
PCR recurrence itself, preserve gradients for every band and RHS, and retain
second/higher derivatives and forward AD through a tested differentiable path.
Any native first-order backward must be explicit and verified independently;
it must not silently turn full mode into the solve-level first-order adjoint.
This remains an investigation, not an implemented default or a measured gain.

## Phase 6: vector splines and full-AD parallel elimination

The current default is `fd_vanilla(..., kernels="auto")`. It retains the actual
QuantLib financial engine and uses a guarded tensor solve in full AD. Adjoint
mode continues to run the original QuantLib solver inside its explicit VJP
boundary. `kernels="vectorized"` retains the upstream Thomas recurrence for
every full-AD solve. The Phase 5 and earlier measurements below are historical.
This is a numerical-kernel extension within Approach A, not a replacement
financial engine. The pricing API remains one contract on CPU.

### Spline expressions and provenance

`spline-kernels.patch` vectorizes expressions in the pinned QuantLib cubic spline:
interval slopes, interior RHS assembly, Hyman monotonicity filtering, cubic
coefficients, and primitive interval integrals. The original scalar paths and
evaluation formulas remain. The primitive's sequential prefix accumulation is
retained, and `primitiveConstants()` remains eagerly available. Fixed mesh
geometry stays passive. Active knots, values, and queries keep their gradients.

The vector limiter preserves ordered min/max ties and the original
`correction != derivative` branch. It avoids unused zero divisions and falls
back to the scalar expressions for nonfinite intermediate values. The new
`TridiagonalAdjoint` uses the original upstream tridiagonal solve in forward
and transpose solve in backward, with all three bands and RHS connected. It
rejects higher derivatives, like the existing splitting adjoint. This dispatch
has its own `tridiagonal-adjoint.patch`.

The spline expressions are derived from QuantLib 1.40's
`cubicinterpolation.hpp`; their copyright notices are preserved in
`cpp/spline_kernels.cpp`, and the upstream license is distributed in
[`LICENSES/QuantLib.txt`](../../LICENSES/QuantLib.txt). The upstream checkout stays
at `06edddbdb85f006c43a449b5fc846a00c67477f3`. Numerical dispatch patches remain
separate from math compatibility patches. No XAD/QuantLibAAD code or stock
double-based QuantLib library is used.

### Parallel solve and conservative automatic selection

`cpp/parallel_solve.cpp` implements functional parallel cyclic reduction (PCR).
Each stage eliminates both neighbours simultaneously; the coupling distance
doubles, giving logarithmic sequential depth. Full, higher, and forward AD
flow through the tensor arithmetic. This is a different elimination algorithm
and floating-point evaluation order from QuantLib's Thomas recurrence, with
the same assembled linear system and time discretization.

The standalone `tridiagonal_solve_pcr` accepts compact bands with shapes
`(..., N-1)`, `(..., N)`, `(..., N-1)` and RHS `(..., N, R)` on CPU or CUDA,
with broadcast batch axes, shared coefficients, strided views, and multiple
RHS. Its default checks reject nonfinite inputs/output and every zero or
nonfinite stage pivot. Those checks synchronize CUDA. `check_pivots=False`
avoids host value reads; the caller then owns admissibility and residual checks.
PCR can reject an invertible matrix that Thomas solves, such as
`[[1, 1], [1, 0]]`. Nonsingularity alone does not establish admissibility.

Automatic CPU selection is narrower. It considers splitting and spline solves
with at least eight rows, full AD, and gradient recording enabled. Every input
must be finite; each diagonal magnitude must exceed its row's off-diagonal
magnitude sum by a relative margin of at least `1e-7`. Diagonal magnitudes must
lie in `[1e-12, 1e12]`, their max/min ratio must not exceed `1e6`, and RHS
magnitudes must not exceed `1e50`. A computed candidate must also pass a
componentwise scaled residual bound of `64*N*epsilon`, with a `1e-300`
absolute floor. These deliberately conservative limits restrict an optimization,
not the accepted pricing inputs. Outside them, the original QuantLib solve
runs, preserving its own errors and pivot policy. No exception is swallowed.

Selection and residual checks read values without building a reverse graph;
the solve itself receives the original graph-connected tensors. No coefficient
is detached and no tensor is updated in place. Custom-adjoint re-entry never
selects PCR. Graph-disabled inference still uses the native QuantLib solve;
forward dual tensors under `no_grad` keep a differentiable original path.
`kernels="pcr"` explicitly selects checked PCR splitting in full mode and can
raise for its stricter pivot pattern. `parallel-solve.patch` retains all
original implementations as selectable fallbacks.

PCR's shifted neighbours now use whole-tensor rolls. Structural zero boundary
bands prevent wrapped values from coupling into the system, and this invariant
is retained by each elimination stage. `QLT_PCR_ROLL=0` selects the original
explicitly padded shifts for an ablation. This does not introduce periodic
boundary conditions. `QLT_SPLINE_KERNELS=0` and `QLT_TRIDIAGONAL_ADJOINT=0`
independently disable the spline vector expressions and spline solve adjoint.
All three switches default to one and are immutable after first use.

### Current end-to-end evidence

The [matched 30/61 comparison](../benchmarks/fd_phase6_timing.json) runs each
case in a fresh process, with one intra-op/inter-op thread, eight warmups and
nine timed samples. Baseline is the Phase 5 `final` configuration with
`kernels="vectorized"`; current uses all Phase 6 features and `kernels="auto"`.
Totals include construction, forward, VJP, and graph cleanup. Neural work
includes the 17-parameter network and squared pricing loss.

| Workload | Mode | Phase 5 total ms | Current total ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| Nonuniform 3×5 surface | full | 1,987.39 | 481.63 | 4.13× |
| Neural field and loss | full | 2,013.16 | 495.59 | 4.06× |
| Nonuniform 3×5 surface | adjoint | 267.47 | 183.69 | 1.46× |
| Neural field and loss | adjoint | 268.58 | 185.14 | 1.45× |

All 12 cross-case checks pass: maximum price difference `1.066e-14`, maximum
coefficient/neural VJP difference `7.106e-14`. These are measurements on this
host, not universal performance guarantees. The absolute timings differ from
earlier runs, and raw samples show transitions from fast initial calls into a
slower sustained regime. Its cause remains unestablished. Do not multiply
speedups across historical reports or discard the slower samples.

A [process-CPU-time diagnostic](../benchmarks/fd_phase6_cpu_time_diagnostic.json)
adds fifteen timed samples after eight warmups. Current adjoint medians are
190.74 ms elapsed / 187.50 ms process CPU; full AD medians are 493.96 / 484.38 ms.
The process clock includes all worker threads and has coarse resolution on
Windows. The slower observed regime therefore consumes CPU time and is not
explained solely by descheduling or waiting; its underlying cause remains open.

The [initial spline ablation](../benchmarks/fd_phase6_spline_ablation.json) measured
roughly 36.16 → 27.14 ms for neural adjoint at 30/61 and 78.78 → 57.82 ms for
the surface at 60/121. Its first 30/61 baseline surface worker measured 181 ms,
unlike the [pre-change baseline](../benchmarks/fd_phase6_baseline.json)'s 38.80 ms;
that apparent larger speedup is not used as a headline. The
[initial PCR pilot](../benchmarks/fd_phase6_pcr_pilot.json) measured approximately
2.06–2.07 s → 0.664 s for full AD with padded PCR splitting, before automatic
selection and roll shifts. Source fingerprints distinguish these stages.

The [roll-versus-padding ablation](../benchmarks/fd_phase6_roll_ablation.json)
holds automatic selection and every other feature fixed. Full surface pricing
at 30/61 improves **609.10 → 500.77 ms (1.22×)**, with identical price and VJP.
It uses the same eight warmups/nine samples in separate processes.

The [60/121 comparison](../benchmarks/fd_phase6_scaling.json) reverses configuration
order (current before baseline). The [event comparison](../benchmarks/fd_phase6_events.json)
prices a neural-volatility Bermudan put with a cash dividend on an exercise
date. Both retain the same extended sampling protocol:

| Workload | Mode | Phase 5 total ms | Current total ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 60/121 nonuniform surface | full | 7,666.33 | 1,040.83 | 7.37× |
| 60/121 nonuniform surface | adjoint | 589.77 | 403.27 | 1.46× |
| 30/61 Bermudan neural loss with dividend | full | 2,169.21 | 549.19 | 3.95× |
| 30/61 Bermudan neural loss with dividend | adjoint | 315.40 | 232.69 | 1.36× |

The event comparison's maximum price difference is `1.422e-14` and maximum
neural-gradient difference is `8.172e-14`. The gain grows with spatial size in
these full-AD cases, consistent with replacing a sequential scalar recurrence
by vector elimination stages; this is not a convergence or universal scaling
claim.

The [isolated memory/graph measurements](../benchmarks/fd_phase6_memory.json)
compare the same configurations. OS high-water growth is one cold measurement
per case, including native allocations and allocator retention; it is not
unique tensor storage or a repeated median.

| Grid | Mode | Peak working-set growth MiB, Phase 5 → current | Reachable NPV graph nodes, Phase 5 → current |
| --- | --- | ---: | ---: |
| 30/61 | full | 64.21 → 29.35 | 28,638 → 7,146 |
| 30/61 | adjoint | 21.87 → 20.98 | 2,895 → 2,326 |
| 60/121 | full | 189.93 → 45.22 | 103,908 → 15,084 |
| 60/121 | adjoint | 27.61 → 25.04 | 5,625 → 4,396 |

At 60/121, full AD uses about 76% less measured peak growth and 85% fewer graph
nodes. The adjoint graph improves more modestly because it already omits the
elimination recurrence.

The [current 30/61 profiles](../benchmarks/fd_phase6_profile.json) record
28,228/28,736 forward/backward ATen calls in full AD and 6,230/8,992 in adjoint.
Full AD executes 34 PCR solves (33 rollback systems and one spline system).
Its instrumented PCR scopes occupy 196.2 ms inside 372.9 ms for rollback and
interpolation. Adjoint's corresponding scope is 125.8 ms, with 114.3 ms in
rollback and 18.8 ms in grid sampling. These nested instrumented times are not
wall-clock benchmarks and must not be added. The spline remainder is now much
smaller; full AD's staged tensor elimination and adjoint's per-step tensor
dispatch are the next substantial boundaries.

### Batched CPU/CUDA solve measurements

The [standalone PCR benchmark](../benchmarks/fd_phase6_batched_pcr.json) measures
the same tensor solver and all-band/RHS VJP on one CPU thread and an NVIDIA
GeForce RTX 4060 Laptop GPU. Each task uses a fresh process with eight warmups
and nine samples. CUDA timings synchronize after forward, backward, and
cleanup. Validation runs separately against CPU dense solves and VJPs;
maximum value/VJP differences are `1.111e-15` / `1.777e-15`.

| Systems × nodes × RHS | Pivot checks | CPU total ms | CUDA resident total ms | CUDA with transfers total ms |
| --- | --- | ---: | ---: | ---: |
| 1 × 61 × 1 | enabled | 1.82 | 45.49 | 45.36 |
| 1 × 61 × 1 | disabled | 1.51 | 38.08 | 38.04 |
| 256 × 257 × 4 | enabled | 666.59 | 64.50 | 78.65 |
| 256 × 257 × 4 | disabled | 634.87 | 58.21 | 67.36 |

For the larger unchecked case, the measured gain over the same CPU PCR is
**10.9× resident / 9.4× including transfers**. Resident throughput is about
4,398 systems/s; the transfer-inclusive result is about 3,801 systems/s.
Transfers include ordinary unpinned input H2D and result/all-gradient D2H
copies. This is not an overlapped transfer pipeline. Checks add per-stage
device synchronizations; disabling them requires a justified matrix domain
and independent residual validation.

The small system is much slower on CUDA. These two workload sizes do not
locate a general crossover, and the CPU comparator is single-thread tensor
PCR, not an optimized batched native Thomas implementation. These measurements
support further work on batched numerical rollback; they do not establish
batched financial prices or GPU option-pricing throughput.

### Verification and remaining work

The current native build passes **328 tests**. New coverage includes every
public spline coefficient/primitive and limiter flag, active knot/value/query
VJPs against the original spline and independent dense natural splines, first
and second differences, arbitrary strided tridiagonal bands, RHS/output aliasing,
batched CPU/CUDA PCR values and all-band/RHS VJPs, second derivatives, forward
AD with and without `no_grad`, stock prices/vega for three schemes, and
European/American/Bermudan dividend cases. Automatic dispatch tests verify
which algorithm actually runs, its gradients, and fallback for Thomas-admissible
zero diagonals, weak dominance, extreme scales, and invalid upstream pivots.

The [final evidence validation](../benchmarks/fd_phase6_validation.json) retains
source, binary, test, script, generated-patch, and log hashes. It rechecks all
raw timing samples against a fixed reference per financial case: **1,326
comparisons pass**, with maximum price difference `1.422e-14` and VJP difference
`9.238e-14`. All 69 files in a freshly generated overlay match the build
overlay byte for byte, and the pinned upstream worktree is clean. The eight
separate CPU/CUDA dense-reference validations also pass. Historical reports
are explicitly labelled when their native-source fingerprint differs.

Full AD still retains each elimination stage across the trajectory. Adjoint
mode still records coefficient sampling, assembly, and step operations. Both
retain financial setup and scalar boundary/interpolation work. Follow-up
measurements target graph/memory scaling, wider per-step VJPs/checkpointing,
native execution at additional explicit first-order adjoint boundaries,
reuse of graph-connected spatial samples, and batched solve throughput.
Standalone CUDA solves do not establish GPU option-pricing throughput.

### Reproduction

`scripts/benchmark_fd_optimizations.py` retains historical `final` as the
Phase 5 configuration; `current` enables Phase 6. `--kernel-overrides` permits
the same driver to compare original Thomas with automatic selection. Every
case still gets its own process. The commands for the additional ablations,
scaling, events, memory, profiles, and batched solves are retained in
[`fd_phase6_commands.json`](../benchmarks/fd_phase6_commands.json).

```powershell
.venv/Scripts/python.exe scripts/build.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels final,current --kernel-overrides final=vectorized,current=auto --grids 30x61 --variants optimized --surfaces surface,neural --modes full,adjoint --tasks timing --warmup 8 --repeats 9 --no-stock --output docs/benchmarks/fd_phase6_timing.json
.venv/Scripts/python.exe scripts/benchmark_parallel_solve.py --shapes 1x61x1,256x257x4 --warmup 8 --repeats 9 --output docs/benchmarks/fd_phase6_batched_pcr.json
```

## Phase 5: persistent tensor arrays and native solve execution

The current CPU implementation preserves Approach A and the actual QuantLib
engine. It removes scalar dispatch at the existing custom-adjoint boundary,
keeps tensor backing across array operations, and vectorizes local-volatility
sampling and elementwise array arithmetic. The Phase 4 and earlier sections
below are historical measurements. Batched financial pricing, GPU rollback,
and two-factor engines remain open milestones; these results concern the
single-contract CPU engine.

### Boundaries, provenance, and derivative contracts

The per-solve adjoint now copies the operator's **values inside its explicit
custom-autograd forward**, then runs the same compiled upstream
`TripleBandLinearOp::solve_splitting` on passive `TorchReal` values. Its backward
uses the upstream `TridiagonalOperator` transpose solve on native values and
returns the analytical band/RHS VJP. Assembly of `a*L+b*I` stays outside the
boundary, retaining sensitivities to all three bands, RHS, and scales a and b.
There is no automatic tensor-to-double conversion in the scalar adapter,
replacement Thomas algorithm, stock QuantLib linkage, or XAD implementation.
The custom adjoint still rejects higher derivatives explicitly.

Array entries can retain an immutable tensor owner and an element index.
Packing recognizes an unchanged consecutive run by tensor identity **and every
entry's index**, returning its owner or a slice instead of stacking scalar
views. Copying preserves provenance; arithmetic/reassignment creates a new
value and drops it. Partial changes and reordered entries fall back to packing.
In adjoint pricing, scalar views are materialized only when scalar arithmetic
actually consumes them; comparison reads retain the existing piecewise branch
policy. Full AD uses eager `unbind` views because repeated lazy selects made
its Thomas-elimination graph slower. No tensor is mutated in place.

Reconstruction respects `no_grad` boundaries, including views originally
created without recording. Full-mode graph-disabled inference can use the
native solve boundary, but a forward-mode dual tensor selects the original
full path: `no_grad` does not disable forward AD. Tests cover that distinction.

The tensor-backed volatility adapter samples an entire spatial grid at a time.
Only fixed geometry (indices and interpolation weights) is cached per pricing
call; coefficient tensors remain graph-connected. Time interpolation, spatial
edge holding, singleton axes, squaring, and time-range checks retain the scalar
adapter's definitions. Other term structures, active coordinates, and the
upstream illegal-volatility-overwrite path retain scalar sampling.

`array-kernels.patch` adds dispatch before the 40 existing binary/scalar
`std::transform` operations in QuantLib's `Array`. Size checks, moves, operand
order, and the scalar fallback remain. The hooks pack all inputs before
rebinding output handles, including self aliases. All-passive arrays continue
using native arithmetic. Sampling and arrays are selected with
`kernels="vectorized"`; `scalar` and `apply` retain their earlier boundaries.
The generated `sampling-kernel.patch` and `array-kernels.patch` are separate
from math compatibility, the adjoint hook, band kernels, and profiling patches.

The Python pricing bridge exposes NPV only. A thread-local, exception-safe
scope now avoids computing the engine's unused delta, gamma, and theta results.
The original all-results branch is retained in `pricing-output.patch` and is
the default **outside this bridge's scope**. Crucially, the upstream theta
snapshot and its stopping time remain in the rollback: removing that time
would change the discretization. NPV interpolation, exercise/dividend order,
meshing, damping, and all exposed financial functionality are unchanged.

### Controlled ablation and verification

The [incremental 30/61 ablation](../benchmarks/fd_phase5_ablation.json) uses a
nonuniform 3×5 surface, European call, adjoint mode, one intra-op and inter-op
thread, three warmups and five timed samples per fresh worker. Total includes
financial/engine construction, forward, coefficient VJP, and graph cleanup.
Profiler and memory workers are separate from timing.

| Enabled optimizations, cumulatively | Total median ms |
| --- | ---: |
| Phase 4 baseline | 2,213.67 |
| Native execution inside solve adjoint | 1,752.07 |
| Retain tensor backing across arrays | 1,511.23 |
| Vector coefficient sampling | 150.44 |
| Vector array arithmetic | 61.58 |
| Lazy scalar views in adjoint mode | 42.74 |
| Avoid unused engine Greeks | 34.12 |

This is a **64.9×** measured total reduction against the Phase 4 baseline in
the same experiment. Prices are identical across all seven configurations;
maximum coefficient-gradient difference is `1.4211e-14`. The large transition
after sampling also reduces the graph substantially; it should not be read as
a universal additive cost attribution. The host has documented allocator and
cold/warm regime effects. Every raw warmup and timed sample is retained.

### End-to-end pricing and neural-loss results

The [30/61 comparison](../benchmarks/fd_phase5_timing.json) measures baseline and
final code in fresh workers with the same three warmups/five samples. Neural
rows include network evaluation, price, squared loss, parameter VJP, and cleanup.
Surface rows differentiate NPV with respect to the volatility samples. Median
phase times need not add to median total.

| Workload | Mode | Forward ms, baseline → final | Backward ms, baseline → final | Total ms, baseline → final | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Constant volatility | adjoint | 614.68 → 17.89 | 943.78 → 13.25 | 1,576.59 → 31.85 | 49.5× |
| Nonuniform 3×5 | full | 939.48 → 504.04 | 1,796.73 → 864.56 | 2,801.13 → 1,401.22 | 2.00× |
| Nonuniform 3×5 | adjoint | 837.25 → 19.71 | 1,354.42 → 14.93 | 2,214.88 → 35.13 | 63.0× |
| Neural field and loss | full | 936.27 → 508.96 | 1,816.39 → 864.87 | 2,805.11 → 1,404.06 | 2.00× |
| Neural field and loss | adjoint | 842.88 → 21.49 | 1,361.64 → 17.88 | 2,264.52 → 39.97 | 56.7× |

All 26 cross-process/reference checks pass, with identical prices and maximum
gradient difference `6.395e-14`. Inference falls from 79.73 to 13.82 ms for
constant volatility, 99.84 to 14.64 ms for the sampled surface, and 101.07 to
14.47 ms for the neural workload. The final constant-volatility inference is
still about **74× slower** than its stock forward-only reference (0.1870 ms);
there is no stock reverse-mode baseline.

The constant/full row in that report is deliberately excluded from this table:
its final samples are 244.5, 247.6, 484.3, 1,400.9 and 1,381.9 ms. The median
does not describe a settled regime. Raw samples and the subsequent longer
confirmation run are retained instead of presenting its apparent 4.5× ratio
as a reliable improvement.

The [extended confirmation](../benchmarks/fd_phase5_constant_steady.json), with
eight warmups and nine samples, gives **2,239.51 → 1,390.87 ms (1.61×)** for
constant-volatility full AD. Final samples span 1,364.72–1,481.90 ms, consistent
with the late samples in the shorter run. Baseline samples still show occasional
load spikes; the report retains them all.

The [neural Bermudan put](../benchmarks/fd_phase5_events.json), with a cash dividend
on an exercise date, improves from **2,373.40 to 39.24 ms (60.5×)** in adjoint
mode and **3,024.13 to 1,477.13 ms (2.05×)** in full mode. This includes all
neural-loss parameter gradients and the same sampling protocol. It supplements
the independent exercise/dividend difference tests; it does not characterize
all event schedules or network sizes.

At [60 steps / 121 points](../benchmarks/fd_phase5_scaling.json), nonuniform-surface
adjoint total falls from **8,430.97 to 84.57 ms (99.7×)** under the same warmup
and repetition protocol. Prices are identical and the maximum surface-gradient
difference is `9.770e-15`. The larger grid is still a single CPU contract, not
a batched throughput measurement.

### Memory and remaining operator costs

The [isolated cold memory and graph workers](../benchmarks/fd_phase5_memory.json)
show the following reductions for the nonuniform surface:

| Steps / points | Mode | Peak working-set growth MiB, baseline → final | Reachable NPV graph nodes, baseline → final |
| --- | --- | ---: | ---: |
| 30 / 61 | full | 102.77 → 64.43 | 60,912 → 28,638 |
| 30 / 61 | adjoint | 65.17 → 21.88 | 37,349 → 2,895 |
| 60 / 121 | full | 333.56 → 189.93 | 227,832 → 103,908 |
| 60 / 121 | adjoint | 189.78 → 27.60 | 137,489 → 5,625 |

At 60/121, adjoint OS peak growth is about 85% lower and the reachable graph has
about 96% fewer nodes. OS high-water growth includes allocator retention and
native allocations; these are single cold measurements, not unique tensor
storage or repeated memory medians. Graph counts refer to returned NPV only.

The [matched 6/21 profiles](../benchmarks/fd_phase5_profile.json) record full-mode
ATen calls dropping from 20,342/18,049 forward/backward to 14,048/9,197, and
adjoint calls from 22,616/21,896 to 3,846/2,590. These are instrumented counts,
not wall-clock speedups. Full-mode graph size still exceeds 100,000 nodes at
60/121: the upstream scalar solve recurrence remains a substantial bottleneck.

The [final 30/61 adjoint profile](../benchmarks/fd_phase5_final_profile.json) has
12,865 forward ATen calls and only 692 `select` calls (versus 34,001 selects in
the intermediate profile). All 549 unpack boundaries remain, but most no
longer materialize scalar views. The instrumented rollback scope is 75.5 ms
inside a 171.8 ms rollback-and-interpolation scope; the latter's remaining work
is primarily the original monotonic cubic spline construction. This is evidence
for the next optimization boundary, not an uninstrumented timing decomposition.
No theta-interpolation scope is executed in the final NPV-only bridge.

### Verification and reproduction

The [native build and test record](../benchmarks/fd_phase5_validation.json) records
**194 passed in 29.65 seconds**, plus 18 PyTorch internal JIT deprecation
warnings from the forward-AD checks. No tests were skipped. This includes
independent stock QuantLib prices/volatility bumps, dense nonsymmetric solve
VJPs for every coefficient and scale, first/second differences, nonuniform
surface and neural-parameter directional differences, all exposed schemes,
European/American/Bermudan exercise, and coincident dividends/exercise. New
array tests cover all 40 overloads, compound self aliases, strided/expanded
views, copies, partial rebinding, reordered/subrange packing, first/higher
derivatives, and grad-mode boundaries. The pinned QuantLib checkout is clean.
The validation artifact includes source and native binary hashes. These are
local Windows results; no remote Linux execution is claimed.

The [initial remeasurement](../benchmarks/fd_phase5_baseline.json),
[first combined optimization](../benchmarks/fd_phase5_optimized.json),
[intermediate profile](../benchmarks/fd_phase5_intermediate_profile.json), and
[lazy-view pilot](../benchmarks/fd_phase5_lazy_pilot.json) preserve the investigation.
The intermediate profile counted 549 unpacks and 34,001 `aten::select` calls in
one 30/61 forward. Lazy views reduced adjoint total from about 61 to 42 ms, but
applying them to full AD raised total from about 1.46 to 1.82 seconds. The final
implementation therefore selects eager views for full AD. Instrumented profile
times are not used as wall-clock speedups.

Diagnostic flags must be set before loading the library. Defaults are all `1`:
`QLT_NATIVE_ADJOINT`, `QLT_PACKED_ARRAYS`, `QLT_VECTOR_SAMPLING`,
`QLT_ARRAY_KERNELS`, `QLT_LAZY_ARRAYS`, and `QLT_PRICE_ONLY`.
`scripts/benchmark_fd_optimizations.py` sets them in isolated workers, combines
raw evidence, checks source fingerprints across levels, and rejects price/VJP
regressions. `baseline` disables all six; `final` enables all six. Older
`QLT_PASSIVE_CONSTANTS` and `QLT_SAMPLE_CACHE` switches remain available.

```powershell
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,solve,packed,sampled,arrays,lazy,final --grids 30x61 --variants optimized --kernels vectorized --surfaces surface --modes adjoint --tasks timing --warmup 3 --repeats 5 --output docs/benchmarks/fd_phase5_ablation.json
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,final --grids 30x61 --variants optimized --kernels vectorized --surfaces constant,surface,neural --modes inference,full,adjoint --tasks timing --warmup 3 --repeats 5 --output docs/benchmarks/fd_phase5_timing.json
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,final --grids 30x61 --variants optimized --kernels vectorized --surfaces neural --contracts bermudan --modes full,adjoint --tasks timing --warmup 3 --repeats 5 --output docs/benchmarks/fd_phase5_events.json
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,final --grids 60x121 --variants optimized --kernels vectorized --surfaces surface --modes adjoint --tasks timing --warmup 3 --repeats 5 --output docs/benchmarks/fd_phase5_scaling.json
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,final --grids 30x61,60x121 --variants optimized --kernels vectorized --surfaces surface --modes full,adjoint --tasks memory,graph --warmup 1 --repeats 1 --output docs/benchmarks/fd_phase5_memory.json
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,final --grids 6x21 --variants optimized --kernels vectorized --surfaces surface --modes full,adjoint --tasks profile --warmup 2 --repeats 1 --output docs/benchmarks/fd_phase5_profile.json
.venv/Scripts/python.exe scripts/benchmark_fd_optimizations.py --levels baseline,final --grids 30x61 --variants optimized --kernels vectorized --surfaces constant --modes full --tasks timing --warmup 8 --repeats 9 --output docs/benchmarks/fd_phase5_constant_steady.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 30x61 --variants optimized --kernels vectorized --surfaces surface --modes adjoint --tasks profile --warmup 3 --output docs/benchmarks/fd_phase5_final_profile.json
```

`--surfaces neural` includes a deterministic 2–4–1 tanh network, bounded
sigmoid volatility, squared NPV loss, and all 17 parameter VJPs in the timed
work. Other surfaces differentiate NPV with respect to the supplied surface.
The baseline remains the actual TorchReal engine with Phase 4 band kernels;
it is not a Python contract loop or stock reverse-mode AD.

### Remaining work

Full AD still records the scalar upstream elimination recurrence. Adjoint
pricing still has scalar spline construction/limiting and financial setup;
both strategies remain much slower than stock forward-only pricing. The
retained per-step graph grows with the trajectory. Next prioritize vectorizing
spline slope/coefficient construction and its piecewise monotonicity filter,
with the existing spline as an oracle and independent VJP/difference checks.
For full AD and eventual batching, investigate a tensor-level band solve against
the upstream recurrence, dense solutions, and first/higher derivatives; define
pivot behavior explicitly before changing a default. A wider step VJP and
checkpointing remain subsequent experiments. The existing
GPU tensor primitives do not establish GPU option-pricing throughput. Preserve
exact event schedules when coordinating contracts; any replacement rollback
engine or other major architectural pivot still requires discussion.

## Phase 4: batched tensor kernels

The objective is efficient batched option pricing **and coefficient/neural
VJPs on GPU**, using tensor-level parallelism within the existing QuantLib
integration. Single-contract scalar overhead does not determine the throughput
of that eventual system. This milestone delivers vectorized CPU pricing hooks
and batched CPU/CUDA numerical primitives; complete batched pricing and a GPU
rollback remain subsequent milestones. The Phase 3/2/1 sections below are
historical snapshots.

### Implemented boundary and batch contract

`cpp/tensor_kernels.cpp` implements band application, row scaling (`mult`),
and assembly (`axpyb`) using whole-vector ATen arithmetic. Three narrow hooks
in `TripleBandLinearOp` are generated into `tensor-kernels.patch`, separate
from compatibility, adjoint, and profiling patches. The original scalar
implementations remain selectable. The pinned upstream checkout is untouched;
no stock-double QuantLib library or XAD implementation is linked or copied.

`fd_vanilla(..., kernels="vectorized")` selects all three hooks by default;
`kernels="scalar"` retains upstream arithmetic, and `kernels="apply"` isolates
application. This is independent of `mode="full"` or `mode="adjoint"`. A
thread-local RAII scope selects kernels for each synchronous pricing call.
The financial objects, mesher, original forward/transpose solve algorithms,
rollback, event ordering, exercise branch policy, and final interpolation
remain in use. Tensor operations retain higher derivatives in full mode;
the existing per-solve adjoint remains first-order only.

The application hook uses QuantLib's actual left/right neighbourhood indices,
including reflected endpoints. It does not assume the endpoint off-diagonals
are zero. Packing checks only whether values originated as passive C++
constants, never `requires_grad` or grad mode. An all-passive array can be
materialized in one allocation; tensor inputs are stacked with their graphs
intact. Unpacking uses one `unbind` node and rebinds scalar handles. All
assembly sources are packed before outputs are rebound, preserving upstream
self-alias cases without in-place tensor mutation. Pack/unpack have separate
profiler scopes and a dedicated roundtrip microbenchmark.

Two public numerical primitives establish the batch contract:

| Primitive | Input shapes | Meaning |
| --- | --- | --- |
| `tridiagonal_apply(lower, diagonal, upper, rhs)` | `(..., N-1)`, `(..., N)`, `(..., N-1)`, `(..., N, R)` | Apply independent or shared band matrices to R RHS vectors |
| `band_axpyb(a, x, y, b)` | scales `(..., N)` or `(..., 1)`; bands `(..., 3, N)` | Assemble `y + a*x + diag(b)` in lower/diagonal/upper row storage |

Leading dimensions broadcast exactly as PyTorch batch dimensions. A `(N,)`
diagonal can be shared by `(B,N,R)` states; a `(B,N)` diagonal assigns a
different operator to each batch member. Gradients reduce over shared batch
and RHS axes automatically. N must be at least two; the final RHS axis is
explicit even for R=1. Noncontiguous and expanded views are supported.
Row-packed assembly retains endpoint entries; compact tridiagonal application
uses lower[...,1:] and upper[...,:-1].

These primitives use dense strided float64 tensors on a common CPU or CUDA
device. They perform no per-element host reads, finite-value scans, or CUDA
synchronizations. Nonfinite values propagate as in PyTorch arithmetic. ATen
dispatches CUDA operations through the installed PyTorch wheel; no custom CUDA
toolkit/build or CUDA TorchReal scalars are introduced. CUDA support of these
primitives is not CUDA support for `fd_vanilla` or either solve API. There is
no new batched option-pricing API hidden behind a Python loop.

### Validation and measurement protocol

The Windows native build and `python -m pytest -q` passed **179 tests in
46.65 seconds**, with no skips. The 51 additional cases cover nonsymmetric
dense-reference application and VJPs for all bands and RHS values, shared
coefficients, multiple batch axes, cross broadcasting, strided/expanded
views, first/second finite differences, and no input mutation. Assembly is
checked against independent dense matrices. Hook tests cover nonzero
reflected endpoints at N=2 and N=9, empty/scalar/vector axpyb scales, row
scaling, and output aliases of both source operators. CUDA batched assembly
and application values/VJPs match an independent dense CPU reference.

Engine checks compare scalar/application-only/all-vector paths for nonuniform
surfaces through European, American, and Bermudan exercise, including a cash
dividend on an exercise date, the exposed time schemes, and both derivative
modes. Directional coefficient differences supplement the independent stock
price/vega, neural-gradient, state-lifetime, and full-mode second-derivative
tests retained from earlier phases. This evidence is local Windows execution,
not a claim of an observed remote Linux CI run.

`scripts/investigate_fd.py` adds `--kernels` and `--contracts` to the existing
isolated timing/memory/graph/profile harness. Its default remains `scalar` so
historical Phase 3 commands continue to select their original numerical path.
`--contracts bermudan` includes a coincident dividend/exercise event. Stock
timing/reference comparisons in the harness remain the European constant-vol
workload; event correctness is covered by the tests. Warmups and every timed
sample remain recorded, including the known cold/warm transition.

`scripts/benchmark_tensor_kernels.py` measures a pipeline of row-band assembly
and application over B systems and R RHS vectors. It compares one batched
invocation with a loop of B tensor invocations, using one shared stencil and
independent per-system coefficients/states. It records inference and
forward/backward/cleanup separately, systems/second and RHS-vectors/second,
and validates all outputs and VJPs in separate workers. Each timed backend and
gradient mode has its own process. CUDA wall times synchronize
at boundaries. Device-resident results and pageable H2D + inference + D2H
results are distinct. Memory workers are fresh processes, separate from
correctness graphs, timing, and warmups; CPU OS high water and CUDA allocator
peaks have their respective limitations. Kernel throughput is not contracts
priced per second, and the loop reference is not stock QuantLib.

### Measured CPU pricing improvement

The [30/61 kernel ablation](../benchmarks/fd_phase4_kernels.json) uses three
warmups and five timed samples per fresh process, with one intra-op and one
inter-op thread. Both Phase 3 constant optimizations remain enabled. Forward
includes financial-object/engine construction. Total includes graph cleanup;
median phase times need not sum to the median total.

| Surface | Mode | Forward ms, scalar → vectorized | Backward ms, scalar → vectorized | Total ms, scalar → vectorized | Total speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Constant | full | 1059.46 → 724.99 | 2238.56 → 1428.55 | 3391.04 → 2225.59 | 1.52× |
| Constant | adjoint | 951.85 → 703.58 | 1776.56 → 1208.72 | 2795.45 → 1934.23 | 1.45× |
| Nonuniform 3×5 | full | 1352.53 → 1019.25 | 2833.95 → 2041.10 | 4328.06 → 3110.08 | 1.39× |
| Nonuniform 3×5 | adjoint | 1211.09 → 875.86 | 2280.15 → 1445.48 | 3574.65 → 2347.87 | 1.52× |

Application alone yields constant-surface totals of 2779.10 ms (full) and
2514.17 ms (adjoint); adding assembly supplies a further measured improvement.
Graph-disabled constant inference falls from 96.51 to 76.45 ms; nonuniform
inference falls from 121.72 to 100.43 ms. All 27 cross-process/reference checks
pass: prices are identical across kernel/mode choices and the maximum
coefficient-gradient difference is `1.42e-13` (approximately). These are
single-contract CPU measurements, still far from efficient production pricing.
They establish a useful local improvement, not the eventual GPU throughput.

The [event workload](../benchmarks/fd_phase4_events.json) uses the same 30/61
grid and nonuniform surface for a Bermudan put with a cash dividend on an
exercise date. Adjoint total falls from 3704.89 to 2339.63 ms (1.58×), using
three warmups and five samples. Prices are identical and the maximum surface
VJP difference is `1.29e-14` (approximately). This supplements event correctness
tests with a measured example; it does not characterize every event schedule.

The same-run stock forward median is 0.1712 ms. The vectorized constant
inference remains about 447× slower and differentiated forward about
4,100–4,200× slower. There is no stock reverse-mode timing baseline. These
ratios concern the current single-contract bridge and scalar remainder.

The separate [memory/graph workers](../benchmarks/fd_phase4_memory.json) use one
cold forward/backward per memory case, and independent graph census processes:

| Surface | Mode | Peak working-set growth MiB, scalar → vectorized | Reachable NPV graph nodes, scalar → vectorized |
| --- | --- | ---: | ---: |
| Constant | full | 124.62 → 84.29 | 81,138 → 43,956 |
| Constant | adjoint | 87.10 → 45.73 | 57,575 → 20,393 |
| Nonuniform 3×5 | full | 142.62 → 102.82 | 98,094 → 60,912 |
| Nonuniform 3×5 | adjoint | 107.89 → 65.13 | 74,531 → 37,349 |

OS high-water growth includes native allocations and allocator retention and
is not a unique live-tensor memory measure. These are single cold measurements,
not repeated memory medians; graph counts describe the returned NPV graph.

The [6/21 nonuniform profiles](../benchmarks/fd_phase4_profile.json) show an
important limit of these boundaries. In full-mode forward, `aten::mul` calls
fall from 6,482 to 4,526 and `aten::add` from 2,495 to 1,197, but conversions
introduce stack/unbind/view operations. Total recorded ATen calls **increase**
from 15,541 to 20,342 forward and from 13,959 to 18,049 backward. Adjoint
totals similarly rise from 17,815/17,806 to 22,616/21,896. The hooks reduce
arithmetic graph size and measured runtime, but do not eliminate scalar/vector
conversion overhead. These instrumented call counts are not timing estimates;
the uninstrumented timing table above supplies performance evidence.

### Batched CPU/CUDA throughput and conversion costs

The [isolated batch experiment](../benchmarks/tensor_phase4_isolated.json) uses
N=257, B=1/16/256, R=1/8, float64, three warmups and seven timed repetitions.
The host has an NVIDIA GeForce RTX 4060 Laptop GPU and PyTorch 2.10.0+cu126;
CPU measurements use one thread. Each backend/gradient mode has a fresh process,
and correctness comparisons use separate workers. The table reports medians
for the **assembly + application pipeline**, including backward and cleanup,
with inputs already on the selected device:

| Systems B | RHS R | Batched CPU total ms | Batched CUDA total ms | CPU/CUDA time ratio | CUDA systems/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.2465 | 0.7369 | 0.33× | 1,357 |
| 1 | 8 | 0.2809 | 0.8809 | 0.32× | 1,135 |
| 16 | 1 | 0.3316 | 0.7217 | 0.46× | 22,170 |
| 16 | 8 | 0.7677 | 0.8384 | 0.92× | 19,084 |
| 256 | 1 | 1.7402 | 0.7722 | 2.25× | 331,520 |
| 256 | 8 | 10.9095 | 0.9121 | 11.96× | 280,671 |

At B=256/R=8, CUDA processes about 2.25 million RHS vectors/s for this
pipeline and VJP. These are numerical-system/RHS rates, **not option pricing
rates**. A solve, time trajectory, coefficient sampling, exercise/dividends,
and interpolation are absent. CPU batching itself helps: at B=16/R=1 the CPU
loop total is 3.8525 ms versus 0.3316 ms batched (11.6×). All tested batch/loop
values agree exactly; maximum VJP differences are below `9e-14`, reflecting
different reduction ordering for the shared stencil.

Transfer cost changes the crossover. At B=256/R=8, graph-disabled inference
is 2.8528 ms on CPU, 0.3069 ms on device-resident CUDA inputs, and 1.4604 ms
including pageable host-input upload and output download. At B=256/R=1,
resident CUDA inference is 0.2342 ms versus CPU 0.5261 ms, but including
transfers raises CUDA to 0.6917 ms. Transfer measurements here are inference
only; a full transfer-inclusive neural-loss/VJP workload remains future work.
This provides concrete support for keeping coefficients and rollback state
resident on GPU, without implying a universal GPU advantage.

The separate [B=256/N=257/R=8 memory probe](../benchmarks/tensor_phase4_memory.json)
shows a tradeoff: batched CUDA peak allocated growth above resident inputs is
30.63 MiB, versus 26.05 MiB for the loop. Batched evaluation materializes larger
intermediate tensors simultaneously; faster throughput does not imply lower
device memory. CPU OS peak growth is 82.71 MiB batched versus 91.25 MiB loop.
Each is one cold forward/backward measurement, not a memory median. CUDA
allocated/reserved counters exclude allocations outside PyTorch's allocator.

The [exploratory sweep](../benchmarks/tensor_phase4_throughput.json) additionally
contains N=61, but ran validation and all backends/modes within each case's
process. Allocator/process history materially affected results: for example,
B=256/N=257/R=8 CPU batched total was 52.57 ms there, versus 10.91 ms in the
isolated run. Its CUDA total was 10.27 ms versus 0.912 ms isolated. Those pilot
timings are retained for transparency and are not the headline throughput
comparison. The exact cause of the host's state dependence is unresolved.
Even isolated results are limited to this host, dtype, workload and repeat
count; preserve all warmups and samples when reproducing them.

The isolated report also records a CPU application-boundary microbenchmark
at N=257. Graph-disabled scalar/vector hooks take 2.624/1.688 ms; the resident
tensor primitive takes 0.0393 ms. Hook timing includes construction of a test
QuantLib mesher/operator and scalar input views, so the gap cannot all be
attributed to packing. A separate unbind/stack roundtrip takes 0.2185 ms
without recording and 0.7822 ms with backward/cleanup. These measurements
motivate keeping arrays in tensor form across several kernels, rather than
assuming a vector kernel removes conversion costs at every boundary.

Reproduce the main measurements with the pinned environment:

```powershell
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 30x61 --variants optimized --kernels scalar,apply,vectorized --tasks timing --warmup 3 --repeats 5 --output docs/benchmarks/fd_phase4_kernels.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 30x61 --variants optimized --kernels scalar,vectorized --modes full,adjoint --tasks memory,graph --output docs/benchmarks/fd_phase4_memory.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 6x21 --variants optimized --kernels scalar,vectorized --surfaces surface --modes full,adjoint --tasks profile --output docs/benchmarks/fd_phase4_profile.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 30x61 --variants optimized --kernels scalar,vectorized --surfaces surface --contracts bermudan --modes adjoint --tasks timing --warmup 3 --repeats 5 --output docs/benchmarks/fd_phase4_events.json
.venv/Scripts/python.exe scripts/benchmark_tensor_kernels.py --devices cpu,cuda --nodes 257 --output docs/benchmarks/tensor_phase4_isolated.json
.venv/Scripts/python.exe scripts/benchmark_tensor_kernels.py --devices cpu,cuda --batches 256 --nodes 257 --rhs 8 --tasks memory --output docs/benchmarks/tensor_phase4_memory.json
```

### Next milestone toward GPU pricing

1. **Batched band solve with a validated VJP.** Reuse the batch/RHS contract
   above, preserve the upstream scalar solve as an oracle, and check arbitrary
   nonsymmetric systems against dense solves and independent differences.
   Explicitly define pivot failures and first/higher-derivative support.
   Compare CPU and CUDA throughput across B, N and R, including factorization
   reuse for multiple RHS. A serial spatial recurrence may parallelize across
   B and R; its CUDA launch cost must be measured against a fused or parallel
   band solver before selecting an implementation.
2. **Keep numerical state in tensors across a whole step.** The current
   engine converts at each hook and its solves/sampling/rollback still use
   scalar tensors. Extend narrow boundaries to batched coefficient sampling,
   assembly, apply, and solve, retaining QuantLib-derived stencils and the
   existing financial/event semantics. Measure a custom step VJP and then
   checkpointing, including their recomputation cost. Keep coefficient and
   neural graphs connected across the boundary.
3. **Coordinate compatible contracts explicitly.** Use a B axis for distinct
   operators/meshes, and an R axis only for payoffs that genuinely share an
   operator, grid, and stepping/event plan. Different strikes usually change
   the current automatic mesh, so they cannot automatically be treated as
   extra RHS vectors. Group contracts by compatible grid shape, numerical
   scheme and exact rollback/event schedule; coordinates and coefficients may
   differ by batch member. Preserve each contract's original steps, damping,
   dividend/exercise ordering and interpolation. A union schedule that adds
   time steps is a numerical change and cannot be silently substituted.
4. **Measure device-resident option batches and neural loss VJPs.** Upload
   fixed mesh/stencil/event metadata once, retain rollback states and model
   coefficients on device, and return prices or required adjoints at an
   explicit boundary. Compare against independent single-contract QuantLib
   results, dense/difference VJPs, serial CPU pricing, and batched CPU kernels.
   Report steady-state prices/second, weighted-loss VJP throughput, peak
   memory, warmup/compilation costs if any, and transfer-inclusive timings for
   B=1,16,256 and larger feasible batches, with R=1 and multiple RHS. Include
   nonuniform neural fields and event-heavy contracts, not only constant-vol
   European workloads. Report the GPU crossover or its absence.

This is a staged extension of Approach A. Synchronizing batches across
independent engine calls requires an explicit coordinator; changing the scalar
type alone cannot provide it. A complete replacement rollback engine or other
major architecture pivot still requires discussion. The present CPU overhead
is evidence about the current scalar boundaries, not a verdict on eventual
batched GPU practicality.

## Phase 3: CPU performance investigation

The priority is reducing the cost of the integrated engine while preserving
QuantLib's financial objects, discretization, rollback, event ordering, and
interpolation. Approach A remains the implementation. No stock-double QuantLib
library, replacement pricing engine, GPU path, or XAD code has been introduced.
The Phase 2 and Phase 1 sections below describe their historical snapshots.

### Implemented changes and their differentiation contracts

`TorchReal` now stores either a native double constant or a CPU float64 tensor.
Only construction from a C++ number creates a passive value. Arithmetic and
math functions on two passive values use native double operations; mixing a
tensor with a constant dispatches to LibTorch. **Every tensor constructor keeps
the tensor**, regardless of `requires_grad` and grad mode. There is no test of
`requires_grad` that demotes coefficients into doubles. Compound assignments
still rebind values, and tensor arithmetic never mutates aliases in place.
Materializing a constant as a tensor returns a new handle by value, avoiding
mutable lazy caches shared by copies. This removes allocations for market,
mesh, stencil, and other constant computations without changing QL_REAL or
recompiling any QuantLib dependency with a different scalar type.

`TensorLocalVol` lazily caches graph-connected coefficient views for indices
actually visited by the engine. The cache belongs to one pricing call and is
discarded with its local-volatility object; returned graphs own the views.
Singleton axes and flat-extrapolation intervals use the selected endpoint
directly, avoiding repeated interpolation of the same sample with zero weights.
Interior intervals retain the original bilinear expression and branch policy.
This does not batch or cache whole prices, change event ordering, or detach a
neural surface. Construction scales with visited coefficients, not the size of
an otherwise unused surface.

Two process-wide diagnostic switches reproduce each ablation. Set them **before
importing QuantLibTorch**; defaults are `1` and values must be `0` or `1`:

| Variant | `QLT_PASSIVE_CONSTANTS` | `QLT_SAMPLE_CACHE` |
| --- | ---: | ---: |
| eager (original scalar behavior) | 0 | 0 |
| constants | 1 | 0 |
| cache | 0 | 1 |
| optimized (default) | 1 | 1 |

These are diagnostic controls, not new differentiation modes. Both `full` and
`adjoint` still use the same upstream engine and solve algorithm. The adjoint
still performs scalar tensor arithmetic inside its unrecorded forward and
transpose solves; optimizing constants alone does not remove that cost.

### Reproducible measurement and attribution

`scripts/investigate_fd.py` runs each case/task in a fresh subprocess with one
intra-op and one inter-op PyTorch thread. It measures constant and deterministic
nonuniform 3x5 surfaces, graph-disabled inference, full AD, and per-solve AD.
Timing includes market/engine construction and separately reports forward,
backward, graph cleanup, raw warmups, and every timed sample. OS peak working
set, reachable autograd nodes, repeated tensor saves, and profiler traces run
in **separate processes**, so those measurements do not contaminate timing.
Prices and VJPs are checked across configurations; constant surfaces also use
the stock QuantLib process and its fixed-mesh volatility bump.

Native profiler scopes cover the bridge, engine, operator assembly, local-vol
sampling, band assembly/application/solve, rollback, events, dividend handling,
and interpolation/theta work. `profiling-scopes.patch` records this instrumentation
separately from `math-adl.patch` and `adjoint-hook.patch`. It changes no numerical
expressions. Scope durations are nested, and inclusive durations must not be
summed. The report also supplies outermost occurrences of each scope to avoid
double-counting the adjoint hook's re-entry into `solve_splitting`. Profiler times
include instrumentation overhead; use the timing task for performance ratios.

Process memory is the OS working-set high water after a cold first forward and
backward, including native allocations, interpreter/LibTorch memory and allocator
retention. The report records pre-price, post-forward, post-backward and
post-cleanup values. Peak growth subtracts the pre-price high water; it is not
an exact incremental allocation count or graph-storage size. Saved-tensor bytes
remain a different measure that counts repeated and transient saves, including
upstream Greeks the Python API does not return. Reachable nodes count the graph
of the returned NPV only.

```powershell
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 30x61 --variants eager,constants,cache,optimized --output docs/benchmarks/fd_phase3_ablation.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 10x31,60x121 --surfaces constant --modes full,adjoint --output docs/benchmarks/fd_phase3_scaling.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 6x21 --modes full,adjoint --tasks profile --output docs/benchmarks/fd_phase3_profile.json
.venv/Scripts/python.exe scripts/investigate_fd.py --grids 30x61 --variants optimized --surfaces constant --modes full,adjoint --tasks timing --single-thread-backward --output docs/benchmarks/fd_phase3_single_thread_backward.json
```

Add `--trace-dir build/phase3_traces` to the profiling command to export Chrome
traces. Trace files are optional generated artifacts; the recorded JSON contains
operator counts and scope summaries without them.

### Correctness and limitations

After the Phase 3 native build, `python -m pytest -q` reports **128 passed in
50.63 seconds** on the Windows host. This includes the original 124 tests plus
mixed constant/tensor arithmetic VJPs, numerical first/second derivative checks,
copy/alias invariants, and a check that tensor inputs stay tensor-backed under
`no_grad` with and without `requires_grad`. Existing tests retain stock pricing,
nonuniform surface and neural directional differences, noncontiguous inputs,
exercise/dividends, graph lifetimes, and higher-derivative restrictions.
The pinned QuantLib checkout remains clean; the native link still includes only
LibTorch and platform libraries. Remote Linux execution is not claimed.

The original implementation was remeasured before editing; raw measurements are
in [fd_phase3_original.json](../benchmarks/fd_phase3_original.json). At 30 steps / 61
points, full forward/backward medians were 1.712 / 3.102 seconds, and adjoint
medians were 1.605 / 2.707 seconds. Stock forward was 0.174 ms. This independently
reproduces the Phase 2 limitation on this host.

The controlled [30/61 ablation](../benchmarks/fd_phase3_ablation.json) uses two
warmups and three timed samples per case. These are European calls under
Douglas with two damping steps; event-heavy performance is not measured here.
All 77 cross-process price/VJP comparisons pass: prices are identical and the
largest coefficient-gradient difference is `2.14e-13` (rounding upward).

| Surface | Mode | Forward ms, eager → optimized | Backward ms, eager → optimized | Total speedup, including cleanup | Peak working-set growth MiB, eager → optimized |
| --- | --- | ---: | ---: | ---: | ---: |
| Constant | full | 1755.53 → 1056.93 | 3136.90 → 2220.53 | 1.49× | 171.77 → 124.74 |
| Constant | adjoint | 1661.01 → 960.53 | 2688.16 → 1760.06 | 1.62× | 135.81 → 87.23 |
| Nonuniform 3×5 | full | 1850.05 → 1243.71 | 3130.53 → 2579.93 | 1.31× | 172.25 → 142.61 |
| Nonuniform 3×5 | adjoint | 1769.59 → 1147.92 | 2764.98 → 2110.72 | 1.39× | 135.10 → 107.77 |

For constant volatility, passive constants alone improve the full-mode total
from 5.020 to 4.589 seconds; sample caching alone reduces it to 3.615 seconds;
both reduce it to 3.362 seconds. The respective adjoint totals are 4.465,
4.050, 3.126, and 2.764 seconds. Gains are smaller for the nonuniform surface
because interpolation remains necessary. Both surfaces use the same pricing
implementation and retain the QuantLib engine.

The constant-surface adjoint graph shrinks from 91,794 to 57,575 reachable nodes;
its peak working-set growth falls by 35.8%. Repeated saved bytes fall from
586,312 to 301,984. Passive constants alone leave reachable node counts unchanged
while substantially reducing saves, reinforcing that saved bytes are a poor
proxy for graph memory or runtime. The original full mode saves only 760,544
counted bytes but grows the process high water by 171.77 MiB.

Graph-disabled inference improves from 177.03 to 97.70 ms for the constant
surface and from 197.34 to 115.41 ms for the nonuniform surface. The stock
constant-volatility forward median in the same ablation is 0.190 ms. Optimized
inference is still approximately **514×** slower, and differentiated forward
execution is still approximately **5,000–5,600×** slower. There is no stock
reverse-mode timing comparison. Neither optimization makes this a production
pricing implementation.

Cold and warm behavior differ sharply on this Windows host. For example,
optimized full mode's first forward is 0.175 seconds but its warmed median is
1.057 seconds. The cause of this transition is not established; it must not be
discarded as an isolated outlier or attributed to an algorithm without further
evidence. Raw warmups are preserved, and the performance ratios above compare
warmed measurements. Profiling overhead and allocator/process state also limit
generalization to other hosts.

The [larger-grid check](../benchmarks/fd_phase3_scaling.json) uses 60 time steps /
121 points, with the same warmup/repeat counts and a constant surface:

| Mode | Total seconds, eager → optimized | Speedup | Peak working-set growth MiB, eager → optimized | Reachable nodes, eager → optimized |
| --- | ---: | ---: | ---: | ---: |
| full | 18.779 → 12.872 | 1.46× | 603.92 → 426.25 | 439,387 → 309,798 |
| adjoint | 16.697 → 10.580 | 1.58× | 466.09 → 281.02 | 349,044 → 219,455 |

The scaling report's 30 cross-process checks pass with maximum price difference
`3.56e-15` and coefficient-gradient difference `3.56e-13`. Its 10/31 cases are
also retained, but their timing ratios are not used as evidence of algorithmic
speedup: some small cases remain in the fast regime throughout their samples,
whereas others transition. A fixed warmup count does not establish a universal
steady state. Each process-memory result is one cold execution, not a repeated
memory median.

A separate [backward-thread diagnostic](../benchmarks/fd_phase3_single_thread_backward.json)
disables PyTorch's multithreaded backward in the worker only. Full-mode backward
is 2.192 seconds versus 2.221 seconds in the main experiment; the two later
adjoint backward samples are 1.753 seconds versus a 1.760-second main median.
The forward transition persists (adjoint timed samples: 0.146, 0.510, 0.946
seconds). This does not explain or remove the variability, and no caller thread
policy is changed by the library.

### What the operator traces establish

The [CPU profiles](../benchmarks/fd_phase3_profile.json) use 6 steps / 21 points,
two warmups, and eight separate case processes. Recorded ATen calls are:

| Surface | Mode | Forward calls, eager → optimized | Backward calls, eager → optimized |
| --- | --- | ---: | ---: |
| Constant | full | 48,893 → 13,902 | 24,443 → 12,172 |
| Constant | adjoint | 52,115 → 16,176 | 32,601 → 16,019 |
| Nonuniform 3×5 | full | 54,418 → 15,541 | 24,441 → 13,959 |
| Nonuniform 3×5 | adjoint | 57,640 → 17,815 | 32,599 → 17,806 |

For constant full mode, recorded `aten::scalar_tensor` calls fall from 9,105
to zero and `aten::item` calls from 7,308 to 911. Sampling views (`aten::select`)
fall from 1,512 to 2 for the constant surface and from 1,512 to 30 for the 15
nonuniform coefficients. This directly confirms removal of repeated scalar
materialization and sample-view work. It does not equate operator-call counts
with allocation counts or wall-clock speedup.

In the optimized nonuniform full-mode trace, operator assembly takes 13.00 ms
of the 35.68 ms bridge scope, including 4.71 ms of sampling. Band solves take
7.88 ms; rollback as a whole takes 26.08 ms. These nested scopes identify work
outside the solve that a solve-only optimization would leave in place. They
are single instrumented observations, not universal cost percentages. Profiler
times also show the host's differing execution regimes, so cross-case profiler
timings are not used to calculate the performance ratios above.

### Architectural direction after this investigation

The measured reductions help both differentiation strategies, but scalar
dispatch and large elementary-operation graphs remain the central limitation.
The next performance work should preserve the integrated engine and proceed in
this order:

1. Prototype vector operations for coefficient sampling, band assembly, and
   band application behind narrow upstream hooks. Convert an array at a kernel
   boundary rather than dispatching a tensor operation for each scalar. Retain
   the current upstream full-AD path as a reference, with nonsymmetric dense
   VJP checks, independent coefficient differences, event cases, and separate
   conversion-cost measurements. This is a reversible kernel experiment within
   Approach A; a complete tensor-engine replacement would require discussion.
2. Measure a coarser custom VJP boundary spanning a step, including assembly
   and the solve. The existing per-solve adjoint still records assembly and
   rollback arithmetic and executes tensor operations inside its unrecorded
   forward/transpose solves. Any passive numerical execution must be explicit
   inside a validated custom-autograd boundary, never an implicit coefficient
   conversion in `TorchReal`.
3. Evaluate trajectory checkpoint/recompute boundaries for memory. Preserve
   dividend/exercise ordering, active-set behavior, theta snapshots, and final
   interpolation. Report the time cost alongside OS peak memory and include
   nonuniform neural surfaces and event-heavy contracts.
4. Investigate the host's persistent cold/warm transition with longer isolated
   repetitions and native allocation traces. Treat this separately from
   numerical improvements. Only after CPU kernels and batching have measured
   benefits should CUDA or two-factor expansion become the priority.

## Phase 2: actual finite-difference vanilla engine

The current implementation compiles and runs QuantLib v1.40's
`FdBlackScholesVanillaEngine`, attached to its `VanillaOption`, with
`QuantLib::Real = TorchReal`. This extends Approach A; it does not replace
QuantLib's PDE assembly or rollback with a new tensor PDE implementation.
The original prototype APIs remain available. The Phase 1 findings below are
retained as historical evidence about that separate path.

```text
PyTorch network -> sampled CPU float64 local volatility
  -> custom TensorLocalVol : QuantLib::LocalVolTermStructure
  -> GeneralizedBlackScholesProcess with external local volatility
  -> VanillaOption / FdBlackScholesVanillaEngine
  -> FdmBlackScholesMesher + FdmLogInnerValue + vanillaComposite
  -> FdmBlackScholesOp / FirstDerivativeOp / SecondDerivativeOp
  -> Fdm1DimSolver / FdmBackwardSolver / upstream time-stepping scheme
  -> TripleBandLinearOp (full AD or per-solve custom VJP)
  -> upstream dividend/exercise conditions and spline interpolation
  -> scalar NPV -> loss -> network gradients
```

### Genuine upstream reuse and financial scope

The bridge constructs actual QuantLib financial objects and invokes
`VanillaOption::NPV()`. The engine's `calculate()` implementation is upstream
code. In particular:

- `FlatForward`, or `DiscountCurve` with log-linear discount interpolation,
  supplies each risk-free/dividend yield curve. `Actual365Fixed` computes
  times, and `GeneralizedBlackScholesProcess` supplies forward rates and local
  volatility to the operator. Quotes and term-structure handles are real
  QuantLib objects, created anew for each pricing call.
- `FdmBlackScholesMesher` selects the domain and concentrates the grid around
  the strike. `FdmLogInnerValue` performs upstream payoff averaging.
- `FdmStepConditionComposite::vanillaComposite` creates actual stopping times,
  `FdmDividendHandler` cash-dividend jumps, and American/Bermudan step
  conditions. Coincident dividend/exercise dates follow upstream ordering.
- `FdmBlackScholesOp` samples local volatility at step midpoints and spatial
  nodes, then builds its operator with QuantLib's first/second derivative
  stencils and curve forward rates. QuantLib retains its boundary behavior.
- `FdmBackwardSolver` performs rollback, including stopping-time subdivisions
  and damping; `Fdm1DimSolver` retains its theta snapshot and monotonic cubic
  natural-spline interpolation. The engine still computes its upstream
  delta/gamma/theta results internally; this Python API returns NPV only.

`fd_vanilla` supports calls/puts, European exercise, American exercise starting
at valuation, and dated Bermudan exercise. Curves are flat continuously
compounded rates or explicit dated discount factors. Cash dividends are fixed,
nonnegative amounts under the spot model. Exposed schemes are Douglas,
Crank–Nicolson, and implicit Euler. Valuation/maturity and event dates are
explicit `datetime.date` values, with Actual/365 Fixed and unadjusted dates.
Calendar adjustment, alternate day counters, forward-start American exercise,
escrowed dividends, quanto, barriers, and other instrument types are not
exposed or claimed as tested. Compiling their dependencies is not evidence of
supporting those features.

QuantLib's evaluation date is set under a native mutex and restored with
an RAII guard even on exceptions. The guard preserves the raw null date too:
upstream `SavedSettings` converts that default to today's explicit date, which
would otherwise anchor a formerly floating evaluation date. Objects and caches are private to each
call; returned autograd graphs retain tensors, not pointers to those financial
objects. This is a functional pricing interface, not a persistent market-object
binding or a general concurrent QuantLib API.

### Custom code and differentiation boundary

`cpp/fd_bridge.cpp` contains market-object construction, input validation, and
`TensorLocalVol`. The latter interpolates sampled volatility bilinearly in
time and **log absolute spot**, with flat extrapolation and constant singleton
axes. This interpolation policy is custom QuantLibTorch code; it is not a claim
to reproduce QuantLib `FixedLocalVolSurface`'s spot-linear interpolation.
The Python layer validates fixed inputs and calls the native operation. It
does not assemble operators, time steps, dividends, or exercise projections.

Only the supplied surface is differentiable in this API. Financial inputs,
coordinates, and mesh settings reject tensors rather than silently discarding
their gradients. A separate, fixed `BlackConstantVol(mesh_volatility)` controls
the automatic mesher; the external local-volatility term structure controls
the pricing operator. Stock comparisons and finite differences hold this
mesher volatility fixed. This avoids conflating coefficient VJPs with mesh
movement, and permits neural surfaces with a stable discretization. Users must
check domain and mesh convergence for their surface: a mesh sized for .2 is
not automatically adequate for every local-volatility field.

Full mode records LibTorch arithmetic through operator assembly, band solves,
rollback, dividend interpolation, exercise projections, and final interpolation.
Comparisons select discrete branches; derivatives are conditional on those
branches, including exercise active sets and spline monotonicity limiters.

Adjoint mode uses the same forward engine and the **same upstream
`TripleBandLinearOp::solve_splitting` forward algorithm**. A small, separately
recorded hook calls `cpp/adjoint.cpp`; its custom C++ autograd function disables
recording and re-enters the original solve with the hook temporarily disabled.
For `A = a*L + b*I`, assembly of A's three diagonals happens outside that custom
function so coefficient and scale dependencies remain connected. The function
saves the diagonals and solution, solves `A.T * lambda = grad_output` using
QuantLib's `TridiagonalOperator`, and returns the diagonal/RHS VJPs. The original
full-mode solve remains untouched. The hook is thread-local, scoped with RAII,
and restricted to a one-dimensional operator layout.

Adjoint mode explicitly rejects `create_graph=True`. It supports first-order
VJPs, not double backward. Assembly, event handling, and interpolation still
record ordinary autograd operations in both modes. This is a per-solve adjoint,
not a checkpointed whole-engine adjoint: memory grows with grid size and time
steps. The original Python `european_call` continues to use custom Python PDE
assembly/boundaries and the older `TridiagonalOperator` interface; it is not the
new engine implementation.

### Compatibility, provenance, and build

The source remains pinned to QuantLib v1.40 commit
`06edddbdb85f006c43a449b5fc846a00c67477f3`. The explicit closure in
`cmake/quantlib_sources.cmake` compiles 80 upstream translation units with one
scalar type and links only LibTorch libraries. `QL_NO_AUTO_LINK` prevents an
implicit stock QuantLib library link; compile-time assertions check `Real`.
No QuantLibAAD/XAD implementation is used.

`scripts/prepare_quantlib.py` follows the source manifest's header closure and
generates an auditable overlay. Compatibility work includes ADL math lookup,
mixed scalar/constant `min`/`max`, and correcting `Null<Real>()` to the underlying
max-float sentinel. `QL_NULL_AS_FUNCTIONS` uses the upstream function form.
The expanded math wrapper includes operations required by Boost/QuantLib
dependencies. Explicit integral conversions for mesh/control quantities reject
differentiable, nonfinite, and out-of-range values; there is still no implicit
floating conversion. Compound arithmetic rebinds tensor handles without
in-place tensor mutation.

`math-adl.patch` records compatibility separately from `adjoint-hook.patch`.
The latter adds an include and a small dispatch after upstream RHS-size
validation. There are no changes to the forward pricing formulas. The clean
upstream checkout retains its original license and sources. Generated-file
reconciliation removes stale overlays within the build output; the checkout
is never patched in place. The closure is substantially broader than the
prototype's two patched headers and is not a whole-library compatibility claim.

Build against the same Python/PyTorch ABI used at runtime. The tested baseline
is PyTorch 2.10.0, CPU float64; this package now pins that version. The CPU build
also works with the host's CUDA-enabled wheel without a CUDA toolkit. A PCH
reduces repeated include cost, and `--jobs` controls compiler parallelism.

```powershell
.venv/Scripts/python.exe scripts/bootstrap.py
.venv/Scripts/python.exe scripts/build.py --jobs 2
.venv/Scripts/python.exe -m pip install --no-build-isolation --no-deps -e .
.venv/Scripts/python.exe -m pip install QuantLib==1.40
.venv/Scripts/python.exe -m pytest -q
ctest --test-dir build/native -C Release --output-on-failure
.venv/Scripts/python.exe examples/neural_fd_engine.py --mode full
.venv/Scripts/python.exe examples/neural_fd_engine.py --mode adjoint
.venv/Scripts/python.exe scripts/benchmark_fd.py --repeats 5 --output docs/benchmarks/fd_engine_cpu.json
```

The QuantLib wheel is only an independent reference dependency, loaded by
`tests/stock_quantlib_reference.py` in a **separate process**. It is never linked
to or imported by the TorchReal extension. The CI workflow installs this
pinned oracle and runs the same suite. Local Windows evidence does not imply
an observed remote Linux CI result. Editable/source installation is still
required; native wheel packaging remains deferred.

### Validation and measured limits

Local verification on 2026-10-31: the final Windows native build succeeded and
`python -m pytest -q` reported **124 passed in 73.75 seconds**, with no skips or
failures. The editable 0.2.0 installation also completed. Upstream QuantLib's
checkout remained clean. The full suite includes all 41 original tests.

Tests compare stock QuantLib prices and fixed-mesh volatility bumps across
European/American/Bermudan exercise, puts/calls, negative rates, discrete
dividends, nonflat dated curves, and the exposed schemes. A time-varying,
spot-constant surface is also compared with QuantLib `FixedLocalVolSurface`.
Nonuniform log-spot surfaces use full/adjoint agreement and independent central
directional differences, including a dividend on a Bermudan exercise date.
Neural-parameter pricing-loss gradients are checked by directional differences.
The band-solve hook is separately compared with `torch.linalg.solve` for
nonsymmetric matrices and arbitrary output weights, differentiating all three
diagonals, RHS, and scales `a,b`. Legacy solver/PDE tests remain in the suite.

The stock engine comparisons pass at absolute price tolerance `2e-10`; stock
central-bump volatility derivatives pass relative tolerance `3e-6` (absolute
`2e-6`). These are comparisons of the same discretization with a fixed mesh,
not exact continuous-model price claims. Full-mode second derivatives pass
`gradgradcheck`, and both modes pass a small sampled-surface `gradcheck`.
Additional tests cover noncontiguous views, no input mutation, scalar null and
integer-conversion invariants, exponent-math extremes, graph lifetime across
later pricing calls, evaluation-date restoration on success/error, concurrent
explicit-date calls, and explicit CUDA rejection.

For S=K=100, r=.03, q=.01, T=1, constant local vol .2, mesh vol .2, Douglas
stepping and two damping steps, measured European convergence is:

| Calculation | Call price | Parallel local-volatility derivative |
| --- | ---: | ---: |
| Analytic Black–Scholes | 8.8273212254 | 38.7151754159 |
| QuantLib engine, 10 time steps / 21 grid points | 8.9395431450 | 39.2776807733 |
| QuantLib engine, 60 time steps / 121 grid points | 8.8302824214 | 38.7298758165 |

The deterministic neural Bermudan-put example includes a cash dividend on an
exercise date. Both modes produce price `9.2947435045`, loss `0.6316172380`, and
network-gradient norm `7.0676973882`. This demonstrates the complete neural
coefficient → real pricing engine → loss gradient chain; it is not a calibration
or model-quality claim.

The controlled CPU run in [fd_engine_cpu.json](../benchmarks/fd_engine_cpu.json)
uses one PyTorch thread, five warmup-excluded samples, and separate timing and
save-count runs. Both modes return exactly the same prices at each measured
grid; VJPs agree to rounding precision. Forward time includes construction of
curves, surface, instrument, and engine in both implementations. The stock
process excludes interpreter/import startup from timing and uses its native
`LocalConstantVol`; TorchReal uses the tensor surface adapter, so the overhead
includes that adapter as well as graph recording and scalar dispatch.

| Steps / points | Mode | Forward median (ms) | Backward median (ms) | Saved tensor count | Counted saved bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 / 31 | full | 419.97 | 584.49 | 20,358 | 162,864 |
| 10 / 31 | adjoint | 403.26 | 514.66 | 14,535 | 128,552 |
| 30 / 61 | full | 1,716.98 | 3,119.76 | 95,068 | 760,544 |
| 30 / 61 | adjoint | 1,604.63 | 2,684.54 | 65,435 | 586,312 |

Stock QuantLib forward medians are **0.103 ms** and **0.183 ms**, respectively.
This scalar-tensor implementation is therefore thousands of times slower for
these small forward problems. It is a correctness/integration milestone, not
a production-performance result. At 30/61, per-solve adjoints reduce counted
saved bytes by approximately 23% and saved tensor count by 31%; assembly and
the rest of the engine still create substantial graphs. The first small-grid
full forward sample was unusually fast (69 ms vs the other samples around
420 ms); all samples are retained. These local measurements do not establish a
general performance ranking or model how costs scale to larger grids.

Saved-tensor statistics count all forward saves, including repeated and transient
saves, not live/unique storage, graph-node memory, or peak resident memory. Scalar substitution continues to
allocate and dispatch many small tensors. It does not provide tensor batching
or vectorize QuantLib loops. No GPU execution, GPU speedup, `torch.compile`
support, or two-factor differentiable engine is claimed.

### Milestones proposed at the end of Phase 2

1. Profile operator assembly, sampling, interpolation, and solve graphs at
   larger grids; introduce checkpoint/recompute boundaries through upstream
   rollback and retain exact event ordering. Measure process peak memory as
   well as tensor-save counts.
2. Extend the financial interface deliberately: selectable day counters and
   calendars, escrowed cash dividends, and carefully defined financial-input
   sensitivities with a fixed/reusable mesh. Test each against stock engines.
3. Prototype a batched tensor implementation of the band kernel behind the
   existing hook, retaining upstream assembly/engine behavior. Benchmark CPU
   and then CUDA with controlled workloads before making performance claims.
4. Extend to QuantLib's two-dimensional mesh/operator/ADI path for neural
   stochastic-volatility/SDE coefficients. Generalize adjoints to directional
   solves and mixed derivatives, and validate with a stock Heston baseline.
   The current one-dimensional hook intentionally rejects that case.

## Phase 1 evidence (historical prototype)

Status: a compiled numerical-kernel integration with two differentiation modes.
No architectural pivot has been made. The tested baseline is QuantLib v1.40 at
`06edddbdb85f006c43a449b5fc846a00c67477f3` and PyTorch 2.10.0 on CPU.

### What ran in Phase 1

```text
PyTorch network -> sampled volatility tensor -> Python PDE assembly
  -> compiled QuantLib TridiagonalOperator<TorchReal semantics>
  -> prices -> pricing loss -> PyTorch network gradients
```

`TridiagonalOperator` is not a C++ template: `QuantLib::Real` is changed globally
using `QL_INCLUDE_FIRST=quantlibtorch/torch_real.hpp` and `QL_REAL`. The build
compiles upstream `tridiagonaloperator.cpp` and `errors.cpp`, not a reimplementation
of their numerical algorithm. `cpp/bridge.cpp` translates tensor views to
QuantLib arrays and stacks the resulting scalars without detaching them. A
static assertion checks the scalar alias. Every QuantLib translation unit in
the shared library uses the same scalar; no stock QuantLib binary is linked.

`TorchReal` stores a zero-dimensional CPU double tensor. Arithmetic returns new
tensors and compound assignment rebinds handles. Copies therefore behave as
values without mutating tensors retained by autograd. Comparisons and error
printing read scalar values for control flow; they do not differentiate branches.
There is no implicit conversion back to double that could silently lose a graph.

The narrow numeric-limits specialization reports the underlying double limits;
in particular `epsilon()` is a constexpr double as required by QuantLib's
comparison implementation. This is not a full standard floating-point type.

### Phase 1 compilation experiment and compatibility change

The hook alone is insufficient for this kernel. Reproduce the negative probe:

```text
python scripts/build.py --build-dir build/unpatched --no-math-patch
```

MSVC fails with C2665 at `std::sqrt`, `std::fabs`, `std::log`, `std::exp`, and
`std::pow` calls in `ql/math/array.hpp` and `ql/math/comparison.hpp`: no overload
accepts `TorchReal`. No conversion to double was added to bypass this failure.

`scripts/prepare_quantlib.py` generates an overlay for these two headers:
import the standard math functions with `using std::...`, then make calls
unqualified so argument-dependent lookup finds the wrapper's overloads.
The generated diff is `build/native/quantlib-overlay/math-adl.patch`. The solver
source and checkout stay unchanged. No nonstandard math overloads are inserted
into `namespace std`. This is a small engineering compatibility issue, not
evidence that Approach A is fundamentally infeasible.

### Phase 1 differentiation

Full mode registers a PyTorch `CompositeImplicitAutograd` operation. The
upstream Thomas elimination arithmetic builds the reverse-mode graph. Tests
also verify second derivatives on small systems.

For adjoint mode, a Python `autograd.Function` invokes the same compiled solver
without recording its internal graph. For `A x = b` and upstream vector `g`,
the backward pass solves `A^T lambda = g` by swapping the off-diagonals. It returns:

```text
grad_b       = lambda
grad_diag[i] = -lambda[i]   * x[i]
grad_low[i]  = -lambda[i+1] * x[i]
grad_high[i] = -lambda[i]   * x[i+1]
```

The saved solution and diagonals suffice for this first-order VJP. The function
is explicitly once-differentiable; it does not silently return incorrect second
derivatives. Matrix assembly, boundary contributions, and cross-time-step
dependencies stay in PyTorch's graph. This is a custom adjoint **per solve**, not
yet a checkpointed adjoint for the entire PDE trajectory. Memory still grows
with the number of steps and nodes.

### Phase 1 PDE scope and validation

The example uses `x = log(S/K)` and the risk-neutral generator
`L = .5 sigma^2 dxx + (r-q-.5 sigma^2) dx - r`, centered spatial differences, and
`(I - dt L_i) u_i = u_(i+1)` with known boundary contributions on the RHS.
Rows of sigma are chronological and rollback visits them in reverse. The lower
boundary is zero; the upper boundary is the positive part of
`Smax exp(-q tau) - K exp(-r tau)`. Terminal values are the call payoff and
the price uses linear interpolation. Financial inputs, mesh, and boundaries
are fixed, while all local-volatility samples are differentiable.

The initial native test suite verifies:

- Forward residuals and arbitrary vector VJPs against `torch.linalg.solve`,
  including nonsymmetric matrices, both off-diagonals, RHS, and size two.
- Finite-difference `gradcheck` for both modes and `gradgradcheck` for full mode.
- Input validation, noncontiguous views, no input mutation, pivot failures,
  and explicit rejection of CUDA tensors and adjoint double backward.
- Analytic constant-volatility European prices and vega, full/adjoint equality
  for a nonuniform volatility field, and central directional differences of
  coefficient sensitivities and a neural-network pricing-loss gradient.

Local verification on 2026-10-31: **41 tests passed** through the CTest entry
point using the editable environment. Both neural examples produced price
11.35577190, loss 5.54966125, and network-gradient norm 20.85999953. The
unpatched compile probe failed as expected; the patched native build succeeded.

For S=K=100, r=.03, q=.01, T=1, sigma=.2, x in [-1.2,1.2]:

| Calculation | Call price |
| --- | ---: |
| Analytic Black–Scholes | 8.8273212254 |
| 20 time steps, 39 interior nodes | 8.6851953561 |
| 120 time steps, 159 interior nodes | 8.8130832506 |

The refined parallel volatility derivative is 38.6966376579; analytic vega is
approximately 38.7152. The tests allow price error .04 and vega relative error
.3%, and require refinement to reduce price error by at least fivefold. These
are discretization tests, not claims of exact PDE prices.

Review reproduced a negative call price on a coarse drift-dominated centered
stencil (sigma=.001, r=.01, q=.1). The API now rejects negative off-diagonal
generator weights and steps with `1 + dt*r <= 0`, requiring mesh refinement.
This preserves the prototype's monotone, strictly diagonally dominant implicit
systems. Boundary truncation and domain adequacy remain caller responsibilities.

The bounded sigmoid neural volatility example fixes the stock drift to r-q.
For a continuous-time model, uniformly bounded volatility gives the usual
finite-horizon exponential-integrability condition for the discounted stock
stochastic exponential; smooth bounded network coefficients also provide a
reasonable well-posedness starting point. Arbitrary sampled arrays alone do not
establish a continuous-time model or prove martingality.

### Phase 1 CPU measurements

`scripts/benchmark.py` records warmup-excluded forward/backward medians, prices,
parallel volatility derivatives, and autograd saved-tensor statistics. The
checked-in JSON records a 20-step, 31-node grid, one PyTorch thread, seven
repetitions, and CPU float64. Timings are machine/load dependent. Saved bytes
count repeated saves, **not** unique storage, graph-node overhead, or peak RSS.
The recorded adjoint forward samples range from roughly 14 to 84 ms within one
run, so these initial timings do not support a reliable speed ranking. Raw
samples are retained instead of interpreting that variability as an algorithmic
effect; controlled timing remains a follow-up experiment.

Full mode saves 6,821 tensors (88,000 counted bytes); adjoint mode saves 141
(29,760 counted bytes). Both return the same price and parallel-volatility VJP.
The scalar adapter still allocates and dispatches many tiny tensor operations.
These measurements establish neither scaling behavior nor superiority over
stock QuantLib. There is no GPU performance result. CUDA is available on the
development host, but the prototype deliberately rejects GPU tensors.

### Phase 1 build and repository decisions

- CMake compiles only the selected upstream kernel with C++17 and links the
  installed PyTorch's `torch`, `torch_cpu`, and `c10` libraries. Avoiding CUDA
  CMake discovery lets a CPU-only extension use a CUDA wheel without a toolkit.
- Python source/editable installation is supported. Native wheel packaging is
  deferred. Python/PyTorch/compiler ABI must match at build and runtime.
- Dependency checkout, native binaries, and temporary experiment logs are
  ignored. CI builds from the pinned checkout and runs the same tests.
- Local Windows builds/tests are verified; the included Linux CI job is not a
  claim of an already observed successful remote run.
- QuantLib's license is retained in its checkout and overlay headers.
  QuantLibAAD/XAD were inspected for integration design, not copied. Their
  different licensing needs separate review before future code reuse.

### Next experiments proposed at the end of Phase 1

These historical next steps motivated Phase 2 above; the current plan appears
under "Recommended next milestones".

1. Extend the scalar compatibility probe to `TripleBandLinearOp`, its mesher,
   and the modern finite-difference path. Inventory additional std/Boost math,
   conversions, and `Null<Real>` sentinel behavior with compile/runtime tests.
   The upstream generic Null implementation returns default zero for this
   wrapper; do not expose broader engines until that is explicitly addressed.
2. Integrate an existing QuantLib pricing engine and financial objects, then
   test curves, discrete cash dividends, and exercise projections. This is
   required before claiming preservation of those capabilities.
3. Compare both strategies at increasing grid sizes against a stock-double
   QuantLib baseline; measure peak memory, time, and numerical error. Consider
   checkpointing and full-trajectory adjoints only with evidence.
4. Investigate batched tensor kernels/GPU execution at measured bottlenecks.
   Scalar substitution alone does not vectorize QuantLib loops. A kernel-level
   hybrid may eventually be useful, but committing to a major pivot requires
   discussion with the user.

No current result justifies abandoning Approach A. Whole-library compatibility
and useful GPU performance remain open, distinct questions.

## Primary source references

- [Finite-difference vanilla engine](https://github.com/lballabio/QuantLib/blob/v1.40/ql/pricingengines/vanilla/fdblackscholesvanillaengine.cpp)
- [Black-Scholes operator and local-volatility sampling](https://github.com/lballabio/QuantLib/blob/v1.40/ql/methods/finitedifferences/operators/fdmblackscholesop.cpp)
- [Modern band operator](https://github.com/lballabio/QuantLib/blob/v1.40/ql/methods/finitedifferences/operators/triplebandlinearop.cpp)
- [Vanilla exercise/dividend conditions](https://github.com/lballabio/QuantLib/blob/v1.40/ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.cpp)
- [C++ autograd custom functions](https://docs.pytorch.org/tutorials/advanced/cpp_autograd.html)
- [QuantLib include-first/real hook](https://github.com/lballabio/QuantLib/blob/v1.40/ql/qldefines.hpp)
- [QuantLib type aliases](https://github.com/lballabio/QuantLib/blob/v1.40/ql/types.hpp)
- [Compiled tridiagonal implementation](https://github.com/lballabio/QuantLib/blob/v1.40/ql/methods/finitedifferences/tridiagonaloperator.cpp)
- [Array math](https://github.com/lballabio/QuantLib/blob/v1.40/ql/math/array.hpp)
- [Scalar comparisons](https://github.com/lballabio/QuantLib/blob/v1.40/ql/math/comparison.hpp)
- [Null sentinel implementation](https://github.com/lballabio/QuantLib/blob/v1.40/ql/utilities/null.hpp)
- [QuantLibAAD adapter (AGPL)](https://github.com/auto-differentiation/QuantLibAAD/blob/647b4011d1fb12c5eb217b7736541640c3fa3f0a/ql/qlaad.hpp)
- [QuantLibAAD CMake interface](https://github.com/auto-differentiation/QuantLibAAD/blob/647b4011d1fb12c5eb217b7736541640c3fa3f0a/ql/CMakeLists.txt)
