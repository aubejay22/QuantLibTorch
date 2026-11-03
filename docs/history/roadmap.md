> Historical snapshot through Phase 26, archived during release preparation.
> Statements of current support and next work describe their original checkpoints.
> Use [current architecture](../architecture.md), [capabilities](../capabilities.md)
> and [release support](../release.md) for current contracts. Command paths are relative
> to the repository root. Markdown links have been relocated; evidence is unchanged.

# General-purpose library engineering roadmap

## Extended integration session (Phase 26)

Starting checkpoint: `092abf1`, clean Phase 25 worktree. Execute and validate
these milestones in order, retaining all existing public APIs and backends:

1. Original `FdHestonVanillaEngine`, process/model and automatic mesh preparation.
   Freeze explicitly prepared geometry during pricing; expose five Heston
   parameters, spot and yield curves through full AD and supported equation
   adjoints. Validate European, American and Bermudan prices with isolated stock
   QuantLib and fixed-mesh parameter bumps. Include a reusable model/engine API
   and an explanatory notebook.
2. Reusable graph-connected rates and dated discount curves for fixed-mesh
   latent, contract-batch and asset-batch pricing. Validate graph freshness,
   curve interpolation, CPU/CUDA parity and checkpoint VJPs.
3. Integrate a further compatible original pricing engine after a dependency
   audit; the original Black-Scholes shout engine is the initial candidate.
   Gate claims on usable financial inputs and independent prices/gradients.
4. Broaden original numerical/financial controls where validated, including
   damping and explicit mesh configuration. Preserve original boundary behavior;
   reject unsupported combinations explicitly.
5. Run integrated regression, review upstream overlay provenance and update
   capability/install examples. Create clean Git checkpoints for substantial
   validated milestones and record exact derivative/device limitations.

The session continues beyond its first successful milestone. Existing CUDA,
batching, neural models and higher-order full AD are regression requirements.

The Phase 25 plan below started from `6121346` (Phase 24) and is retained as
historical acceptance evidence. The practical library priority and Approach A
remain in force; Phase 26 starts from the later `092abf1` checkpoint above.

## Phase 25 implementation and acceptance gates

1. Reusable instrument/exercise/market/engine objects and a PyTorch pricing
   module. Preserve functional APIs. Gate: configuration validation, equality
   with functional engines, model parameter registration and repeated training.
2. Original analytic European engines with graph-connected financial inputs.
   Gate: separate stock-QuantLib prices/Greeks, first and higher reverse
   derivative checks, dated curves, cash amounts and explicit unsupported errors.
3. Broader original FD engine configuration: three additional ADI schemes and
   escrowed dividends. Gate: independent stock prices and volatility bumps
   across European, American and Bermudan exercise in both differentiation modes.
4. Exercise and spot-dividend support in contract/asset batches on CPU/CUDA.
   Gate: original scalar and independent stock comparisons, all four schemes,
   coincident/maturity events, state/neural sensitivities, checkpoint replay,
   heterogeneous schedules, and a measured event workload.
5. Practical packaging and focused user documentation. Gate: source archive
   contains native build inputs, binary wheel contains ABI-matched library,
   isolated installed import/pricing, and explicit failure for missing/stale
   native artifacts. No portable release claim without platform evidence.
6. Integration checkpoint. Gate: full `python -m pytest`, clean pinned QuantLib
   checkout, reviewed changes, exact capability matrix and Git checkpoints.

## Subsequent engineering priorities

1. Extend fixed-mesh rate/curve inputs to the existing `fd_vanilla` and
   `fd_basket` APIs. Phase 26 implements these inputs for latent/packed engines;
   continue keeping active values outside passive geometry caches.
2. Heston adjoint damping, leverage/quanto adapters, and compatible Heston
   batching with exact original boundary/event behavior. Original Heston and
   shout scalar engines are now integrated; barrier/rebate engines are the next
   concrete candidates, while CEV needs active special-function boundaries.
3. Broader boundary/solver controls and packed damping, followed by richer
   calendar/day-counter selection. Phase 26 adds explicit original/custom Heston
   and shout meshes, two day counters and three original calendar helpers.
4. Sparse dividend maps, broader dividend layout support and chunking for
   larger portfolios, driven by practical workload measurements.
5. Windows/Linux CI-built release artifacts, compiler/runtime compatibility
   validation and Linux wheel portability checks before public distribution.

Extended calibration-identifiability or convergence research is not a release
gate. Numerical checks remain scoped to the implemented financial behavior.
