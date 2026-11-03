# Engineering workflow

Read PROJECT_CONTEXT.md and docs/architecture.md before changing direction.
Approach A (QuantLib QL_REAL + LibTorch) is the current priority. Discuss major
architecture pivots with the user; reversible experiments do not require a pivot.

- Preserve upstream algorithm provenance. Pin dependencies; keep compatibility
  patches small, reproducible, and distinct from numerical changes.
- Never link a stock double-based QuantLib library into the TorchReal build.
- Do not silently detach coefficient tensors or use in-place tensor updates in
  the scalar adapter. Test VJPs against independent references and differences.
- Run `python -m pytest` after native builds; see README.md for setup.
- Record measured findings, limitations, and next steps in docs/architecture.md.
- CPU correctness precedes optimization. GPU benefit requires measurements.
- Subagents may handle independent research, implementation, or review. Assign
  disjoint file ownership and review their changes before integrating.
- Do not copy QuantLibAAD/XAD implementation without reviewing its license.
