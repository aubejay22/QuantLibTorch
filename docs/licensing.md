# Licensing and provenance

The owner authorized **BSD-3-Clause** for eligible original QuantLibTorch code
on 2026-11-02. [LICENSE](../LICENSE) contains the complete terms and the
attribution `Copyright (c) 2026, QuantLibTorch contributors`. The license follows
the [OSI BSD-3-Clause text](https://opensource.org/license/bsd-3-clause).
Third-party implementations retain their own copyright and license; the
project does not claim ownership of them.

## Dependency inventory

| Component | Use and license record |
| --- | --- |
| QuantLib 1.40, pinned commit `06edddbdb85f006c43a449b5fc846a00c67477f3` | Selected source compiled with `QL_REAL=TorchReal`; [full pinned notices](../LICENSES/QuantLib.txt) retained, byte-identical to upstream `LICENSE.TXT`. Generated overlays retain original headers. |
| Boost 1.82.0 (validated Windows build headers) | Header dependency; [Boost Software License 1.0](../LICENSES/Boost.txt) included. No separate Boost library linked. |
| nlohmann/json 3.11.2 | Explicitly pinned header dependency included through PyTorch functorch headers; [exact upstream MIT license](../LICENSES/JSON.txt) and its bundled MIT copyright notices in [NOTICE.txt](../LICENSES/NOTICE.txt) are retained. No separate header tree ships in the wheel. |
| Abseil integer-sequence fallback inside JSON | The pre-C++14 fallback explicitly cites Apache-2.0 source at commit `10cb35e459f5ecca5b2ff107635da0bfa41011b4`. C++17 excludes this fallback; its [upstream Apache license](../LICENSES/Abseil.txt) and attribution are nevertheless retained. |
| PyTorch 2.10.0 | Headers and dynamic linking to separately installed torch; [matching distribution license](../LICENSES/PyTorch.txt) included. The wheel does not vendor torch, CUDA or NVIDIA libraries. |
| Stock QuantLib Python 1.40 | Test-only independent financial reference, run separately from TorchReal native code. Not a runtime dependency. |
| QuantLibAAD / XAD | Historical design references only. No implementation copied or dependency added. |

QuantLib-derived portions of `cpp/operator_arithmetic.cpp`,
`cpp/thomas_arithmetic.cpp`, `cpp/step_arithmetic.cpp`,
`cpp/step_preparation.cpp`, `cpp/step_adjoint.cpp` and
`cpp/spline_kernels.cpp` retain their upstream attribution and license.
`scripts/prepare_quantlib.py` preserves source headers in compatibility overlays;
`scripts/prepare_packed.py` copies pinned stage bodies with their source headers.
The original checkout is not modified. The CMake source manifest identifies the
compiled upstream source closure; no stock double-based QuantLib is linked.

The same distinction applies to historical source archives: original work is
covered by the project license, and upstream portions retain their own terms.
Existing third-party notices are not replaced by the project's BSD license.
The wheel must contain `LICENSE` and all `LICENSES/*.txt` as distribution license
files. See [the release audit](release-audit.md) for inspection scope and limits.

The first remote Linux build exposed an implicit nlohmann/json dependency that
the local Conda include directory had supplied. The initial license inventory
was incomplete: the earlier wheel omitted its MIT notice and is not approved
for publication. The dependency is now explicitly pinned and the wheel must
be rebuilt with `JSON.txt` and the corrected notices. The MIT terms permit this
distribution alongside BSD-3-Clause, provided the upstream copyright and
permission notices are retained. The single header's embedded UTF-8 decoder,
Grisu2 conversion and Hedley identify MIT terms; their copyright notices are
preserved in `NOTICE.txt`. An explicit inner comment identifies the Abseil
pre-C++14 fallback as Apache-2.0, despite the enclosing file's MIT SPDX line.
The C++17 build excludes that fallback. Its original license is retained and
is not replaced by the JSON or project license. These permissive terms do not
prevent licensing the project's original work under BSD-3-Clause.

The owner's statement of ownership authorizes licensing original project work.
The repository review found no contrary ownership notices or incompatible
copied implementation. This is a provenance review of available source and
records, not an independent determination of employment or contractual rights.

## Distribution name

The owner selected `quantlibtorch`, subject to registry availability. A registry
HTTP 404 does not reserve a name or guarantee acceptance. The release process
checks the actual TestPyPI and PyPI endpoints before publishing.
