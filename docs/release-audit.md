# Public release security and provenance audit

Audit date: 2026-11-02. Baseline: local release-candidate checkpoint
`1002f78`. This review covers source publication and license eligibility;
native and registry validation are additional gates in [release.md](release.md).

**Inventory correction after actual remote CI:** the first Linux build found
an implicit nlohmann/json 3.11.2 header dependency supplied locally by Conda and
absent from the official torch wheel. The initial inventory and prior wheel
license approval are superseded by this finding. The previously audited wheel
`3962507cedc6f852588163ef0fd9d7183d68fd3718007b10d90735bb36eab7de`
must not be published: it omits the JSON MIT notice. Its native and numerical
measurements remain historical evidence, not authorization to distribute it.
The dependency is being made explicit and rebuilt with the complete notices;
the replacement wheel requires a fresh artifact and installed-package audit.

## Scope and measured findings

The baseline has 21 reachable commits, 153 trees and 992 distinct file blobs.
All reachable blobs were scanned, including the expanded contents of 40 source
ZIPs: 4,329 payloads before separately expanding two gzip profiler traces
(13,237,668 and 59,044,947 bytes). The working-tree inventory contained 843
tracked files. The retained binary-format evidence consists of source ZIPs,
compressed traces and a generated PNG/PDF plot, not installed libraries.
No PE or ELF executable was found in reachable source blobs or expanded ZIPs.
No private-key, GitHub/PyPI/Slack token, AWS access-key, credential-bearing URL
or quoted password/API-token assignment matched the audit patterns.

Personal home-directory paths occurred in 279 report payloads and both
compressed traces. Historical commits also contain a personal author email.
These details prevent publishing the original history unchanged under this
release's privacy requirements. No credential values are reproduced here.

The original repository, commits, reports and archives are preserved locally.
`scripts/prepare_public_export.py` produces a separate source snapshot from a
clean, committed revision and does not copy `.git` or private ancestry. It
replaces personal home prefixes with `<USER_HOME>` and the local commit authors'
private emails with `<PRIVATE_EMAIL>` in textual evidence. The two modified
gzip trace containers are rebuilt deterministically. Numerical source, source
ZIP archives and third-party notices are required to remain byte-identical;
the exporter aborts if they require personal-data edits.

The exporter scans expanded payloads and rejects detected credentials,
executable blobs, credential-like filenames, symlinks and unsafe paths. It
refuses a dirty source revision, a nonempty destination or a destination
outside the checkout's ignored build directory. The public repository must
start with a new commit using the verified owner's GitHub noreply identity.
This creates a public snapshot without rewriting or deleting original history.

The export's `docs/public-source-manifest.json` records the precise local source
revision, payload counts and original/public SHA256 values for transformed
evidence. A copy remains in the local ignored build directory. Historical
reports' embedded hashes continue to identify original local evidence; the
manifest explains their relation to sanitized public copies. Prices,
gradients, timings and test outcomes are not changed by sanitization.

## License and algorithm provenance

The owner explicitly authorized BSD-3-Clause for eligible original work. The
review found no conflicting ownership notice or copied implementation with
incompatible terms. [LICENSE](../LICENSE) grants BSD-3-Clause for original
QuantLibTorch work; [third-party notices](../LICENSES/NOTICE.txt) state the
exceptions. This conclusion relies on the owner's representation of ownership
and the repository's provenance, not independent employment-contract records.

QuantLib is pinned at `06edddbdb85f006c43a449b5fc846a00c67477f3`.
`LICENSES/QuantLib.txt` is byte-identical to that checkout's `LICENSE.TXT`, with
SHA256 `33a785cdb16936a5cc598c5c2115661cc3289f88eac8b4116ba6e18d1deed833`.
The checkout is preserved and generated compatibility overlays retain upstream
headers. The CMake closure compiles selected sources with `QL_REAL=TorchReal`;
the target links torch, torch_cpu and c10 from the active Python environment,
and does not link a stock QuantLib library.

Derived spline, operator, Thomas and step arithmetic identifies its QuantLib
source and retains copyright notices. The packed-stage generator retains
upstream headers with the copied stage bodies. Historical archives retain
those upstream obligations. No QuantLibAAD/XAD implementation or dependency was
found; historical design comparisons are not redistribution.

Boost's license and the matching PyTorch distribution's notices are retained.
The corrected inventory also includes nlohmann/json 3.11.2, whose single header
is included by PyTorch functorch headers. `LICENSES/JSON.txt` preserves the
exact upstream `v3.11.2/LICENSE.MIT` bytes (1,076 bytes), SHA256
`86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14`.
The header also identifies MIT-licensed portions by Bjorn Hoehrmann, Florian
Loitsch and Evan Nemerson; their attribution is retained in `NOTICE.txt`.
An explicit inner comment identifies a pre-C++14 Abseil integer-sequence
fallback as Apache-2.0 despite the enclosing MIT SPDX line. The C++17 build
excludes that fallback; its attribution and original license from upstream
commit `10cb35e459f5ecca5b2ff107635da0bfa41011b4` are nevertheless retained in
`NOTICE.txt` and `Abseil.txt`. These permissive terms are compatible with
distribution of original BSD-3-Clause work while retaining upstream terms.
The QuantLibTorch wheel must include the project LICENSE and all notice files,
and must not bundle torch DLLs, CUDA/NVIDIA libraries, build logs, source
snapshots or local environments. [licensing.md](licensing.md) gives the inventory.

## Artifact and final publication gates

Native `__FILE__` diagnostic strings initially exposed developer home paths.
The build uses compiler source-prefix mapping to address this; each final DLL
and wheel must be scanned after rebuilding. Merely sanitizing source reports
does not sanitize a previously compiled binary. The final wheel's archive
members, metadata, dependencies and license files must be inspected separately.

The mapped Windows rebuild's 3,150,848-byte native library has SHA256
`bdfc33fa7212ba5408ee8ba9b6c562e6239707b8db9f345b245783280ad4eef6`.
An independent rescan found zero Windows/POSIX personal home-path matches in
its raw bytes and in its NUL-stripped representation (covering wide strings).
This closes the identified native path leak for that exact binary; it does
not substitute for the final wheel and installed-package checks.

After the release preparation commit, generate and independently inspect the
public export and its new Git commit before pushing. Confirm there is no old
ancestry, personal email or home path. Recheck final artifacts and new files;
the baseline review does not automatically certify later additions.

This is a pattern-based scan plus provenance and content review, not a proof
that arbitrary undisclosed secrets cannot exist. It does not replace native
compatibility, installation, Trusted Publisher or registry publication gates.
