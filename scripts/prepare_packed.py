"""Instantiate pinned QuantLib ADI stage bodies for whole tensor states.

This is a type adapter, separate from the scalar compatibility overlay. Stage
arithmetic and boundary hook order are copied verbatim from the pinned sources.
The unchanged QuantLib FiniteDifferenceModel supplies rollback and stopping times.
"""
from pathlib import Path
import hashlib
import json
import re
import sys


def prepare(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    chunks, manifest = [], {}
    for name, method in [("Douglas", "douglas"), ("Hundsdorfer", "hundsdorfer"),
                         ("CraigSneyd", "craig_sneyd"), ("ModifiedCraigSneyd", "modified_craig_sneyd")]:
        relative = f"ql/methods/finitedifferences/schemes/{name.lower()}scheme.cpp"
        content = (source/relative).read_text(encoding="utf-8")
        signature = f"void {name}Scheme::step(array_type& a, Time t)"
        start = content.index(signature)
        begin = content.index("{", start)
        depth, end = 1, begin+1
        while depth:
            depth += (content[end] == "{")-(content[end] == "}")
            end += 1
        body = content[begin:end]
        # Array is the only body substitution. All scalars here are passive
        # numerical configuration; model tensors never enter scalar wrappers.
        adapted = re.sub(r"\bArray\b", "at::Tensor", body)
        chunks.append(content[:content.index("#include")] +
            f"// Generated from {relative}; see LICENSES/QuantLib.txt.\n"
            f"void PackedEvolver::step_{method}(at::Tensor& a, double t) " + adapted + "\n")
        manifest[relative] = dict(source_sha256=hashlib.sha256((source/relative).read_bytes()).hexdigest(),
                                 stage_sha256=hashlib.sha256(body.encode()).hexdigest())
    (output/"packed_stages.inc").write_text("\n".join(chunks), encoding="utf-8", newline="\n")
    # A second state/operator adapter uses identical bodies. Only the owning
    # class in the method signature changes; keep the original artifact stable.
    (output/"multi_stages.inc").write_text("\n".join(chunks).replace(
        "void PackedEvolver::", "void MultiEvolver::"), encoding="utf-8", newline="\n")
    (output/"packed_stages.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
    # Event arithmetic is evaluated by upstream objects at plan construction;
    # retain their exact provenance separately from unchanged stage generation.
    event_sources = [
        "ql/methods/finitedifferences/finitedifferencemodel.hpp",
        "ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.cpp",
        "ql/methods/finitedifferences/stepconditions/fdmbermudanstepcondition.cpp",
        "ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.cpp",
        "ql/methods/finitedifferences/utilities/fdmdividendhandler.cpp",
        "ql/methods/finitedifferences/utilities/fdminnervaluecalculator.cpp",
    ]
    (output/"packed_events.json").write_text(json.dumps({relative: {
        "source_sha256": hashlib.sha256((source/relative).read_bytes()).hexdigest()
    } for relative in event_sources}, indent=2)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
