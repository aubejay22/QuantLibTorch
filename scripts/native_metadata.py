"""Generate immutable native provenance at CMake configure time."""
from pathlib import Path
import json
import runpy
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def build_metadata():
    import torch
    from bootstrap import QUANTLIB_COMMIT
    digest = runpy.run_path(str(ROOT / "_build_support.py"))["native_source_digest"]
    return {
        "torch_version": str(torch.__version__),
        "python_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "quantlib_commit": QUANTLIB_COMMIT,
        "native_source_sha256": digest(ROOT),
    }


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    content = '#pragma once\n#define QLT_BUILD_INFO R"qlt(' + json.dumps(build_metadata(), sort_keys=True) + ')qlt"\n'
    header = destination / "quantlibtorch_build_info.hpp"
    if not header.exists() or header.read_text(encoding="utf-8") != content:
        header.write_text(content, encoding="utf-8")
