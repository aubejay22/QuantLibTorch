"""Packaging gates prevent incomplete or ABI-mismatched binary distributions."""
from pathlib import Path
import hashlib
import json
import runpy
import shutil
import sys
import sysconfig

import pytest
import torch
from setuptools.errors import SetupError


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = runpy.run_path(str(ROOT / "setup.py"))


@pytest.fixture
def native_fixture(tmp_path):
    (tmp_path / "scripts").mkdir()
    for filename in ("CMakeLists.txt", "scripts/prepare_quantlib.py", "scripts/prepare_packed.py",
                     "scripts/native_metadata.py", "scripts/bootstrap.py", "scripts/build.py", "_build_support.py"):
        (tmp_path / filename).write_text("fixed build input", encoding="utf-8")
    native = tmp_path / "python" / "quantlibtorch" / "_native"
    native.mkdir(parents=True)
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    library = native / ("quantlibtorch_native" + suffix)
    library.write_bytes(b"a test binary; never loaded")
    metadata = {
        "torch_version": str(torch.__version__),
        "python_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "quantlib_commit": "06edddbdb85f006c43a449b5fc846a00c67477f3",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "native_source_sha256": PACKAGING["native_source_digest"](tmp_path),
    }
    stamp = native / "build_info.json"
    stamp.write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path, library, stamp, metadata


def test_binary_distribution_is_never_tagged_pure_python():
    assert PACKAGING["NativeDistribution"]().has_ext_modules()


def test_missing_native_is_actionable(tmp_path):
    with pytest.raises(SetupError, match="scripts/build.py"):
        PACKAGING["validate_native"](tmp_path)


def test_matching_native_is_packagable(native_fixture):
    PACKAGING["validate_native"](native_fixture[0])


def test_native_changed_since_build_is_rejected(native_fixture):
    root, library, _, _ = native_fixture
    library.write_bytes(b"replaced after provenance was generated")
    with pytest.raises(SetupError, match="library_sha256"):
        PACKAGING["validate_native"](root)


@pytest.mark.parametrize("changed_source", ["CMakeLists.txt", "scripts/bootstrap.py", "scripts/build.py"])
def test_native_source_changed_since_build_is_rejected(native_fixture, changed_source):
    root, _, _, _ = native_fixture
    (root / changed_source).write_text("new native sources", encoding="utf-8")
    with pytest.raises(SetupError, match="native_source_sha256"):
        PACKAGING["validate_native"](root)


@pytest.mark.parametrize("key", ["torch_version", "python_tag", "platform", "cxx11_abi", "quantlib_commit"])
def test_incompatible_build_is_rejected(native_fixture, key):
    root, _, stamp, metadata = native_fixture
    metadata[key] = "incompatible"
    stamp.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SetupError, match=key):
        PACKAGING["validate_native"](root)


def test_loader_rejects_mismatched_torch_before_loading(native_fixture, monkeypatch):
    root, library, stamp, metadata = native_fixture
    metadata["torch_version"] = "2.0.0+wrong"
    stamp.write_text(json.dumps(metadata), encoding="utf-8")
    script = library.parent.parent / "solver.py"
    shutil.copyfile(ROOT / "python" / "quantlibtorch" / "solver.py", script)
    attempted = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: attempted.append(path))
    with pytest.raises(ImportError, match="torch_version"):
        runpy.run_path(str(script))
    assert attempted == []


@pytest.mark.parametrize("damage", ["missing_metadata", "invalid_metadata", "changed_binary"])
def test_loader_rejects_incomplete_or_modified_wheel_before_loading(native_fixture, monkeypatch, damage):
    _, library, stamp, _ = native_fixture
    if damage == "missing_metadata":
        stamp.unlink()
    elif damage == "invalid_metadata":
        stamp.write_text("[]", encoding="utf-8")
    else:
        library.write_bytes(b"different binary")
    script = library.parent.parent / "solver.py"
    shutil.copyfile(ROOT / "python/quantlibtorch/solver.py", script)
    attempted = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: attempted.append(path))
    with pytest.raises(ImportError):
        runpy.run_path(str(script))
    assert attempted == []


def test_loader_rejects_relabelled_native_binary(native_fixture, monkeypatch):
    _, library, _, metadata = native_fixture
    script = library.parent.parent / "solver.py"
    shutil.copyfile(ROOT / "python/quantlibtorch/solver.py", script)
    compiled = {key: value for key, value in metadata.items() if key != "library_sha256"}
    compiled["torch_version"] = "2.9.0+cpu"
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)
    monkeypatch.setattr(torch.ops.quantlibtorch, "build_info", lambda: json.dumps(compiled), raising=False)
    with pytest.raises(ImportError, match="binary and build metadata disagree"):
        runpy.run_path(str(script))


def test_sdist_is_explicitly_disabled():
    command = PACKAGING["WheelOnlySdist"](PACKAGING["NativeDistribution"]())
    with pytest.raises(SetupError, match="wheel-first"):
        command.run()
