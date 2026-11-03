"""Release promotion rejects mismatched provenance before gaining upload rights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "release_artifacts", Path(__file__).resolve().parents[1] / "scripts/release_artifacts.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def make_wheel(tmp_path, *, missing_notice=None, binary=b"synthetic native payload", native_override=None,
               python_requirement="<3.14,>=3.13"):
    wheel = tmp_path / "quantlibtorch-0.3.0-cp313-cp313-win_amd64.whl"
    native = dict(torch_version="2.10.0+cu126", python_tag="cpython-313", platform="win-amd64",
                  cxx11_abi=True, quantlib_commit="06edddbdb85f006c43a449b5fc846a00c67477f3",
                  native_source_sha256="a" * 64, library_sha256=hashlib.sha256(binary).hexdigest())
    native.update(native_override or {})
    info = "quantlibtorch-0.3.0.dist-info/"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(info + "METADATA", "Metadata-Version: 2.1\nName: quantlibtorch\nVersion: 0.3.0\n"
                         f"Requires-Python: {python_requirement}\nLicense: BSD 3-Clause License\n"
                         "Requires-Dist: torch==2.10.0+cu126\n\n")
        for name in ("LICENSE", "QuantLib.txt", "Boost.txt", "PyTorch.txt", "JSON.txt", "Abseil.txt", "NOTICE.txt"):
            if name != missing_notice:
                archive.writestr(info + name, "Test fixture notice\n")
        archive.writestr("quantlibtorch/_native/build_info.json", json.dumps(native))
        archive.writestr("quantlibtorch/_native/quantlibtorch_native.dll", binary)
    return wheel


def test_manifest_covers_wheel_native_abi_python_and_notices(tmp_path):
    wheel = make_wheel(tmp_path)
    result = release.inspect_wheel(wheel)
    assert result["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert result["requires_python"] == "<3.14,>=3.13"
    assert len(result["notices_sha256"]) == 7
    assert result["native"]["platform"] == "win-amd64"


@pytest.mark.parametrize("notice", ["LICENSE", "QuantLib.txt", "Boost.txt", "PyTorch.txt", "JSON.txt", "Abseil.txt", "NOTICE.txt"])
def test_missing_notice_blocks_promotion(tmp_path, notice):
    with pytest.raises(SystemExit, match="Required license/notice"):
        release.inspect_wheel(make_wheel(tmp_path, missing_notice=notice))


@pytest.mark.parametrize("path", [b"\\".join([b"C:", b"Users", b"private-name", b"source.cpp"]),
                                 b"/".join([b"", b"home", b"private-name", b"source.cpp"]),
                                 "\\".join(["C:", "Users", "private-name", "source.cpp"]).encode("utf-16-le")])
def test_private_native_path_blocks_promotion(tmp_path, path):
    with pytest.raises(SystemExit, match="private home-directory"):
        release.inspect_wheel(make_wheel(tmp_path, binary=path))


@pytest.mark.parametrize("override", [{"torch_version": "2.10.0+cpu"}, {"python_tag": "cpython-314"},
                                     {"cxx11_abi": False}, {"quantlib_commit": "b" * 40},
                                     {"library_sha256": "c" * 64}])
def test_unvalidated_native_configuration_blocks_promotion(tmp_path, override):
    with pytest.raises(SystemExit):
        release.inspect_wheel(make_wheel(tmp_path, native_override=override))


def test_broader_python_metadata_blocks_promotion(tmp_path):
    with pytest.raises(SystemExit, match="constrain Python"):
        release.inspect_wheel(make_wheel(tmp_path, python_requirement=">=3.13"))


@pytest.mark.parametrize("change", [{"head_sha": "b" * 40}, {"event": "pull_request"},
                                   {"conclusion": "failure"}, {"head_branch": "untrusted"},
                                   {"head_repository": {"full_name": "someone/fork"}},
                                   {"path": ".github/workflows/untrusted.yml"}])
def test_foreign_failed_or_unreviewed_run_blocks_promotion(monkeypatch, change):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/QuantLibTorch")
    response = dict(repository={"full_name": "owner/QuantLibTorch"},
                    head_repository={"full_name": "owner/QuantLibTorch"},
                    path=".github/workflows/release-candidate.yml", event="workflow_dispatch",
                    status="completed", conclusion="success", head_sha="a" * 40, head_branch="master")
    response.update(change)
    monkeypatch.setattr(release, "read_json", lambda url, **kw: (
        response if "/actions/runs/" in url else {"default_branch": "master"}))
    with pytest.raises(SystemExit):
        release.check_run("123", "release-candidate.yml", "a" * 40)


def test_index_mismatch_blocks_download_before_fetching_bytes(tmp_path, monkeypatch):
    manifest = dict(version="0.3.0", filename="quantlibtorch-0.3.0-cp313-cp313-win_amd64.whl", sha256="a" * 64)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(release, "read_json", lambda url: {"urls": [dict(
        filename=manifest["filename"], digests={"sha256": "b" * 64}, yanked=False)]})
    with pytest.raises(SystemExit, match="Index digest differs"):
        release.download(argparse.Namespace(directory=tmp_path, index="testpypi", output=tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["quantlibtorch/.env", "quantlibtorch/_native/torch_cpu.dll", "../payload.py"])
def test_unexpected_wheel_members_block_promotion(tmp_path, name):
    wheel = make_wheel(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(name, b"unexpected")
    with pytest.raises(SystemExit, match="wheel member"):
        release.inspect_wheel(wheel)
