"""Native dependency downloads fail closed and verified caches work offline."""
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("release_bootstrap", Path(__file__).parents[1] / "scripts/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    content = b"independent pinned header fixture\n"
    monkeypatch.setattr(bootstrap, "JSON_SHA256", hashlib.sha256(content).hexdigest())
    header = tmp_path / "third_party" / f"nlohmann-json-{bootstrap.JSON_VERSION}" / "include/nlohmann/json.hpp"
    return tmp_path, header, content


def test_verified_cache_never_uses_network(dependency, monkeypatch):
    root, header, content = dependency
    header.parent.mkdir(parents=True)
    header.write_bytes(content)
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: pytest.fail("cache must be offline"))
    assert bootstrap.ensure_json_header(root) == header.parent.parent


def test_cmake_check_does_not_download_missing_header(dependency, monkeypatch):
    root, header, _ = dependency
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: pytest.fail("check must be offline"))
    with pytest.raises(SystemExit, match="bootstrap.py"):
        bootstrap.ensure_json_header(root, download=False)
    assert not header.exists()


def test_download_is_verified_before_installation(dependency, monkeypatch):
    root, header, content = dependency
    calls = []
    def download(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(content)
    monkeypatch.setattr(bootstrap, "urlopen", download)
    bootstrap.ensure_json_header(root)
    assert header.read_bytes() == content
    assert calls == [(bootstrap.JSON_URL, 60)]
    assert list(header.parent.iterdir()) == [header]


def test_bad_download_is_not_installed(dependency, monkeypatch):
    root, header, _ = dependency
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: io.BytesIO(b"wrong response"))
    with pytest.raises(SystemExit, match="checksum mismatch"):
        bootstrap.ensure_json_header(root)
    assert not header.exists()


def test_modified_cache_is_preserved_and_rejected(dependency, monkeypatch):
    root, header, _ = dependency
    header.parent.mkdir(parents=True)
    header.write_bytes(b"local changes")
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: pytest.fail("must not overwrite local changes"))
    with pytest.raises(SystemExit, match="checksum mismatch"):
        bootstrap.ensure_json_header(root)
    assert header.read_bytes() == b"local changes"
