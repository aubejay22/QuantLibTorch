"""Privacy export must preserve source and measured values while removing paths."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

WINDOWS_HOME = "C:" + chr(92) + "Users" + chr(92) + "Example"
POSIX_HOME = "/" + "home" + "/example"
MAC_HOME = "/" + "Users" + "/example"

@pytest.fixture
def exporter(tmp_path, monkeypatch):
    location = Path(__file__).resolve().parents[1] / "scripts/prepare_public_export.py"
    spec = importlib.util.spec_from_file_location("public_export_test", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module, tmp_path, location


def prepare(module, root, files, monkeypatch, dirty=False):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def git(*args):
        if args == ("status", "--porcelain"):
            return b" M changed" if dirty else b""
        if args == ("rev-parse", "HEAD"):
            return b"a" * 40
        if args == ("ls-files", "-z"):
            return b"\0".join(name.encode() for name in files)
        if args == ("log", "--all", "--format=%ae%n%ce"):
            return b"private@example.invalid\nprivate@example.test\n"
        raise AssertionError(args)

    monkeypatch.setattr(module, "git", git)


def test_paths_json_gzip_and_source_integrity(exporter, monkeypatch):
    module, root, location = exporter
    report = {"path": WINDOWS_HOME + r"\work\trace.json", "price": 10.25,
              "gradient": [0.5, 2.75], "contact": "private@example.test"}
    encoded = json.dumps(report).encode()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("cpp/source.cpp", b"double price = 10.25;\n")
    files = {
        "docs/report.json": encoded,
        "docs/trace.json.gz": gzip.compress(encoded),
        "docs/report.txt": (WINDOWS_HOME + "\\work\n" + POSIX_HOME + "/work\n" + MAC_HOME + "/work\n").encode(),
        "docs/source.zip": archive_buffer.getvalue(),
        "scripts/prepare_public_export.py": location.read_bytes(),
        "cpp/source.cpp": b"double price = 10.25;\n",
        "LICENSES/upstream.txt": b"Copyright upstream <author@upstream.test>\n",
    }
    prepare(module, root, files, monkeypatch)
    output = root / "build/public"
    result = module.export(output)
    clean = json.loads((output / "docs/report.json").read_bytes())
    assert clean == {**report, "path": r"<USER_HOME>\work\trace.json", "contact": "<PRIVATE_EMAIL>"}
    assert json.loads(gzip.decompress((output / "docs/trace.json.gz").read_bytes())) == clean
    assert (output / "docs/report.txt").read_bytes() == b"<USER_HOME>\\work\n<USER_HOME>/work\n<USER_HOME>/work\n"
    for name in ("scripts/prepare_public_export.py", "cpp/source.cpp", "docs/source.zip", "LICENSES/upstream.txt"):
        assert (output / name).read_bytes() == files[name]
    assert all(item["path"].startswith("docs/") for item in result["transformations"])
    assert result["personal_path_or_author_email_findings_after_transform"] == 0
    assert not (output / ".git").exists()


@pytest.mark.parametrize("name", ["scripts/example.py", "cpp/source.cpp", "LICENSES/upstream.txt"])
def test_personal_source_requires_review(exporter, monkeypatch, name):
    module, root, _ = exporter
    prepare(module, root, {name: (POSIX_HOME + "/private").encode()}, monkeypatch)
    with pytest.raises(RuntimeError, match="manual review"):
        module.export(root / "build/public")


def test_dirty_revision_and_existing_destination_rejected(exporter, monkeypatch):
    module, root, _ = exporter
    prepare(module, root, {}, monkeypatch, dirty=True)
    with pytest.raises(RuntimeError, match="Commit the reviewed"):
        module.export(root / "build/public")
    output = root / "build/public"
    output.mkdir(parents=True)
    (output / "keep").write_text("preserve")
    with pytest.raises(RuntimeError, match="nonempty"):
        module.export(output)
    assert (output / "keep").read_text() == "preserve"
