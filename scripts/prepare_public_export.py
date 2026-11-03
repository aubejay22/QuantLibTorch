"""Export reviewed source without private history or personal evidence paths.

Run after committing reviewed preparation. The new destination has no .git;
original commits and evidence stay local. Native wheels need a separate audit.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HOME_PATH = re.compile(rb"(?i)(?:[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9][A-Za-z0-9_. -]*|/home/[A-Za-z0-9][A-Za-z0-9_. -]*|/Users/[A-Za-z0-9][A-Za-z0-9_. -]*)")
SECRET_PATTERNS = {
    "private-key": rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b",
    "aws-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "pypi-token": rb"\bpypi-[A-Za-z0-9_-]{30,}",
    "slack-token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
    "credential-url": rb"https?://[^\s/:@]{2,}:[^\s/@]{4,}@",
}
TEXT_SUFFIXES = {".py", ".cpp", ".hpp", ".h", ".inc", ".cmake", ".toml", ".yml", ".yaml", ".md", ".txt", ".json", ".ipynb", ".log", ".in", ".gitignore"}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export(destination: Path) -> dict:
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "build").resolve()):
        raise RuntimeError("Export destination must be inside this checkout's build directory")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty export; use a fresh destination")
    if git("status", "--porcelain").strip():
        raise RuntimeError("Commit the reviewed source before exporting a precisely identified revision")
    commit = git("rev-parse", "HEAD").decode().strip()
    names = [x.decode() for x in git("ls-files", "-z").split(b"\0") if x]
    own_emails = {x for x in git("log", "--all", "--format=%ae%n%ce").splitlines()
                  if x and b"noreply" not in x and not x.endswith(b".invalid")}
    transformations = []
    counts = collections.Counter()

    def transform(name: str, data: bytes) -> bytes:
        counts["payloads"] += 1
        before = data
        if data[:4] == b"PK\x03\x04":
            counts["zip_archives"] += 1
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = [(info, transform(name + "!" + info.filename, archive.read(info)))
                           for info in archive.infolist()]
                if any(content != archive.read(info) for info, content in entries):
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w") as rewritten:
                        for info, content in entries:
                            rewritten.writestr(info, content)
                    data = buffer.getvalue()
        elif data[:2] == b"\x1f\x8b":
            counts["gzip_archives"] += 1
            plain = gzip.decompress(data)
            clean = transform(name.removesuffix(".gz"), plain)
            if clean != plain:
                data = gzip.compress(clean, mtime=0)
        else:
            if data[:2] == b"MZ" or data[:4] == b"\x7fELF":
                raise RuntimeError(f"Unexpected executable in source snapshot: {name}")
            for kind, pattern in SECRET_PATTERNS.items():
                if re.search(pattern, data):
                    raise RuntimeError(f"Review required: {kind} pattern in {name}; value withheld")
            if HOME_PATH.search(data) or any(email in data for email in own_emails):
                if Path(name).suffix.lower() not in TEXT_SUFFIXES:
                    raise RuntimeError(f"Personal metadata in unsupported binary format: {name}")
                if not name.startswith("docs/") or "!" in name:
                    raise RuntimeError(f"Source or third-party notice requires manual review: {name}")
                data = HOME_PATH.sub(b"<USER_HOME>", data)
                for email in own_emails:
                    data = data.replace(email, b"<PRIVATE_EMAIL>")
                if Path(name).suffix.lower() in {".json", ".ipynb"}:
                    json.loads(data)
            if HOME_PATH.search(data) or any(email in data for email in own_emails):
                raise RuntimeError(f"Personal metadata remains after sanitizing {name}")
        if data != before:
            transformations.append({"path": name, "original_sha256": digest(before),
                                    "public_sha256": digest(data)})
        return data

    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            raise RuntimeError(f"Unsafe tracked path: {name}")
        if path.name.lower() in {".env", ".pypirc", "credentials", "id_rsa", "id_ed25519"}:
            raise RuntimeError(f"Credential-like tracked filename requires review: {name}")
        source = ROOT / path
        if source.is_symlink():
            raise RuntimeError(f"Symlink requires explicit review: {name}")
        clean = transform(name, source.read_bytes())
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(clean)
    report = {
        "schema_version": 1,
        "local_source_revision": commit,
        "tracked_file_count": len(names),
        "scanned_payload_counts": dict(counts),
        "credential_pattern_findings": 0,
        "personal_path_or_author_email_findings_after_transform": 0,
        "transformations": transformations,
        "policy": "Personal home-directory prefixes and private Git author emails replaced in textual evidence; changed gzip containers rebuilt; original hashes retained for provenance. Original local files and Git history are unchanged. No old Git ancestry is included. Numerical source, source archives and upstream notices remain byte-identical.",
        "limitations": "Pattern-based scan is not a proof of absence. Native wheels are audited separately. Hashes inside historical reports refer to original local evidence; this manifest maps changed evidence to public copies.",
    }
    serialized = json.dumps(report, indent=2) + "\n"
    (destination / "docs" / "public-source-manifest.json").write_text(serialized, encoding="utf-8", newline="\n")
    local_report = ROOT / "build" / (destination.name + "-local-audit.json")
    local_report.write_text(serialized, encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "public-release")
    args = parser.parse_args()
    report = export(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "transformations"}, indent=2))
    print(f"Sanitized payloads: {len(report['transformations'])}")


if __name__ == "__main__":
    main()
