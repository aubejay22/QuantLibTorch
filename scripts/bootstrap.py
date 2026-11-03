"""Fetch and verify pinned native dependencies; network is needed on first run."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
QUANTLIB_COMMIT = "06edddbdb85f006c43a449b5fc846a00c67477f3"
QUANTLIB_TAG = "v1.40"
JSON_VERSION = "3.11.2"
JSON_SHA256 = "665fa14b8af3837966949e8eb0052d583e2ac105d3438baba9951785512cf921"
JSON_URL = f"https://github.com/nlohmann/json/releases/download/v{JSON_VERSION}/json.hpp"


def ensure_json_header(root=ROOT, *, download=True):
    """Supply Torch's missing header explicitly, never from ambient include paths."""
    include = root / "third_party" / f"nlohmann-json-{JSON_VERSION}" / "include"
    header = include / "nlohmann" / "json.hpp"
    if header.is_file():
        if hashlib.sha256(header.read_bytes()).hexdigest() != JSON_SHA256:
            raise SystemExit("Pinned nlohmann/json header checksum mismatch; preserve the changed "
                             "file and restore the verified dependency before building.")
        return include
    if not download:
        raise SystemExit("Pinned nlohmann/json header missing. Run python scripts/bootstrap.py.")
    with urlopen(JSON_URL, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != JSON_SHA256:
        raise SystemExit("Downloaded nlohmann/json header checksum mismatch; dependency not installed.")
    header.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=header.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(header)
    finally:
        temporary.unlink(missing_ok=True)
    return include


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-json", action="store_true",
                        help="Verify the JSON header without network; print its include directory for CMake")
    args = parser.parse_args()
    if args.check_json:
        print(ensure_json_header(download=False).as_posix())
        return
    target = ROOT / "third_party" / "QuantLib"
    if not target.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", QUANTLIB_TAG,
                        "https://github.com/lballabio/QuantLib.git", str(target)], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != QUANTLIB_COMMIT:
        raise SystemExit(f"Expected QuantLib {QUANTLIB_COMMIT}, got {actual}. Use a clean pinned checkout.")
    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise SystemExit("QuantLib checkout has changes; preserve them and supply a clean pinned checkout.")
    print(f"QuantLib {QUANTLIB_TAG}: {actual}")
    ensure_json_header()
    print(f"nlohmann/json {JSON_VERSION}: {JSON_SHA256}")


if __name__ == "__main__":
    main()
