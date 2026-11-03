"""Fetch the pinned QuantLib source. Requires git and network on first run."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
QUANTLIB_COMMIT = "06edddbdb85f006c43a449b5fc846a00c67477f3"
QUANTLIB_TAG = "v1.40"


def main():
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


if __name__ == "__main__":
    main()
