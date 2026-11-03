"""Shared native source fingerprint, without importing the package or PyTorch."""
from pathlib import Path
import hashlib


def native_source_digest(root: Path) -> str:
    paths = [root / "CMakeLists.txt", root / "scripts" / "prepare_quantlib.py",
             root / "scripts" / "prepare_packed.py",
             root / "scripts" / "native_metadata.py", root / "scripts" / "bootstrap.py",
             root / "scripts" / "build.py", root / "_build_support.py"]
    paths.extend((root / "cmake").rglob("*.cmake"))
    paths.extend(p for p in (root / "cpp").rglob("*")
                 if p.suffix in {".cpp", ".hpp", ".inc"})
    digest = hashlib.sha256()
    for path in sorted(paths):
        # Relative names distinguish identical content at different include paths.
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
