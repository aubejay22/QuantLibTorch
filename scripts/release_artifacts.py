"""Bind release wheels to reviewed GitHub runs and verify index-delivered bytes.

This script uses only the standard library. Authentication is read from the
ephemeral GITHUB_TOKEN environment variable and is never written to evidence.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(url, *, authenticated=False):
    headers = {"User-Agent": "QuantLibTorch-release/0.3.0"}
    if authenticated:
        require(url.startswith("https://api.github.com/"), "Invalid authenticated API URL")
        headers.update({"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"})
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        return json.load(response)


def inspect_wheel(wheel):
    require(re.fullmatch(r"quantlibtorch-[A-Za-z0-9.]+-cp313-cp313-win_amd64\.whl", wheel.name),
            "Release permits only the validated Windows CPython 3.13 wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Wheel contains duplicate archive members")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(metadata_paths) == 1, "Expected exactly one wheel METADATA")
        info = metadata_paths[0].removesuffix("METADATA")
        allowed_info = {"METADATA", "WHEEL", "RECORD", "top_level.txt", "LICENSE",
                        "QuantLib.txt", "Boost.txt", "PyTorch.txt", "JSON.txt", "Abseil.txt", "NOTICE.txt"}
        private_path = rb"(?i)(?:[A-Z]:[\\/]+Users[\\/]|/(?:home|Users)/)"
        for name in names:
            parts = name.split("/")
            require("\\" not in name and not name.startswith("/") and ".." not in parts,
                    "Unsafe wheel member path")
            allowed = (name.startswith("quantlibtorch/") and name.endswith(".py") or
                       name in {"quantlibtorch/_native/build_info.json", "quantlibtorch/_native/quantlibtorch_native.dll"} or
                       name.startswith(info) and name.removeprefix(info) in allowed_info)
            require(allowed, f"Unexpected wheel member: {name}")
            payload = archive.read(name)
            require(not re.search(private_path, payload) and not re.search(private_path, payload.replace(b"\0", b"")),
                    "Wheel contains a private home-directory path")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        require(metadata["Name"] == "quantlibtorch", "Unexpected distribution")
        require(metadata["Version"] == "0.3.0", "Rehearsal and production both require stable 0.3.0 metadata")
        require(set(metadata["Requires-Python"].replace(" ", "").split(",")) == {">=3.13", "<3.14"},
                "Wheel must constrain Python to the validated 3.13 minor")
        require("BSD 3-Clause License" in metadata.get("License", ""), "BSD-3-Clause metadata missing")
        require(metadata.get_all("Requires-Dist", []).count("torch==2.10.0+cu126") == 1,
                "Wheel must pin the validated Torch CUDA variant")
        notices = {}
        for filename in ("LICENSE", "QuantLib.txt", "Boost.txt", "PyTorch.txt", "JSON.txt", "Abseil.txt", "NOTICE.txt"):
            entries = [name for name in names if ".dist-info/" in name and name.endswith("/" + filename)]
            require(len(entries) == 1, f"Required license/notice missing or duplicated: {filename}")
            contents = archive.read(entries[0])
            require(bool(contents.strip()), f"Empty license/notice: {filename}")
            notices[filename] = hashlib.sha256(contents).hexdigest()
        native = json.loads(archive.read("quantlibtorch/_native/build_info.json"))
        require(native["torch_version"] == "2.10.0+cu126", "Unexpected native Torch variant")
        require(native["platform"] == "win-amd64" and native["python_tag"] == "cpython-313" and
                native["cxx11_abi"] is True, "Unexpected native platform/Python/C++ ABI")
        require(native["quantlib_commit"] == "06edddbdb85f006c43a449b5fc846a00c67477f3",
                "Native QuantLib source differs from the reviewed pin")
        require(re.fullmatch("[0-9a-f]{64}", native["native_source_sha256"]), "Invalid native source digest")
        binary = archive.read("quantlibtorch/_native/quantlibtorch_native.dll")
        require(native["library_sha256"] == hashlib.sha256(binary).hexdigest(), "Native library digest mismatch")
    return {"filename": wheel.name, "sha256": digest(wheel), "size": wheel.stat().st_size,
            "version": metadata["Version"], "requires_python": metadata["Requires-Python"],
            "notices_sha256": notices, "native": native}


def manifest(args):
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         text=True).strip(), "Tracked checkout must be clean")
    wheels = list(args.directory.glob("*.whl"))
    require(len(wheels) == 1, "Expected exactly one release wheel")
    result = inspect_wheel(wheels[0])
    result.update({"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "repository": os.environ["GITHUB_REPOSITORY"],
                   "candidate_run_id": int(os.environ["GITHUB_RUN_ID"])})
    (args.directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def check_run(run_id, workflow, revision):
    require(str(run_id).isdigit(), "Workflow run ID must be numeric")
    repository = os.environ["GITHUB_REPOSITORY"]
    result = read_json(f"https://api.github.com/repos/{repository}/actions/runs/{run_id}", authenticated=True)
    require(result["repository"]["full_name"] == repository, "Run belongs to another repository")
    require(result["head_repository"]["full_name"] == repository, "Fork artifacts are forbidden")
    require(result["path"] == f".github/workflows/{workflow}", "Unexpected source workflow")
    require(result["event"] == "workflow_dispatch", "Only manually reviewed builds may publish")
    require(result["status"] == "completed" and result["conclusion"] == "success", "Source run did not succeed")
    require(result["head_sha"] == revision, "Run revision differs from release checkout")
    repo = read_json(f"https://api.github.com/repos/{repository}", authenticated=True)
    require(result["head_branch"] == repo["default_branch"], "Source run must use the default branch")


def verify(args):
    require(re.fullmatch("[0-9a-f]{64}", args.sha256), "Supply the SHA256 of the exact CUDA-validated wheel")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(revision == os.environ["GITHUB_SHA"], "Checkout differs from dispatch revision")
    check_run(args.candidate_run, "release-candidate.yml", revision)
    data = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    require(data["repository"] == os.environ["GITHUB_REPOSITORY"], "Manifest repository mismatch")
    require(data["git_revision"] == revision, "Manifest Git revision mismatch")
    require(data["candidate_run_id"] == int(args.candidate_run), "Manifest build run mismatch")
    wheels = list(args.directory.glob("*.whl"))
    require(len(wheels) == 1, "Expected one release wheel")
    actual = inspect_wheel(wheels[0])
    for key, value in actual.items():
        require(data[key] == value, f"Manifest mismatch: {key}")
    require(actual["sha256"] == args.sha256, "Wheel differs from the CUDA-validated artifact")
    if args.stage == "production":
        require(os.environ["GITHUB_REF"] == "refs/tags/v0.3.0", "Production requires v0.3.0 tag")
        require(actual["version"] == "0.3.0", "Production requires stable 0.3.0 metadata")
        require(args.testpypi_run, "Production requires a successful TestPyPI verification run")
        check_run(args.testpypi_run, "release.yml", revision)
        repository = os.environ["GITHUB_REPOSITORY"]
        jobs = read_json(f"https://api.github.com/repos/{repository}/actions/runs/{args.testpypi_run}/jobs?per_page=100",
                         authenticated=True)["jobs"]
        for name in ("testpypi", "verify-testpypi"):
            require(any(job["name"] == name and job["conclusion"] == "success" for job in jobs),
                    f"TestPyPI run has no successful {name} job")
    else:
        repo = read_json(f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}", authenticated=True)
        require(os.environ["GITHUB_REF"] == "refs/heads/" + repo["default_branch"],
                "TestPyPI must be dispatched from the default branch")
    print(f"Verified {actual['filename']} at Git {revision}: {args.sha256}")


def download(args):
    data = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    require(re.fullmatch(r"quantlibtorch-0\.3\.0-cp313-cp313-win_amd64\.whl", data["filename"]),
            "Invalid published wheel filename")
    require(data["version"] == "0.3.0" and re.fullmatch("[0-9a-f]{64}", data["sha256"]),
            "Invalid published version or digest")
    index = "test.pypi.org" if args.index == "testpypi" else "pypi.org"
    # Poll only an unavailable release; fail closed on a delivered but wrong file.
    for attempt in range(12):
        try:
            result = read_json(f"https://{index}/pypi/quantlibtorch/{data['version']}/json")
            matches = [entry for entry in result["urls"] if entry["filename"] == data["filename"]]
            if matches:
                break
        except HTTPError as error:
            if error.code != 404:
                raise
        if attempt == 11:
            raise SystemExit("Expected release wheel is not available from the index")
        time.sleep(10)
    require(len(matches) == 1, "Ambiguous index wheel")
    entry = matches[0]
    require(not entry["yanked"], "Published wheel is yanked")
    require(entry["digests"]["sha256"] == data["sha256"], "Index digest differs from reviewed artifact")
    require(entry["url"].startswith("https://files.pythonhosted.org/") or
            entry["url"].startswith("https://test-files.pythonhosted.org/"), "Unexpected wheel download host")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / data["filename"]
    require(destination.name == data["filename"], "Invalid wheel filename")
    with urlopen(entry["url"], timeout=120) as response:
        destination.write_bytes(response.read())
    require(digest(destination) == data["sha256"], "Downloaded bytes differ from reviewed artifact")
    print(f"Downloaded verified wheel from https://{index}/project/quantlibtorch/{data['version']}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("manifest")
    create.add_argument("directory", type=Path)
    validate = commands.add_parser("verify")
    validate.add_argument("directory", type=Path)
    validate.add_argument("--candidate-run", required=True)
    validate.add_argument("--sha256", required=True)
    validate.add_argument("--stage", choices=("testpypi", "production"), required=True)
    validate.add_argument("--testpypi-run", default="")
    fetch = commands.add_parser("download")
    fetch.add_argument("directory", type=Path)
    fetch.add_argument("--index", choices=("testpypi", "pypi"), required=True)
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"manifest": manifest, "verify": verify, "download": download}[args.command](args)


if __name__ == "__main__":
    main()
