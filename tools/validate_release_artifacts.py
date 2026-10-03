"""Validate release archive boundaries and metadata without installing dependencies."""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "1.0.0rc3"
FORBIDDEN_PARTS = {
    ".git", ".kilo", ".venv", "venv", ".pytest_cache", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build",
    "worktrees", "checkouts",
}
REQUIRED_MODULES = {
    "__init__.py", "cli/app.py", "daemon/wiring.py", "services/retrieval.py",
    "services/graph.py", "services/temporal.py", "storage/telemetry_repo.py",
    "mcp/server.py", "connectors/manager.py", "services/explainability.py",
    "services/inspection.py", "benchmarks/final.py", "demo.py",
}


def validate_member(name: str, package_root: str) -> None:
    """Reject local state and duplicate ContextOS trees wherever they occur."""
    path = PurePosixPath(name)
    parts = {part.lower() for part in path.parts}
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    if parts & FORBIDDEN_PARTS or re.search(r"(?:^|/)[a-z]:", name, re.I):
        raise ValueError(f"Local development state in archive: {name}")
    leaf = path.name.lower()
    if (
        "scratch" in leaf or leaf.startswith(".env")
        or leaf.endswith((".db", ".sqlite", ".sqlite3", ".pyc", ".pem", ".key"))
        or ".db-" in leaf
    ):
        raise ValueError(f"Local or private file in archive: {name}")
    if name.endswith("contextos/__init__.py") and name != package_root + "__init__.py":
        raise ValueError(f"Nested ContextOS checkout: {name}")


def _validate_metadata(payload: bytes) -> None:
    metadata = email.message_from_bytes(payload)
    if metadata["Name"] != "contextos-memory-runtime" or metadata["Version"] != VERSION:
        raise ValueError("Release metadata name/version mismatch")
    if metadata["Requires-Python"] != ">=3.12":
        raise ValueError("Requires-Python mismatch")
    requirements = metadata.get_all("Requires-Dist", [])
    if "embeddings" not in metadata.get_all("Provides-Extra", []):
        raise ValueError("Missing embeddings extra")
    embedding_requirements = [r for r in requirements if r.startswith("sentence-transformers")]
    if not embedding_requirements or any(
        "extra == 'embeddings'" not in r for r in embedding_requirements
    ):
        raise ValueError("SentenceTransformers must be scoped to the embeddings extra")
    if any(r.startswith("torch") and "extra ==" not in r for r in requirements):
        raise ValueError("Torch must not be a core dependency")


def validate_artifacts(wheel: Path, sdist: Path) -> dict[str, int]:
    """Validate rebuilt archives; raise on contamination or missing release content."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for name in names:
            validate_member(name, "contextos/")
        if not {"contextos/" + name for name in REQUIRED_MODULES} <= names:
            raise ValueError("Wheel is missing required modules")
        _validate_metadata(archive.read(f"contextos_memory_runtime-{VERSION}.dist-info/METADATA"))
        if f'__version__ = "{VERSION}"'.encode() not in archive.read("contextos/__init__.py"):
            raise ValueError("Wheel package version mismatch")
    prefix = f"contextos_memory_runtime-{VERSION}/"
    with tarfile.open(sdist) as source:
        names = set(source.getnames())
        for member in source.getmembers():
            if not member.name.startswith(prefix) or not member.isfile():
                raise ValueError(f"Unexpected sdist member: {member.name}")
            validate_member(member.name[len(prefix):], "src/contextos/")
        if not {prefix + "src/contextos/" + name for name in REQUIRED_MODULES} <= names:
            raise ValueError("Sdist is missing required modules")
        metadata_file = source.extractfile(prefix + "PKG-INFO")
        project_file = source.extractfile(prefix + "pyproject.toml")
        if metadata_file is None or project_file is None:
            raise ValueError("Sdist metadata missing")
        _validate_metadata(metadata_file.read())
        if tomllib.loads(project_file.read().decode())["project"]["version"] != VERSION:
            raise ValueError("Sdist project version mismatch")
    return {"wheel_bytes": wheel.stat().st_size, "sdist_bytes": sdist.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    sizes = validate_artifacts(
        args.dist / f"contextos_memory_runtime-{VERSION}-py3-none-any.whl",
        args.dist / f"contextos_memory_runtime-{VERSION}.tar.gz",
    )
    print(f"Release artifacts PASS: {sizes}")


if __name__ == "__main__":
    main()
