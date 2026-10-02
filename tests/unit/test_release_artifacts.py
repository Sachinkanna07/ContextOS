"""Release validation rejects accidental local files and nested checkouts."""

import pytest
from tools.validate_release_artifacts import validate_member


def test_release_archive_rejects_local_state_and_nested_packages() -> None:
    for name in (
        ".kilo/worktrees/old/src/contextos/__init__.py", ".git/config", ".venv/pyvenv.cfg",
        ".pytest_cache/state", "src/contextos/__pycache__/demo.pyc", "docs/scratch.db",
        "docs/.env", "docs/.vscode/settings.json", "docs/old/src/contextos/__init__.py",
        "../private.txt",
    ):
        with pytest.raises(ValueError):
            validate_member(name, "src/contextos/")
    validate_member("src/contextos/__init__.py", "src/contextos/")
    validate_member("docs/final-benchmark.json", "src/contextos/")
