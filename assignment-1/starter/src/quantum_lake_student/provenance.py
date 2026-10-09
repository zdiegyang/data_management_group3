"""Code provenance shared by the Part I and Part II run records.

Both run.json files must record the code revision (Part I spec: "code
revision"; Part II spec: "code revision"), so the logic lives here once.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

STARTER_DIR = Path(__file__).resolve().parents[2]


def _read_git_head(git_dir: Path) -> str | None:
    """Resolve HEAD to a commit hash from the files in a .git directory."""
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None  # detached HEAD holds the hash itself
    ref = head[len("ref: "):]
    if (git_dir / ref).is_file():
        return (git_dir / ref).read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {ref}"):
                return line.split()[0]
    return None


def git_revision(start: Path = STARTER_DIR) -> str | None:
    """Commit hash of the code, with or without a `git` program.

    The workspace container has no `git`; compose.yaml mounts the repository's
    .git directory read-only and points QUANTUM_GIT_DIR at it.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=start, check=True, capture_output=True, text=True
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        pass
    candidates = [Path(os.environ["QUANTUM_GIT_DIR"])] if os.environ.get("QUANTUM_GIT_DIR") else []
    candidates += [directory / ".git" for directory in (start, *start.parents)]
    for git_dir in candidates:
        if (git_dir / "HEAD").is_file():
            return _read_git_head(git_dir)
    return None


def code_sha256(root: Path = STARTER_DIR) -> str:
    """Hash of the pipeline code (src/ and sql/), recorded even without git."""
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/**/*.py"), *root.glob("sql/**/*.sql")]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()
