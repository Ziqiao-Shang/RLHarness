"""Central path helpers for the RLHarness source checkout."""

from __future__ import annotations

import os
from pathlib import Path

def _project_root() -> Path:
    """Locate the checkout while allowing an explicit installed-package override."""
    configured = os.environ.get("RLHARNESS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file():
        return source_root

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "prompts").is_dir():
        return cwd
    raise RuntimeError(
        "Cannot locate the RLHarness checkout. Set RLHARNESS_ROOT."
    )


ROOT = _project_root()
DEFAULT_MAPTAB_ROOT = Path(
    os.environ.get("MAPTAB_ROOT", str(ROOT.parent / "maptab_data"))
)


def maptab_root() -> Path:
    return Path(os.environ.get("MAPTAB_ROOT", DEFAULT_MAPTAB_ROOT))
