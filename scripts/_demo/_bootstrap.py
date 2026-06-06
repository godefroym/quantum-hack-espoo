"""Environment bootstrap shared by the demonstration scripts.

Pins matplotlib's config dir and the generic cache dir under ``outputs/`` (so a sandboxed,
home-less run never tries to write to ``~/.cache``), and makes the ``src/`` layout importable
without an editable install. Import this module *before* anything that pulls in matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


def bootstrap() -> Path:
    """Prepare caches + import path and return the repository's ``outputs/`` directory."""
    mpl_cache = OUTPUTS / ".matplotlib"
    xdg_cache = OUTPUTS / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return OUTPUTS
