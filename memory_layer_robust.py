"""Compatibility wrapper for :mod:`amem.memory_layer_robust`."""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amem import memory_layer_robust as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
