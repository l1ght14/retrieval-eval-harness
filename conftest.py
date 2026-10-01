"""Make `src` importable as a package from the repository root.

The project uses one import convention - `from src.<module> import ...` everywhere - so a
test can import the library and `eval/run_eval.py` can drive it without either adding
`src/` to `sys.path` and creating a second, conflicting way to reach the same module.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
