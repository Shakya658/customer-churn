"""Run the evaluation export with notebook-compatible helpers and diagnostics."""

from __future__ import annotations

import builtins
import traceback
from pathlib import Path
from typing import Any

from export_evaluation_curves import export_curves

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = ROOT / "evaluation_export_error.txt"


def display(*objects: Any, **_: Any) -> Any:
    """Provide the display function normally injected by Jupyter."""
    return objects[-1] if objects else None


builtins.display = display

try:
    export_curves()
    DIAGNOSTIC_PATH.unlink(missing_ok=True)
except Exception:
    diagnostic = traceback.format_exc()
    DIAGNOSTIC_PATH.write_text(diagnostic, encoding="utf-8")
    print(diagnostic)
    raise
