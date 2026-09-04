"""Import LeRobot ACT without eagerly importing unrelated optional policies."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def enable_act_only_imports() -> None:
    """Work around LeRobot 0.4.4's eager optional-policy package imports.

    This does not replace or modify ACT. It only installs a namespace package
    placeholder so importing ACT does not first import GR00T/XVLA dependencies.
    Saved policy and processor files remain standard LeRobot artifacts.
    """
    if "lerobot.policies" in sys.modules:
        return
    import lerobot

    package = types.ModuleType("lerobot.policies")
    package.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
    package.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = package

