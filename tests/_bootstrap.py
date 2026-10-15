"""Test runner support for direct ``python3 tests/test_*.py`` execution."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_this_test_file(path: str) -> int:
    """Run one test module verbosely when it is launched as a Python script."""
    import pytest

    return pytest.main([path, "-v"])
