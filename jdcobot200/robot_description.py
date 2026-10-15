"""Stable paths to the canonical URDF and MJCF robot descriptions."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_ROOT = REPOSITORY_ROOT / "robot_description"
URDF_PATH = DESCRIPTION_ROOT / "urdf" / "jdcobot200.urdf"
MJCF_PATH = DESCRIPTION_ROOT / "mjcf" / "jdcobot200.xml"
DEFAULT_SCENE_PATH = DESCRIPTION_ROOT / "mjcf" / "scenes" / "empty.xml"


def validate_description_paths() -> None:
    """Raise a useful error if a checkout is missing canonical model assets."""
    required = (URDF_PATH, MJCF_PATH, DEFAULT_SCENE_PATH, DESCRIPTION_ROOT / "meshes")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing robot-description resources: " + ", ".join(missing))
