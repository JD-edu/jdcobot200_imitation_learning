import xml.etree.ElementTree as ET

from _bootstrap import run_this_test_file  # adds project root for direct execution
from jdcobot200.robot_description import (
    DEFAULT_SCENE_PATH,
    MJCF_PATH,
    URDF_PATH,
    validate_description_paths,
)


def test_canonical_descriptions_exist_and_parse():
    validate_description_paths()

    assert ET.parse(URDF_PATH).getroot().tag == "robot"
    assert ET.parse(MJCF_PATH).getroot().tag == "mujoco"
    assert ET.parse(DEFAULT_SCENE_PATH).getroot().tag == "mujoco"


if __name__ == "__main__":
    raise SystemExit(run_this_test_file(__file__))
