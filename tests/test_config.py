from math import isclose, pi

import pytest

from _bootstrap import run_this_test_file  # adds project root for direct execution
from jdcobot200 import load_robot_config


def test_canonical_joint_contract():
    config = load_robot_config()

    assert config.dof == 5
    assert config.joint_names == (
        "base",
        "shoulder",
        "elbow",
        "wrist_pitch",
        "wrist_roll",
        "gripper",
    )
    assert tuple(joint.motor_id for joint in config.joints) == (1, 2, 3, 4, 5, 6)
    assert tuple(joint.direction for joint in config.arm_joints) == (1, 1, 1, -1, 1)


def test_tick_radian_round_trip_respects_direction():
    config = load_robot_config()

    assert isclose(config.logical_tick_to_radian("base", 3072), pi / 2)
    assert isclose(config.logical_tick_to_radian("wrist_pitch", 3072), -pi / 2)
    assert config.radian_to_logical_tick("base", pi / 2) == 3072
    assert config.radian_to_logical_tick("wrist_pitch", pi / 2) == 1024


def test_raw_limits_include_offset():
    config = load_robot_config()

    config.validate_logical_tick("base", 2048)
    with pytest.raises(ValueError, match="outside"):
        config.validate_logical_tick("shoulder", 0)


def test_gripper_named_positions_are_configured_and_safe():
    gripper = load_robot_config().gripper_joint

    assert gripper.open_tick is not None
    assert gripper.closed_tick is not None
    assert gripper.open_tick != gripper.closed_tick
    assert gripper.min_tick <= gripper.open_tick <= gripper.max_tick
    assert gripper.min_tick <= gripper.closed_tick <= gripper.max_tick


if __name__ == "__main__":
    raise SystemExit(run_this_test_file(__file__))
