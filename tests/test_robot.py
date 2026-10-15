from math import isclose, pi

import pytest

from _bootstrap import run_this_test_file
from jdcobot200 import load_robot_config
from jdcobot200.hardware import JDCobot200Robot


class FakeDriver:
    def close(self):
        pass


class FakeMovingDriver(FakeDriver):
    def __init__(self, config):
        self.positions = {joint.motor_id: 2048 - joint.home_offset_tick for joint in config.joints}
        self.individual_writes = []
        self.sync_writes = []

    def set_position(self, motor_id, position):
        self.individual_writes.append((motor_id, position))
        self.positions[motor_id] = position

    def set_positions_sync(self, positions):
        self.sync_writes.append(dict(positions))
        self.positions.update(positions)

    def get_position_filtered(self, motor_id, samples=3):
        return self.positions[motor_id]


def test_raw_logical_radian_conversion_uses_file_offset_and_direction():
    config = load_robot_config()
    robot = JDCobot200Robot(config, driver=FakeDriver())

    base = config.joint("base")
    wrist_pitch = config.joint("wrist_pitch")
    base_home = config.logical_center_tick - base.home_offset_tick
    wrist_positive_90 = (
        config.logical_center_tick - config.protocol_resolution // 4
        - wrist_pitch.home_offset_tick
    )
    assert robot.radian_to_raw_tick(base, 0.0) == base_home
    assert base_home + base.home_offset_tick == config.logical_center_tick
    assert robot.raw_to_logical_tick(base, base_home) == config.logical_center_tick
    assert isclose(
        robot.raw_to_radian(base, base_home + config.protocol_resolution // 4),
        pi / 2,
    )
    assert isclose(robot.raw_to_radian(wrist_pitch, wrist_positive_90), pi / 2)


def test_home_and_ready_targets_are_inside_hardware_limits():
    config = load_robot_config()
    robot = JDCobot200Robot(config, driver=FakeDriver())

    for pose in ("home", "ready"):
        target = robot.arm_pose_raw(pose)
        for joint in config.arm_joints:
            robot.validate_raw_target(joint, target[joint.name])


def test_raw_target_outside_limit_is_rejected():
    config = load_robot_config()
    robot = JDCobot200Robot(config, driver=FakeDriver())

    with pytest.raises(ValueError, match="outside"):
        robot.validate_raw_target(config.joint("shoulder"), 100)


def test_interpolation_uses_proven_individual_writes_by_default(monkeypatch):
    config = load_robot_config()
    driver = FakeMovingDriver(config)
    robot = JDCobot200Robot(config, driver=driver)
    monkeypatch.setattr("jdcobot200.hardware.robot.time.sleep", lambda seconds: None)

    start = {"base": driver.positions[1]}
    target = {"base": driver.positions[1] + 10}
    measured = robot.interpolate_raw(
        start, target, duration=0.1, rate_hz=20, max_step_ticks=5
    )

    assert measured == target
    assert driver.individual_writes
    assert not driver.sync_writes


if __name__ == "__main__":
    raise SystemExit(run_this_test_file(__file__))
