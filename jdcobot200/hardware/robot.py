"""Small, explicit robot-level wrapper around the low-level servo driver."""

from __future__ import annotations

from math import radians
import time
from typing import Iterable, Mapping

from jdcobot200.config import JointConfig, RobotConfig, load_robot_config

from .motor_drive import MiniFeetechDriver


class JDCobot200Robot:
    """Map named joints and calibrated radians to raw STS3215 ticks."""

    def __init__(self, config: RobotConfig | None = None, *, port: str | None = None,
                 driver: MiniFeetechDriver | None = None):
        self.config = config or load_robot_config()
        self.driver = driver or MiniFeetechDriver(
            port or self.config.port,
            self.config.baudrate,
            timeout=self.config.serial_timeout_seconds,
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def read_raw_positions(self, *, samples: int = 3) -> dict[str, int | None]:
        return {
            joint.name: self.driver.get_position_filtered(joint.motor_id, samples=samples)
            for joint in self.config.joints
        }

    def read_torque_states(self) -> dict[str, bool | None]:
        return {
            joint.name: self.driver.get_torque_enabled(joint.motor_id)
            for joint in self.config.joints
        }

    def raw_to_logical_tick(self, joint: JointConfig, raw_tick: int) -> int:
        # 원점에서는 raw encoder + offset == logical center(2048)입니다.
        return int(raw_tick) + joint.home_offset_tick

    def logical_to_raw_tick(self, joint: JointConfig, logical_tick: int) -> int:
        return int(logical_tick) - joint.home_offset_tick

    def raw_to_radian(self, joint: JointConfig, raw_tick: int) -> float:
        logical = self.raw_to_logical_tick(joint, raw_tick)
        return self.config.logical_tick_to_radian(joint.name, logical)

    def radian_to_raw_tick(self, joint: JointConfig, angle: float) -> int:
        logical = self.config.radian_to_logical_tick(joint.name, angle)
        return self.logical_to_raw_tick(joint, logical)

    def validate_raw_target(self, joint: JointConfig, raw_tick: int) -> None:
        if not joint.min_tick <= raw_tick <= joint.max_tick:
            raise ValueError(
                f"{joint.name}: raw target {raw_tick} outside "
                f"[{joint.min_tick}, {joint.max_tick}]"
            )

    def disable_torque(self, joints: Iterable[JointConfig] | None = None) -> None:
        """Disable torque one joint at a time and verify every register write."""
        selected = tuple(joints or self.config.joints)
        failed = []
        for joint in selected:
            disabled = self.driver.set_torque_verified(
                joint.motor_id, False, retries=5, delay=0.08
            )
            if disabled:
                print(f"ID {joint.motor_id} ({joint.name}): torque OFF 확인")
            else:
                failed.append(f"ID {joint.motor_id} ({joint.name})")
            # 연속 패킷 유실을 피하고 다음 서보가 응답할 시간을 확보한다.
            time.sleep(0.05)
        if failed:
            raise RuntimeError(
                "torque 해제 확인 실패: " + ", ".join(failed)
                + ". 일부 관절은 이미 OFF일 수 있으므로 로봇을 계속 지지하세요."
            )

    def hold_and_enable(self, raw_positions: Mapping[str, int],
                        joints: Iterable[JointConfig]) -> None:
        """Write the current goal, then enable torque using the proven legacy flow."""
        selected = tuple(joints)
        for joint in selected:
            current = int(raw_positions[joint.name])
            self.validate_raw_target(joint, current)
            self.driver.set_position(joint.motor_id, current)
        for joint in selected:
            # Motion must not depend on torque-register readback. The working
            # legacy programs use this direct write sequence.
            self.driver.set_torque(joint.motor_id, True)
            time.sleep(0.05)

    def interpolate_raw(self, start: Mapping[str, int], target: Mapping[str, int], *,
                        duration: float, rate_hz: float = 20.0,
                        max_step_ticks: int = 12,
                        sync_write: bool = False) -> dict[str, int]:
        """Send bounded position steps and return measured final ticks.

        Individual writes are the hardware-tested default. Sync Write remains
        opt-in until it has been verified on this controller/servo chain.
        """
        if duration <= 0 or rate_hz <= 0 or max_step_ticks <= 0:
            raise ValueError("duration, rate_hz, and max_step_ticks must be positive")
        joints = tuple(self.config.joint(name) for name in target)
        starts = {joint.name: int(start[joint.name]) for joint in joints}
        targets = {joint.name: int(target[joint.name]) for joint in joints}
        for joint in joints:
            self.validate_raw_target(joint, starts[joint.name])
            self.validate_raw_target(joint, targets[joint.name])

        time_steps = max(1, int(round(duration * rate_hz)))
        distance_steps = max(
            1,
            max(
                (abs(targets[j.name] - starts[j.name]) + max_step_ticks - 1)
                // max_step_ticks
                for j in joints
            ),
        )
        steps = max(time_steps, distance_steps)
        effective_duration = max(duration, steps / rate_hz)
        period = effective_duration / steps
        for index in range(1, steps + 1):
            alpha = index / steps
            command = {
                joint.motor_id: round(
                    starts[joint.name]
                    + alpha * (targets[joint.name] - starts[joint.name])
                )
                for joint in joints
            }
            if sync_write:
                self.driver.set_positions_sync(command)
            else:
                for motor_id, raw_tick in sorted(command.items()):
                    self.driver.set_position(motor_id, raw_tick)
                    time.sleep(0.002)
            time.sleep(period)

        measured = self.read_raw_positions(samples=3)
        return {name: measured[name] for name in target}

    def wait_for_raw_targets(self, target: Mapping[str, int], *, tolerance: int = 15,
                             timeout: float = 2.0) -> dict[str, int]:
        """Wait for feedback convergence or raise instead of waiting forever."""
        deadline = time.monotonic() + timeout
        last: dict[str, int | None] = {}
        while time.monotonic() < deadline:
            measured = self.read_raw_positions(samples=1)
            last = {name: measured[name] for name in target}
            if all(
                last[name] is not None and abs(last[name] - int(goal)) <= tolerance
                for name, goal in target.items()
            ):
                return {name: int(last[name]) for name in target}
            time.sleep(0.05)
        raise TimeoutError(f"target feedback timeout: target={dict(target)}, measured={last}")

    def arm_pose_raw(self, pose: str) -> dict[str, int]:
        if pose == "home":
            degrees = (0.0,) * self.config.dof
        elif pose == "ready":
            degrees = self.config.ready_pose_degrees
        else:
            raise ValueError(f"unknown pose: {pose}")
        return {
            joint.name: self.radian_to_raw_tick(joint, radians(angle))
            for joint, angle in zip(self.config.arm_joints, degrees)
        }
