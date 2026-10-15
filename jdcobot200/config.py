"""Typed loading and validation for the canonical robot configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "jdcobot200" / "robot.yaml"


@dataclass(frozen=True)
class JointConfig:
    name: str
    motor_id: int
    role: str
    direction: int
    home_offset_tick: int
    min_tick: int
    max_tick: int
    open_tick: int | None = None
    closed_tick: int | None = None

    def __post_init__(self) -> None:
        if self.role not in {"arm", "gripper"}:
            raise ValueError(f"{self.name}: role must be arm or gripper")
        if self.direction not in {-1, 1}:
            raise ValueError(f"{self.name}: direction must be -1 or 1")
        if not 0 <= self.min_tick < self.max_tick <= 4095:
            raise ValueError(f"{self.name}: invalid tick limits")
        for label, value in (("open_tick", self.open_tick), ("closed_tick", self.closed_tick)):
            if value is not None and not self.min_tick <= value <= self.max_tick:
                raise ValueError(f"{self.name}: {label} must be inside tick limits")


@dataclass(frozen=True)
class RobotConfig:
    name: str
    dof: int
    port: str
    baudrate: int
    serial_timeout_seconds: float
    protocol_resolution: int
    logical_center_tick: int
    joints: tuple[JointConfig, ...]
    ready_pose_degrees: tuple[float, ...]

    def __post_init__(self) -> None:
        names = [joint.name for joint in self.joints]
        motor_ids = [joint.motor_id for joint in self.joints]
        if len(names) != len(set(names)):
            raise ValueError("joint names must be unique")
        if len(motor_ids) != len(set(motor_ids)):
            raise ValueError("motor IDs must be unique")
        if sum(joint.role == "arm" for joint in self.joints) != self.dof:
            raise ValueError("dof must equal the number of arm joints")
        if len(self.ready_pose_degrees) != self.dof:
            raise ValueError("ready pose must contain one angle per arm joint")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def arm_joints(self) -> tuple[JointConfig, ...]:
        return tuple(joint for joint in self.joints if joint.role == "arm")

    @property
    def gripper_joint(self) -> JointConfig:
        grippers = tuple(joint for joint in self.joints if joint.role == "gripper")
        if len(grippers) != 1:
            raise ValueError("exactly one gripper joint is required")
        return grippers[0]

    def joint(self, name: str) -> JointConfig:
        try:
            return next(joint for joint in self.joints if joint.name == name)
        except StopIteration as exc:
            raise KeyError(f"unknown joint: {name}") from exc

    def logical_tick_to_radian(self, name: str, logical_tick: float) -> float:
        joint = self.joint(name)
        delta = logical_tick - self.logical_center_tick
        return joint.direction * delta * 2.0 * pi / self.protocol_resolution

    def radian_to_logical_tick(self, name: str, radians: float) -> int:
        joint = self.joint(name)
        tick = self.logical_center_tick + joint.direction * radians * self.protocol_resolution / (2.0 * pi)
        return int(round(tick))

    def validate_logical_tick(self, name: str, logical_tick: int) -> None:
        joint = self.joint(name)
        raw_tick = (logical_tick - joint.home_offset_tick) % self.protocol_resolution
        if not joint.min_tick <= raw_tick <= joint.max_tick:
            raise ValueError(
                f"{name}: logical tick {logical_tick} maps to raw tick {raw_tick}, "
                f"outside [{joint.min_tick}, {joint.max_tick}]"
            )


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    return mapping[key]


def load_robot_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RobotConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"robot configuration must be a mapping: {path}")

    serial = _required(data, "serial")
    encoder = _required(data, "encoder")
    raw_joints = _required(data, "joints")
    poses = _required(data, "poses")
    joints = tuple(
        JointConfig(
            name=str(_required(item, "name")),
            motor_id=int(_required(item, "motor_id")),
            role=str(_required(item, "role")),
            direction=int(_required(item, "direction")),
            home_offset_tick=int(_required(item, "home_offset_tick")),
            min_tick=int(_required(item, "min_tick")),
            max_tick=int(_required(item, "max_tick")),
            open_tick=None if item.get("open_tick") is None else int(item["open_tick"]),
            closed_tick=None if item.get("closed_tick") is None else int(item["closed_tick"]),
        )
        for item in raw_joints
    )
    return RobotConfig(
        name=str(_required(data, "name")),
        dof=int(_required(data, "dof")),
        port=str(_required(serial, "port")),
        baudrate=int(_required(serial, "baudrate")),
        serial_timeout_seconds=float(_required(serial, "timeout_seconds")),
        protocol_resolution=int(_required(encoder, "resolution")),
        logical_center_tick=int(_required(encoder, "logical_center_tick")),
        joints=joints,
        ready_pose_degrees=tuple(float(value) for value in _required(poses, "ready_degrees")),
    )
