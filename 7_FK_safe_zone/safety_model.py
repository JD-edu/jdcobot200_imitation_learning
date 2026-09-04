#!/usr/bin/env python3
"""MINK-based FK, clearance, and collision checks for JDCobot200."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mink
import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE_PATH = ROOT / "scene.xml"
ARM_JOINT_NAMES = ("base", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
PROXY_NAMES = (
    "collision_base", "collision_shoulder", "collision_upper_arm",
    "collision_forearm", "collision_wrist", "collision_gripper",
)
# Parent/child proxies are allowed to overlap at their shared joint. Pairs
# separated by at least one intervening link must retain clearance.
SELF_COLLISION_PAIRS = tuple(
    (PROXY_NAMES[i], PROXY_NAMES[j])
    for i in range(len(PROXY_NAMES))
    for j in range(i + 2, len(PROXY_NAMES))
)
FLOOR_CHECK_PROXIES = PROXY_NAMES[1:]


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    tcp_position: np.ndarray
    minimum_self_clearance: float
    closest_self_pair: tuple[str, str]
    minimum_floor_clearance: float
    closest_floor_geom: str
    reasons: tuple[str, ...]


class JDCobotSafetyModel:
    def __init__(self, scene_path: Path = SCENE_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        self.joint_ids = np.array([self.model.joint(n).id for n in ARM_JOINT_NAMES])
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.actuator_ids = np.array([self.model.actuator(n).id for n in ARM_JOINT_NAMES])
        self.lower = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.upper = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.site_id = self.model.site("graspframe").id
        self.floor_id = self.model.geom("floor").id
        self.proxy_ids = {name: self.model.geom(name).id for name in PROXY_NAMES}

    def geom_distance(self, first: str | int, second: str | int) -> float:
        first_id = self.proxy_ids[first] if isinstance(first, str) else first
        second_id = self.proxy_ids[second] if isinstance(second, str) else second
        fromto = np.empty(6, dtype=float)
        return float(mujoco.mj_geomDistance(
            self.model, self.data, first_id, second_id, 2.0, fromto
        ))

    def evaluate(self, q: np.ndarray, collision_margin: float = 0.005,
                 floor_margin: float = 0.010) -> SafetyResult:
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            raise ValueError("q must contain five finite arm joint angles.")
        reasons = []
        if np.any(q < self.lower) or np.any(q > self.upper):
            reasons.append("joint_limit")
        configuration_q = self.configuration.q.copy()
        configuration_q[self.qpos_addresses] = np.clip(q, self.lower, self.upper)
        self.configuration.update(configuration_q)

        self_distances = [
            (self.geom_distance(a, b), (a, b)) for a, b in SELF_COLLISION_PAIRS
        ]
        minimum_self, closest_pair = min(self_distances, key=lambda item: item[0])
        floor_distances = [
            (self.geom_distance(self.floor_id, self.proxy_ids[name]), name)
            for name in FLOOR_CHECK_PROXIES
        ]
        minimum_floor, closest_floor = min(floor_distances, key=lambda item: item[0])
        if minimum_self < collision_margin:
            reasons.append("self_collision_margin")
        if minimum_floor < floor_margin:
            reasons.append("floor_margin")
        return SafetyResult(
            safe=not reasons,
            tcp_position=self.data.site_xpos[self.site_id].copy(),
            minimum_self_clearance=minimum_self,
            closest_self_pair=closest_pair,
            minimum_floor_clearance=minimum_floor,
            closest_floor_geom=closest_floor,
            reasons=tuple(reasons),
        )
