"""MuJoCo FK와 단순 collision proxy 기반 자세 안전 검사."""

# 이 파일은 직접 실행하는 CLI가 아니라 02~04 예제가 import하는 공통 모듈입니다.
# 사용 예:
# from fk_safety import FKSafety
# result = FKSafety().evaluate([0.0, 0.0, 0.0, 0.0, 0.0])  # 단위 radian

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 검증을 마친 공식 단일 MJCF를 사용합니다. 과목 02에서 변환된 오래된 generated
# 후보는 q=0 링크 변환이 공식 모델과 달라 Sim2Real 초기 자세에 사용하면 안 됩니다.
SCENE_PATH = PROJECT_ROOT / "robot_description/mjcf/scenes/empty.xml"
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
PROXY_NAMES = (
    "collision_base", "collision_shoulder", "collision_upper_arm",
    "collision_forearm", "collision_wrist", "collision_gripper",
)
# 바로 이웃한 링크는 관절에서 원래 맞닿으므로 검사하지 않는다.
SELF_COLLISION_PAIRS = tuple(
    (PROXY_NAMES[i], PROXY_NAMES[j])
    for i in range(len(PROXY_NAMES))
    for j in range(i + 2, len(PROXY_NAMES))
)


@dataclass
class FKResult:
    safe: bool
    tcp_position: np.ndarray
    tcp_matrix: np.ndarray
    self_clearance: float
    closest_pair: tuple[str, str]
    floor_clearance: float
    closest_floor_link: str
    reasons: tuple[str, ...]


class FKSafety:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        actuator_ids = [self.model.actuator(name).id for name in JOINT_NAMES]
        self.lower = self.model.actuator_ctrlrange[actuator_ids, 0]
        self.upper = self.model.actuator_ctrlrange[actuator_ids, 1]
        self.site_id = self.model.site("graspframe").id
        self.floor_id = self.model.geom("floor").id
        self.proxy_ids = {name: self.model.geom(name).id for name in PROXY_NAMES}

    def _distance(self, first: int, second: int) -> float:
        nearest_points = np.empty(6)
        return float(mujoco.mj_geomDistance(
            self.model, self.data, first, second, 2.0, nearest_points
        ))

    def evaluate(
        self, q_rad: np.ndarray, self_margin: float = 0.005,
        floor_margin: float = 0.010,
    ) -> FKResult:
        q_rad = np.asarray(q_rad, dtype=float)
        if q_rad.shape != (5,) or not np.all(np.isfinite(q_rad)):
            raise ValueError("관절각은 유한한 숫자 5개여야 합니다.")

        reasons = []
        if np.any(q_rad < self.lower) or np.any(q_rad > self.upper):
            reasons.append("joint_limit")
        self.data.qpos[self.qpos_addresses] = np.clip(q_rad, self.lower, self.upper)
        mujoco.mj_forward(self.model, self.data)

        self_distances = [
            (self._distance(self.proxy_ids[a], self.proxy_ids[b]), (a, b))
            for a, b in SELF_COLLISION_PAIRS
        ]
        self_clearance, closest_pair = min(self_distances, key=lambda item: item[0])
        floor_distances = [
            (self._distance(self.floor_id, self.proxy_ids[name]), name)
            for name in PROXY_NAMES[1:]
        ]
        floor_clearance, closest_floor = min(floor_distances, key=lambda item: item[0])
        if self_clearance < self_margin:
            reasons.append("self_collision")
        if floor_clearance < floor_margin:
            reasons.append("floor_clearance")

        return FKResult(
            safe=not reasons,
            tcp_position=self.data.site_xpos[self.site_id].copy(),
            tcp_matrix=self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
            self_clearance=self_clearance,
            closest_pair=closest_pair,
            floor_clearance=floor_clearance,
            closest_floor_link=closest_floor,
            reasons=tuple(reasons),
        )

    def check_path(self, start: np.ndarray, goal: np.ndarray, steps: int = 101):
        """직선 관절 경로의 모든 중간 자세를 검사하고 최초 위험 결과를 반환합니다."""
        for index, q_rad in enumerate(np.linspace(start, goal, steps)):
            result = self.evaluate(q_rad)
            if not result.safe:
                return False, index, result
        return True, steps - 1, result
