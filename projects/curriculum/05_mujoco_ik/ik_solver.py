"""과목 05의 MuJoCo Jacobian position IK와 안전검사 코드."""

# 이 파일은 직접 실행하는 CLI가 아니라 과목 05의 01~03과 과목 06이 import하는
# 공통 모듈입니다. 기본 명령줄 검사는 다음 파일을 사용합니다.
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/01_simulation_ik.py \
#   --delta-xyz-mm 5 0 0 --start-q-deg 0 20 -40 20 0
#
# Python 코드에서 사용하는 예:
# from ik_solver import SafeIK
# solver = SafeIK()
# result = solver.solve(start_q, target_xyz)
# path = solver.check_path(start_q, result.q_rad, steps=201)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COURSE03 = PROJECT_ROOT / "projects/curriculum/03_mujoco_fk_safety"
if str(COURSE03) not in sys.path:
    sys.path.insert(0, str(COURSE03))

from fk_safety import FKSafety, FKResult, JOINT_NAMES  # noqa: E402


@dataclass(frozen=True)
class IKResult:
    converged: bool
    q_rad: np.ndarray
    tcp_position: np.ndarray
    error_m: float
    iterations: int


@dataclass(frozen=True)
class PathResult:
    safe: bool
    unsafe_step: int | None
    unsafe_result: FKResult | None
    minimum_self_clearance: float
    minimum_floor_clearance: float


class SafeIK:
    def __init__(self, fk: FKSafety | None = None) -> None:
        self.fk = fk or FKSafety()
        self.model = self.fk.model
        self.site_id = self.fk.site_id
        self.dof_addresses = self.model.jnt_dofadr[self.fk.joint_ids]
        # Shoulder부터 TCP까지 각 관절 anchor 사이 길이의 합은 위치 IK가 도달할 수
        # 있는 절대 최대 반경입니다. q=0에서 한 번 계산해 명백히 불가능한 목표를
        # tolerance와 무관하게 조기에 거부합니다.
        self.fk.evaluate(np.zeros(5))
        anchor_points = [
            self.fk.data.xanchor[joint_id].copy()
            for joint_id in self.fk.joint_ids[1:]
        ]
        anchor_points.append(self.fk.data.site_xpos[self.site_id].copy())
        self.maximum_reach_m = float(sum(
            np.linalg.norm(second - first)
            for first, second in zip(anchor_points, anchor_points[1:])
        ))
        self.shoulder_joint_id = self.fk.joint_ids[1]

    def solve(
        self, start_q: np.ndarray, target_xyz: np.ndarray, *,
        tolerance_m: float = 0.001, max_iterations: int = 400,
        damping: float = 1e-3, max_step_rad: float = 0.05,
    ) -> IKResult:
        q = np.asarray(start_q, dtype=float).copy()
        target = np.asarray(target_xyz, dtype=float)
        if q.shape != (5,) or target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("start_q는 5개, target_xyz는 유한한 숫자 3개여야 합니다.")
        if tolerance_m <= 0 or max_iterations < 1 or damping <= 0 or max_step_rad <= 0:
            raise ValueError("IK 설정값은 양수여야 합니다.")
        q = np.clip(q, self.fk.lower, self.fk.upper)
        self.fk.evaluate(q)
        shoulder_position = self.fk.data.xanchor[self.shoulder_joint_id].copy()
        target_radius = float(np.linalg.norm(target - shoulder_position))
        if target_radius > self.maximum_reach_m + 1e-6:
            excess_mm = (target_radius - self.maximum_reach_m) * 1000.0
            raise ValueError(
                "목표 TCP가 최대 도달 반경 밖입니다: "
                f"shoulder→target={target_radius*1000:.2f} mm, "
                f"maximum={self.maximum_reach_m*1000:.2f} mm, "
                f"초과={excess_mm:.2f} mm"
            )
        jac_pos = np.zeros((3, self.model.nv))

        for iteration in range(1, max_iterations + 1):
            current = self.fk.evaluate(q).tcp_position
            error = target - current
            error_norm = float(np.linalg.norm(error))
            if error_norm <= tolerance_m:
                return IKResult(True, q.copy(), current, error_norm, iteration)

            mujoco.mj_jacSite(self.model, self.fk.data, jac_pos, None, self.site_id)
            jac = jac_pos[:, self.dof_addresses]
            # Damped least squares: 특이점 부근에서도 큰 관절 점프를 억제합니다.
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + (damping ** 2) * np.eye(3), error
            )
            largest = float(np.max(np.abs(delta)))
            if largest > max_step_rad:
                delta *= max_step_rad / largest
            q = np.clip(q + delta, self.fk.lower, self.fk.upper)

        current = self.fk.evaluate(q).tcp_position
        error_norm = float(np.linalg.norm(target - current))
        return IKResult(False, q.copy(), current, error_norm, max_iterations)

    def check_path(
        self, start_q: np.ndarray, goal_q: np.ndarray, *, steps: int = 201,
        self_margin: float = 0.005, floor_margin: float = 0.010,
    ) -> PathResult:
        if steps < 2:
            raise ValueError("경로 steps는 2 이상이어야 합니다.")
        min_self = float("inf")
        min_floor = float("inf")
        first_unsafe = None
        for index, q in enumerate(np.linspace(start_q, goal_q, steps)):
            result = self.fk.evaluate(q, self_margin, floor_margin)
            min_self = min(min_self, result.self_clearance)
            min_floor = min(min_floor, result.floor_clearance)
            if not result.safe and first_unsafe is None:
                first_unsafe = (index, result)
        return PathResult(
            safe=first_unsafe is None,
            unsafe_step=None if first_unsafe is None else first_unsafe[0],
            unsafe_result=None if first_unsafe is None else first_unsafe[1],
            minimum_self_clearance=min_self,
            minimum_floor_clearance=min_floor,
        )


def print_solution(start_q, target_xyz, result: IKResult, path: PathResult) -> None:
    print("관절 순서:", ", ".join(JOINT_NAMES))
    print("시작 q(deg):", np.round(np.rad2deg(start_q), 3))
    print("목표 q(deg):", np.round(np.rad2deg(result.q_rad), 3))
    print("목표 TCP(m):", np.round(target_xyz, 6))
    print("도달 TCP(m):", np.round(result.tcp_position, 6))
    print(f"FK 오차: {result.error_m * 1000:.3f} mm ({result.iterations} iterations)")
    print(f"경로 최소 self/floor 여유: {path.minimum_self_clearance*1000:.2f} / {path.minimum_floor_clearance*1000:.2f} mm")
