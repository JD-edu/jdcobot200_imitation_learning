"""02/03 목표 좌표 IK 예제의 공통 안전 검사와 실제 이동 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ik_solver import IKResult, PathResult, SafeIK
from sim2real_control import JOINT_NAMES, MuJoCoViewer, SynchronizedController


ELBOW_IK_LOWER_DEG = -110.0
START_LIMIT_RECOVERY_MARGIN_DEG = 0.5


@dataclass(frozen=True)
class SafeCandidate:
    ik: IKResult
    path: PathResult
    change_deg: np.ndarray


def configure_ik_limits(controller: SynchronizedController) -> None:
    """06 과정에서만 elbow 범위를 하드웨어 끝단보다 여유 있게 확장한다."""
    index = JOINT_NAMES.index("elbow")
    joint = controller.config.joint("elbow")
    hardware_angles = (
        controller.robot.raw_to_radian(joint, joint.min_tick),
        controller.robot.raw_to_radian(joint, joint.max_tick),
    )
    hardware_lower = min(hardware_angles)
    requested_lower = np.deg2rad(ELBOW_IK_LOWER_DEG)
    if requested_lower < hardware_lower:
        raise RuntimeError(
            f"IK elbow 하한 {ELBOW_IK_LOWER_DEG:g}°가 하드웨어 하한 "
            f"{np.rad2deg(hardware_lower):.3f}°보다 작습니다."
        )
    controller.safety.lower[index] = requested_lower


def read_safe_start(
    controller: SynchronizedController,
) -> tuple[dict[str, int], np.ndarray]:
    """현재 자세를 읽고 작은 encoder 경계 오차만 제한값으로 보정한다."""
    raw, q_rad = controller.read(samples=5)
    result = controller.safety.evaluate(q_rad)
    if result.safe:
        return raw, q_rad

    clipped = np.clip(q_rad, controller.safety.lower, controller.safety.upper)
    excess_deg = float(np.max(np.abs(np.rad2deg(q_rad - clipped))))
    clipped_result = controller.safety.evaluate(clipped)
    if (result.reasons == ("joint_limit",)
            and excess_deg <= START_LIMIT_RECOVERY_MARGIN_DEG
            and clipped_result.safe):
        print(
            f"경고: 시작 관절 제한을 {excess_deg:.3f}° 초과하여 "
            "안전모델의 경계값으로 보정합니다."
        )
        return raw, clipped

    raise RuntimeError(
        f"현재 encoder 자세가 안전하지 않습니다: {result.reasons}, "
        f"q(deg)={np.round(np.rad2deg(q_rad), 3)}, raw={raw}"
    )


def validate_candidate(
    controller: SynchronizedController,
    solver: SafeIK,
    start_q: np.ndarray,
    ik: IKResult,
    *,
    max_joint_change_deg: float,
    path_steps: int,
) -> SafeCandidate:
    """IK 수렴, 목표 FK, 관절 변화, 전체 경로와 hardware tick을 검사한다."""
    if not ik.converged:
        raise RuntimeError(f"IK 미수렴: FK 오차 {ik.error_m * 1000:.3f} mm")

    goal_result = controller.safety.evaluate(ik.q_rad)
    if not goal_result.safe:
        raise RuntimeError(f"목표 FK unsafe: {goal_result.reasons}")

    change_deg = np.rad2deg(ik.q_rad - start_q)
    if np.max(np.abs(change_deg)) > max_joint_change_deg:
        raise RuntimeError(
            f"관절 변화 제한 {max_joint_change_deg:g}° 초과: "
            f"{np.round(change_deg, 3)}°"
        )

    path = solver.check_path(start_q, ik.q_rad, steps=path_steps)
    if not path.safe:
        raise RuntimeError(
            f"경로 FK unsafe: step={path.unsafe_step}/{path_steps - 1}, "
            f"reason={path.unsafe_result.reasons}"
        )
    controller.q_to_ticks(ik.q_rad)
    return SafeCandidate(ik=ik, path=path, change_deg=change_deg)


def print_candidate(label: str, candidate: SafeCandidate) -> None:
    print(f"{label} q(deg):", np.round(np.rad2deg(candidate.ik.q_rad), 3))
    print(f"{label} FK TCP(mm):", np.round(candidate.ik.tcp_position * 1000.0, 3))
    print(f"{label} FK 오차: {candidate.ik.error_m * 1000:.3f} mm")
    print(f"{label} 관절 변화(deg):", np.round(candidate.change_deg, 3))
    print(
        f"{label} 경로 최소 self/floor 여유: "
        f"{candidate.path.minimum_self_clearance * 1000:.2f}/"
        f"{candidate.path.minimum_floor_clearance * 1000:.2f} mm"
    )


def execute_candidate(
    controller: SynchronizedController,
    viewer: MuJoCoViewer,
    start_raw: dict[str, int],
    start_q: np.ndarray,
    candidate: SafeCandidate,
    *,
    move_time: float,
    feedback_tolerance_ticks: int,
) -> tuple[dict[str, int], np.ndarray]:
    """검사가 끝난 후보를 실제 로봇과 MuJoCo에 동기 적용한다."""
    try:
        return controller.move_to(
            start_raw,
            start_q,
            candidate.ik.q_rad,
            duration=move_time,
            feedback_tolerance_ticks=feedback_tolerance_ticks,
            on_step=lambda q: viewer.render(q, "Moving to Cartesian target"),
        )
    except TimeoutError as exc:
        measured_raw, measured_q = controller.read(samples=5)
        measured = controller.safety.evaluate(measured_q)
        if not measured.safe:
            raise RuntimeError(
                f"타임아웃 후 encoder 자세가 안전하지 않습니다: {measured.reasons}"
            ) from exc
        print(
            "경고: 목표 도달 타임아웃. 실제 encoder 자세로 재동기화합니다.",
            "오차(deg)=",
            np.round(np.rad2deg(candidate.ik.q_rad - measured_q), 3),
        )
        return measured_raw, measured_q
