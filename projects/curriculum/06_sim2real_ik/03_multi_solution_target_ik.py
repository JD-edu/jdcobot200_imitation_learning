#!/usr/bin/env python3
"""여러 seed에서 목표 TCP의 IK 해를 찾고 최적 안전 경로로 이동합니다."""

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COURSE04 = PROJECT_ROOT / "projects/curriculum/04_sim2real_fk"
COURSE05 = PROJECT_ROOT / "projects/curriculum/05_mujoco_ik"
for path in (PROJECT_ROOT, COURSE04, COURSE05):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ik_solver import SafeIK  # noqa: E402
from sim2real_control import MuJoCoViewer, SynchronizedController  # noqa: E402
from target_ik_common import (  # noqa: E402
    SafeCandidate,
    configure_ik_limits,
    execute_candidate,
    print_candidate,
    read_safe_start,
    validate_candidate,
)


def is_duplicate(q_rad: np.ndarray, candidates: list[SafeCandidate]) -> bool:
    """모든 관절이 2도 이내인 기존 해는 같은 해로 취급한다."""
    return any(
        np.max(np.abs(np.rad2deg(q_rad - item.ik.q_rad))) <= 2.0
        for item in candidates
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="여러 IK 해 중 안전하고 현재 자세에 가까운 해로 이동합니다."
    )
    parser.add_argument(
        "--target-xyz-mm", nargs=3, type=float, required=True,
        metavar=("X", "Y", "Z"), help="목표 TCP 월드 좌표(mm)",
    )
    parser.add_argument("--seeds", type=int, default=64,
                        help="현재 자세를 포함한 IK 시작 seed 수(기본 64)")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--move-time", type=float, default=4.0)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-joint-change-deg", type=float, default=120.0)
    parser.add_argument("--feedback-tolerance-ticks", type=int, default=50)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--port")
    args = parser.parse_args()
    if not 2 <= args.seeds <= 500:
        raise ValueError("seeds는 2 이상 500 이하로 지정해야 합니다.")
    if args.move_time <= 0 or args.tolerance_mm <= 0 or args.max_joint_change_deg <= 0:
        raise ValueError("이동 시간, IK tolerance와 관절 변화 제한은 양수여야 합니다.")
    if not 1 <= args.feedback_tolerance_ticks <= 100 or args.path_steps < 2:
        raise ValueError("feedback tolerance는 1~100 tick, path steps는 2 이상이어야 합니다.")

    target_xyz = np.asarray(args.target_xyz_mm, dtype=float) / 1000.0
    with SynchronizedController(port=args.port) as controller:
        configure_ik_limits(controller)
        solver = SafeIK(controller.safety)
        current_raw, current_q = read_safe_start(controller)
        print("현재 q(deg):", np.round(np.rad2deg(current_q), 3))
        print("입력 목표 TCP(mm):", np.round(target_xyz * 1000.0, 3))

        rng = np.random.default_rng(args.random_seed)
        seeds = [current_q.copy()]
        for _ in range(args.seeds - 1):
            seed = rng.uniform(controller.safety.lower, controller.safety.upper)
            # wrist roll은 TCP 위치에 거의 영향을 주지 않으므로 현재 값을 유지한다.
            seed[4] = current_q[4]
            seeds.append(seed)

        safe_candidates: list[SafeCandidate] = []
        converged_count = 0
        rejected_count = 0
        for seed in seeds:
            try:
                ik = solver.solve(
                    seed, target_xyz, tolerance_m=args.tolerance_mm / 1000.0
                )
                if not ik.converged:
                    continue
                converged_count += 1
                if is_duplicate(ik.q_rad, safe_candidates):
                    continue
                candidate = validate_candidate(
                    controller, solver, current_q, ik,
                    max_joint_change_deg=args.max_joint_change_deg,
                    path_steps=args.path_steps,
                )
                safe_candidates.append(candidate)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                rejected_count += 1

        print(
            f"탐색 결과: seeds={len(seeds)}, 수렴={converged_count}, "
            f"안전한 고유 해={len(safe_candidates)}, 검사 거부={rejected_count}"
        )
        if not safe_candidates:
            raise RuntimeError("목표 좌표로 이동 가능한 안전한 IK 해가 없습니다.")

        # 먼저 최대 단일 관절 변화, 다음 전체 관절 변화와 FK 오차가 작은 해를 고른다.
        safe_candidates.sort(key=lambda item: (
            float(np.max(np.abs(item.change_deg))),
            float(np.linalg.norm(item.change_deg)),
            item.ik.error_m,
        ))
        for index, candidate in enumerate(safe_candidates, 1):
            print(
                f"후보 {index}: max-change={np.max(np.abs(candidate.change_deg)):.2f}°, "
                f"norm={np.linalg.norm(candidate.change_deg):.2f}°, "
                f"error={candidate.ik.error_m * 1000:.3f} mm, "
                f"q={np.round(np.rad2deg(candidate.ik.q_rad), 2)}"
            )
        selected = safe_candidates[0]
        print_candidate("선택 해", selected)
        if input("선택한 해로 실제 로봇을 이동하려면 MOVE 입력: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        viewer = MuJoCoViewer(controller.safety, "Multi-solution Cartesian IK")
        try:
            _, measured_q = execute_candidate(
                controller, viewer, current_raw, current_q, selected,
                move_time=args.move_time,
                feedback_tolerance_ticks=args.feedback_tolerance_ticks,
            )
            viewer.render(measured_q, "DONE | ESC or close window")
            print("도착 q(deg):", np.round(np.rad2deg(measured_q), 3))
            print("도착 TCP(mm):", np.round(
                controller.safety.evaluate(measured_q).tcp_position * 1000.0, 3
            ))
        finally:
            viewer.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError,
            np.linalg.LinAlgError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
