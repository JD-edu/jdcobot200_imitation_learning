#!/usr/bin/env python3
"""입력한 절대 TCP 좌표로 단일 IK 해를 사용해 안전하게 이동합니다."""

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
    configure_ik_limits,
    execute_candidate,
    print_candidate,
    read_safe_start,
    validate_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="현재 자세 하나를 seed로 사용해 절대 TCP 좌표로 IK 이동합니다."
    )
    parser.add_argument(
        "--target-xyz-mm", nargs=3, type=float, required=True,
        metavar=("X", "Y", "Z"), help="목표 TCP 월드 좌표(mm)",
    )
    parser.add_argument("--move-time", type=float, default=4.0)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-joint-change-deg", type=float, default=90.0)
    parser.add_argument("--feedback-tolerance-ticks", type=int, default=50)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--port")
    args = parser.parse_args()
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

        # 현재 encoder 자세 하나만 seed로 넣으므로 단일 IK 해만 계산한다.
        ik = solver.solve(
            current_q, target_xyz, tolerance_m=args.tolerance_mm / 1000.0
        )
        candidate = validate_candidate(
            controller, solver, current_q, ik,
            max_joint_change_deg=args.max_joint_change_deg,
            path_steps=args.path_steps,
        )
        print_candidate("단일 해", candidate)
        if input("이 해로 실제 로봇을 이동하려면 MOVE 입력: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        viewer = MuJoCoViewer(controller.safety, "Single-solution Cartesian IK")
        try:
            _, measured_q = execute_candidate(
                controller, viewer, current_raw, current_q, candidate,
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
