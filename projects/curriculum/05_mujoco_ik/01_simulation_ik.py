#!/usr/bin/env python3
"""MuJoCo에서 목표 TCP의 IK를 풀고 FK 및 전체 경로를 검증합니다."""

# 현재 TCP에서 X 방향으로 5 mm 상대 이동:
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/01_simulation_ik.py \
#   --delta-xyz-mm 5 0 0 --start-q-deg 0 20 -40 20 0
#
# 로봇 base 좌표계의 절대 TCP 위치(m)로 이동:
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/01_simulation_ik.py \
#   --target-xyz 0.10 0.05 0.30 --start-q-deg 0 20 -40 20 0
#
# FK 허용 오차와 안전 경로 표본 수 지정:
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/01_simulation_ik.py \
#   --delta-xyz-mm 0 0 5 --tolerance-mm 0.5 --path-steps 301

import argparse
import sys

import numpy as np

from ik_solver import SafeIK, print_solution


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo position IK와 안전 경로를 검사합니다.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    target.add_argument("--delta-xyz-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--start-q-deg", nargs=5, type=float, default=[0] * 5)
    parser.add_argument(
        "--tolerance-mm", type=float, default=1.0,
        help="IK 종료 허용오차(mm). 큰 값은 정밀도를 낮추며 도달 범위를 늘리지 않습니다.",
    )
    parser.add_argument("--path-steps", type=int, default=201)
    args = parser.parse_args()

    solver = SafeIK()
    start_q = np.deg2rad(np.asarray(args.start_q_deg, dtype=float))
    start = solver.fk.evaluate(start_q)
    if not start.safe:
        raise RuntimeError(f"시작 자세가 안전하지 않습니다: {start.reasons}")
    target_xyz = (
        np.asarray(args.target_xyz, dtype=float) if args.target_xyz is not None
        else start.tcp_position + np.asarray(args.delta_xyz_mm, dtype=float) / 1000.0
    )
    result = solver.solve(start_q, target_xyz, tolerance_m=args.tolerance_mm / 1000.0)
    if not result.converged:
        raise RuntimeError(f"IK 미수렴: 최종 오차 {result.error_m * 1000:.2f} mm")
    path = solver.check_path(start_q, result.q_rad, steps=args.path_steps)
    print_solution(start_q, target_xyz, result, path)
    if not path.safe:
        print(f"UNSAFE step={path.unsafe_step}, reason={path.unsafe_result.reasons}")
        return 2
    print("SAFE: 시뮬레이션 IK/FK와 전체 경로 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
