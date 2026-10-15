#!/usr/bin/env python3
"""한 자세 또는 그 자세까지의 전체 관절 경로를 안전 검사합니다."""

# 실행 예(HOME에서 목표 자세까지 101단계 검사):
# /usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/03_check_pose.py 0 20 -40 20 0
# 실행 예(시작 자세와 검사 단계 수 지정):
# /usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/03_check_pose.py 0 30 -50 20 0 --start-deg 0 20 -40 20 0 --steps 201

import argparse
import sys
import numpy as np

from fk_safety import FKSafety, JOINT_NAMES


parser = argparse.ArgumentParser()
parser.add_argument("goal_deg", nargs=5, type=float, metavar=("J1", "J2", "J3", "J4", "J5"))
parser.add_argument("--start-deg", nargs=5, type=float, default=[0, 0, 0, 0, 0])
parser.add_argument("--steps", type=int, default=101)
args = parser.parse_args()

if args.steps < 2:
    raise SystemExit("--steps는 2 이상이어야 합니다.")
safety = FKSafety()
safe, index, result = safety.check_path(
    np.deg2rad(args.start_deg), np.deg2rad(args.goal_deg), args.steps
)
print("관절 순서:", ", ".join(JOINT_NAMES))
print("TCP 위치(m):", np.round(result.tcp_position, 5))
print(f"자기충돌 여유: {result.self_clearance * 1000:.1f} mm {result.closest_pair}")
print(f"바닥 여유: {result.floor_clearance * 1000:.1f} mm ({result.closest_floor_link})")
if not safe:
    print(f"UNSAFE: step {index}/{args.steps - 1}, 이유={', '.join(result.reasons)}")
    sys.exit(2)
print("SAFE: 전체 관절 경로가 안전 여유를 만족합니다.")
