#!/usr/bin/env python3
"""관절각 5개로 TCP의 위치와 방향을 계산합니다."""

# 실행 예(base shoulder elbow wrist_pitch wrist_roll, 단위 degree):
# /usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/02_fk_pose.py 0 20 -40 20 0

import argparse
import numpy as np

from fk_safety import FKSafety, JOINT_NAMES


parser = argparse.ArgumentParser()
parser.add_argument("q_deg", nargs=5, type=float, metavar=("J1", "J2", "J3", "J4", "J5"))
args = parser.parse_args()

fk = FKSafety()
result = fk.evaluate(np.deg2rad(args.q_deg))
print("관절 순서:", ", ".join(JOINT_NAMES))
print("관절각(deg):", np.round(args.q_deg, 3))
print("TCP 위치(m):", np.round(result.tcp_position, 6))
print("TCP 회전행렬:\n", np.round(result.tcp_matrix, 6))
