#!/usr/bin/env python3
"""키보드로 실제 로봇과 MuJoCo의 Cartesian IK를 안전하게 동기 제어합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/06_sim2real_ik/01_manual_cartesian_ik.py
# /usr/bin/python3 projects/curriculum/06_sim2real_ik/01_manual_cartesian_ik.py \
#   --step-mm 2 --move-time 1.5 --max-joint-change-deg 5
# MuJoCo 창 키: X/Y/Z=축 선택, LEFT/RIGHT=음/양 이동, ESC=종료

import argparse
from pathlib import Path
import sys
import time

from mujoco.glfw import glfw
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COURSE04 = PROJECT_ROOT / "projects/curriculum/04_sim2real_fk"
COURSE05 = PROJECT_ROOT / "projects/curriculum/05_mujoco_ik"
for path in (PROJECT_ROOT, COURSE04, COURSE05):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ik_solver import SafeIK  # noqa: E402
from sim2real_control import JOINT_NAMES, MuJoCoViewer, SynchronizedController  # noqa: E402


START_LIMIT_RECOVERY_MARGIN_DEG = 0.5
ELBOW_IK_LOWER_DEG = -110.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="키보드로 실제 로봇과 MuJoCo를 Cartesian IK 동기 제어합니다."
    )
    parser.add_argument("--step-mm", type=float, default=2.0,
                        help="키 1회당 TCP 이동거리(mm, 기본 2, 허용 0.5~10)")
    parser.add_argument("--move-time", type=float, default=1.5)
    parser.add_argument("--ready-move-time", type=float, default=3.0,
                        help="시작 READY 자세까지의 이동시간(초, 기본 3)")
    parser.add_argument("--max-joint-change-deg", type=float, default=5.0)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument(
        "--feedback-tolerance-ticks", type=int, default=50,
        help="실제 서보 목표 도달 허용오차(encoder tick, 기본 50, 허용 1~100)",
    )
    parser.add_argument("--path-steps", type=int, default=101)
    parser.add_argument("--port")
    args = parser.parse_args()
    if not 0.5 <= args.step_mm <= 10.0:
        raise ValueError("step-mm는 0.5 이상 10 이하로 지정해야 합니다.")
    if (args.move_time <= 0 or args.ready_move_time <= 0
            or args.max_joint_change_deg <= 0 or args.tolerance_mm <= 0):
        raise ValueError("이동 시간, READY 이동 시간, 관절 변화 제한과 tolerance는 양수여야 합니다.")
    if not 1 <= args.feedback_tolerance_ticks <= 100:
        raise ValueError("feedback-tolerance-ticks는 1 이상 100 이하로 지정해야 합니다.")
    if args.path_steps < 2:
        raise ValueError("path-steps는 2 이상이어야 합니다.")

    with SynchronizedController(port=args.port) as controller:
        # 공식 MJCF의 elbow actuator 하한(-96.83도)은 실제 encoder 안전범위보다
        # 좁어 낮은 Z 위치의 IK를 불필요하게 막는다. 이 과정에서만 하드웨어
        # 끝단보다 여유를 둔 -110도까지 사용한다.
        elbow_index = JOINT_NAMES.index("elbow")
        elbow_joint = controller.config.joint("elbow")
        elbow_hardware_angles = (
            controller.robot.raw_to_radian(elbow_joint, elbow_joint.min_tick),
            controller.robot.raw_to_radian(elbow_joint, elbow_joint.max_tick),
        )
        elbow_hardware_lower = min(elbow_hardware_angles)
        requested_elbow_lower = np.deg2rad(ELBOW_IK_LOWER_DEG)
        if requested_elbow_lower < elbow_hardware_lower:
            raise RuntimeError(
                f"IK elbow 하한 {ELBOW_IK_LOWER_DEG:g}°가 하드웨어 하한 "
                f"{np.rad2deg(elbow_hardware_lower):.3f}°보다 작습니다."
            )
        controller.safety.lower[elbow_index] = requested_elbow_lower
        solver = SafeIK(controller.safety)
        feedback_tolerance_ticks = args.feedback_tolerance_ticks
        feedback_tolerance_deg = (
            feedback_tolerance_ticks * 360.0
            / controller.config.protocol_resolution
        )
        # 시작 시 한 번의 일시적인 encoder 이상값 때문에 종료되지 않도록 다시
        # 측정하되, 지속적인 limit 초과는 우회하지 않고 자세히 보고한다.
        for attempt in range(1, 4):
            current_raw, current_q = controller.read(samples=5)
            initial = controller.safety.evaluate(current_q)
            if initial.safe:
                break
            if attempt < 3:
                print(
                    f"경고: 시작 encoder 안전검사 실패({attempt}/3): "
                    f"{initial.reasons}. 다시 측정합니다."
                )
                time.sleep(0.1)
        if not initial.safe:
            q_deg = np.rad2deg(current_q)
            lower_deg = np.rad2deg(controller.safety.lower)
            upper_deg = np.rad2deg(controller.safety.upper)
            violations = [
                (name, angle, lower, upper)
                for name, angle, lower, upper
                in zip(JOINT_NAMES, q_deg, lower_deg, upper_deg)
                if angle < lower or angle > upper
            ]
            max_limit_excess_deg = max(
                (max(lower - angle, angle - upper, 0.0)
                 for _name, angle, lower, upper in violations),
                default=0.0,
            )
            if (initial.reasons == ("joint_limit",)
                    and max_limit_excess_deg <= START_LIMIT_RECOVERY_MARGIN_DEG):
                print(
                    "경고: encoder 자세가 MuJoCo 관절 제한을 최대 "
                    f"{max_limit_excess_deg:.3f}° 초과했습니다. "
                    "0.5° 이내의 경계 오차이므로 제한값으로 보정하여 "
                    "READY 복귀 경로를 검사합니다."
                )
                current_q = np.clip(
                    current_q, controller.safety.lower, controller.safety.upper
                )
                initial = controller.safety.evaluate(current_q)
            else:
                limit_detail = ", ".join(
                    f"{name}={angle:.3f}° (허용 {lower:.3f}~{upper:.3f}°)"
                    for name, angle, lower, upper in violations
                )
                if limit_detail:
                    limit_detail = f", limit 초과: {limit_detail}"
                raise RuntimeError(
                    f"현재 encoder 자세가 안전하지 않습니다: {initial.reasons}, "
                    f"q(deg)={np.round(q_deg, 3)}, raw={current_raw}{limit_detail}"
                )

        # 원점은 Cartesian IK의 특이 자세이므로 관절 여유가 있는 설정 파일의
        # READY 자세로 먼저 이동한다.
        ready_q = np.deg2rad(np.asarray(controller.config.ready_pose_degrees, dtype=float))
        ready_result = controller.safety.evaluate(ready_q)
        if not ready_result.safe:
            raise RuntimeError(f"READY 자세가 안전하지 않습니다: {ready_result.reasons}")
        controller.check_path(current_q, ready_q, steps=args.path_steps)
        controller.q_to_ticks(ready_q)

        def move_and_resync(start_raw, start_q, goal_q, *, duration, on_step):
            """이동 후 작은 추종 오차가 남으면 실제 encoder 자세로 재동기화한다."""
            try:
                return controller.move_to(
                    start_raw, start_q, goal_q,
                    duration=duration,
                    feedback_tolerance_ticks=feedback_tolerance_ticks,
                    on_step=on_step,
                )
            except TimeoutError as exc:
                measured_raw, measured_q = controller.read(samples=5)
                measured_result = controller.safety.evaluate(measured_q)
                if not measured_result.safe:
                    raise RuntimeError(
                        "타임아웃 후 encoder 자세가 안전하지 않습니다: "
                        f"{measured_result.reasons}"
                    ) from exc
                error_deg = np.rad2deg(goal_q - measured_q)
                if np.max(np.abs(error_deg)) > args.max_joint_change_deg:
                    raise RuntimeError(
                        "타임아웃 후 목표와 실제 자세 차이가 너무 큽니다: "
                        f"{np.round(error_deg, 3)} deg"
                    ) from exc
                print(
                    "경고: 목표 허용오차 내에 도달하지 못했습니다. "
                    "실제 encoder 자세로 다시 동기화합니다.",
                    f"관절 오차(deg)={np.round(error_deg, 3)}",
                )
                return measured_raw, measured_q

        print(f"X/Y/Z: 축 선택, LEFT/RIGHT: 안전할 때만 {args.step_mm:g} mm 이동")
        print("각 키 입력 전에 목표 자세와 전체 경로의 FK safety를 검사합니다.")
        print(
            f"서보 도착 허용오차: {feedback_tolerance_ticks} encoder tick "
            f"({feedback_tolerance_deg:.3f}°)"
        )
        print(
            f"IK elbow 하한: {ELBOW_IK_LOWER_DEG:g}° "
            f"(하드웨어 하한 {np.rad2deg(elbow_hardware_lower):.3f}°)"
        )
        print("현재 q(deg):", np.round(np.rad2deg(current_q), 3))
        print("이동할 READY q(deg):", np.round(np.rad2deg(ready_q), 3))
        if input("READY 자세 이동과 수동 IK 제어를 시작하려면 MANUAL 입력: ").strip() != "MANUAL":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        viewer = MuJoCoViewer(controller.safety, "Sim2Real Cartesian IK")
        state = {"axis": 0, "request": None, "moving": False, "status": "READY"}

        def on_key(window, key, _scancode, action, _mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key in (glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z):
                state["axis"] = {glfw.KEY_X: 0, glfw.KEY_Y: 1, glfw.KEY_Z: 2}[key]
                state["status"] = f"SELECT {'XYZ'[state['axis']]}"
                print("선택축:", "XYZ"[state["axis"]])
            elif not state["moving"] and key == glfw.KEY_LEFT:
                state["request"] = -1
            elif not state["moving"] and key == glfw.KEY_RIGHT:
                state["request"] = 1

        viewer.set_key_callback(on_key)
        try:
            print(f"READY 자세로 이동합니다 ({args.ready_move_time:g}초).")
            current_raw, current_q = move_and_resync(
                current_raw, current_q, ready_q, duration=args.ready_move_time,
                on_step=lambda q: viewer.render(q, "Moving to READY pose"),
            )
            ready_measured = controller.safety.evaluate(current_q)
            print("IK 시작 q(deg):", np.round(np.rad2deg(current_q), 3))
            print("IK 시작 TCP(m):", np.round(ready_measured.tcp_position, 6))

            while not viewer.should_close():
                axis = int(state["axis"])
                title = (f"Sim2Real SAFE Cartesian IK | axis={'XYZ'[axis]} | "
                         f"step={args.step_mm:g}mm | {state['status']} | ESC: close")
                viewer.render(current_q, title)
                request = state["request"]
                state["request"] = None
                if request is None:
                    time.sleep(0.01)
                    continue

                current_fk = controller.safety.evaluate(current_q)
                if not current_fk.safe:
                    raise RuntimeError(f"현재 encoder 자세가 안전하지 않습니다: {current_fk.reasons}")
                target_xyz = current_fk.tcp_position.copy()
                target_xyz[axis] += int(request) * args.step_mm / 1000.0
                label = f"{'XYZ'[axis]}{'+' if request > 0 else '-'}"

                try:
                    ik = solver.solve(current_q, target_xyz,
                                      tolerance_m=args.tolerance_mm / 1000.0)
                    if not ik.converged:
                        raise RuntimeError(f"IK 미수렴: FK 오차 {ik.error_m * 1000:.2f} mm")

                    # 실제 이동 전에 목표 자세의 FK safety를 먼저 검사합니다.
                    goal_fk = controller.safety.evaluate(ik.q_rad)
                    if not goal_fk.safe:
                        raise RuntimeError(f"목표 FK unsafe: {goal_fk.reasons}")
                    change_deg = np.rad2deg(ik.q_rad - current_q)
                    if np.max(np.abs(change_deg)) > args.max_joint_change_deg:
                        raise RuntimeError(f"관절 변화 제한 초과: {np.round(change_deg, 3)} deg")
                    path = solver.check_path(current_q, ik.q_rad, steps=args.path_steps)
                    if not path.safe:
                        raise RuntimeError(f"경로 FK unsafe: step={path.unsafe_step}, "
                                           f"reason={path.unsafe_result.reasons}")
                    controller.q_to_ticks(ik.q_rad)
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    state["status"] = f"REJECT {label}"
                    print(f"거부 {label} {args.step_mm:g}mm: {exc}")
                    continue

                print(f"허용 {label} {args.step_mm:g}mm: target={np.round(target_xyz, 5)}, "
                      f"FK error={ik.error_m * 1000:.3f}mm, "
                      f"q={np.round(np.rad2deg(ik.q_rad), 2)}")
                state["moving"] = True
                state["status"] = f"MOVE {label}"
                current_raw, current_q = move_and_resync(
                    current_raw, current_q, ik.q_rad, duration=args.move_time,
                    on_step=lambda q: viewer.render(q, title),
                )
                state["moving"] = False
                state["status"] = "READY"
        finally:
            viewer.close()

        print("수동 IK 제어를 종료했습니다. 자세 유지를 위해 arm torque는 ON 상태입니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
