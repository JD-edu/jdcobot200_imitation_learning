#!/usr/bin/env python3
"""1~5와 LEFT/RIGHT 키로 실제 로봇과 MuJoCo를 안전하게 동기 제어합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/03_sync_manual_control.py
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/03_sync_manual_control.py --step-deg 5 --move-time 1

import argparse
import sys
import time

from mujoco.glfw import glfw
import numpy as np

from sim2real_control import JOINT_NAMES, MuJoCoViewer, SynchronizedController


def main() -> int:
    parser = argparse.ArgumentParser(description="실제/MuJoCo 관절 키보드 동기 제어")
    parser.add_argument("--step-deg", type=float, default=5.0, help="LEFT/RIGHT 1회 이동각(기본 5도)")
    parser.add_argument("--move-time", type=float, default=1.0, help="키 1회 이동 시간")
    parser.add_argument("--port")
    args = parser.parse_args()
    if not 0 < args.step_deg <= 5:
        raise ValueError("--step-deg는 0보다 크고 5도 이하여야 합니다.")
    if args.move_time <= 0:
        raise ValueError("--move-time은 양수여야 합니다.")

    with SynchronizedController(port=args.port) as controller:
        current_raw, current_q = controller.read(samples=5)
        initial = controller.safety.evaluate(current_q)
        if not initial.safe:
            raise RuntimeError(f"현재 자세가 안전하지 않습니다: {initial.reasons}")
        print(f"1~5: 관절 선택, LEFT/RIGHT: 안전할 때만 {args.step_deg:g}도 이동, ESC: 종료")
        print("관절:", ", ".join(f"{i+1}={name}" for i, name in enumerate(JOINT_NAMES)))
        print("각 키 입력마다 전체 경로를 검사한 후에만 실제 명령을 보냅니다.")
        if input("동기 수동 제어를 시작하려면 MANUAL 입력: ").strip() != "MANUAL":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        viewer = MuJoCoViewer(controller.safety, "Sim2Real manual control")
        state = {"selected": 0, "delta": None}

        def on_key(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif glfw.KEY_1 <= key <= glfw.KEY_5:
                state["selected"] = key - glfw.KEY_1
                print("선택:", JOINT_NAMES[state["selected"]])
            elif key == glfw.KEY_LEFT:
                state["delta"] = -np.deg2rad(args.step_deg)
            elif key == glfw.KEY_RIGHT:
                state["delta"] = np.deg2rad(args.step_deg)

        viewer.set_key_callback(on_key)
        try:
            while not viewer.should_close():
                selected = int(state["selected"])
                title = (
                    f"Sim2Real SAFE manual | selected {selected+1}: "
                    f"{JOINT_NAMES[selected]} | LEFT/RIGHT {args.step_deg:g} deg"
                )
                viewer.render(current_q, title)
                delta = state["delta"]
                state["delta"] = None
                if delta is None:
                    time.sleep(0.01)
                    continue

                goal_q = current_q.copy()
                goal_q[selected] += float(delta)
                # 과목 03과 동일하게 목표까지 21개 자세를 먼저 검사합니다.
                safe, _, result = controller.safety.check_path(current_q, goal_q, 21)
                if not safe:
                    print(
                        "거부:", ", ".join(result.reasons),
                        f"self={result.self_clearance*1000:.1f}mm",
                        f"floor={result.floor_clearance*1000:.1f}mm",
                    )
                    continue
                try:
                    controller.q_to_ticks(goal_q)
                except ValueError as exc:
                    print("거부: hardware_limit", exc)
                    continue
                print(
                    "허용:", JOINT_NAMES[selected],
                    f"{np.rad2deg(goal_q[selected]):.1f} deg",
                )
                try:
                    current_raw, current_q = controller.move_to(
                        current_raw, current_q, goal_q, duration=args.move_time,
                        on_step=lambda q: viewer.render(q, title),
                    )
                except TimeoutError as exc:
                    # 작은 정상상태 오차나 순간적인 부하 때문에 목표 도달 판정이
                    # 실패해도 수동 제어 전체를 종료하지 않는다. 다음 이동은 반드시
                    # 다시 읽은 실제 자세를 시작점으로 삼아 경로를 검사한다.
                    current_raw, current_q = controller.read(samples=5)
                    measured_result = controller.safety.evaluate(current_q)
                    if not measured_result.safe:
                        raise RuntimeError(
                            "타임아웃 후 encoder 자세가 안전하지 않습니다: "
                            f"{measured_result.reasons}"
                        ) from exc
                    print(
                        "경고: 목표 허용오차 내에 도달하지 못했습니다. "
                        "실제 encoder 자세로 다시 동기화합니다.",
                        exc,
                    )
        finally:
            viewer.close()
        print("수동 제어를 종료했습니다. 자세 유지를 위해 arm torque는 ON 상태입니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
