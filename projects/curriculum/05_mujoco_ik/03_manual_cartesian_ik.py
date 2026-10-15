#!/usr/bin/env python3
"""키보드로 XYZ 목표를 바꾸며 MuJoCo IK의 도달 영역과 한계를 탐색합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/03_manual_cartesian_ik.py
# 기본 시작 자세(deg): 0 20 -40 20 0 (관절 여유가 있는 READY 자세)
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/03_manual_cartesian_ik.py \
#   --start-q-deg 0 20 -40 20 0 --step-mm 15 --move-time 0.8
#
# MuJoCo 창 키:
# X/Y/Z       이동축 선택
# LEFT/RIGHT  선택축의 음/양 방향으로 TCP 이동
# ESC         종료

from __future__ import annotations

import argparse
import sys
import time

import mujoco
from mujoco.glfw import glfw
import numpy as np

from ik_solver import JOINT_NAMES, SafeIK


def smoothstep(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


class CameraController:
    """MuJoCo 표준 마우스 회전·이동·확대/축소 제어."""

    def __init__(self, model, scene, camera) -> None:
        self.model, self.scene, self.camera = model, scene, camera
        self.buttons = {
            glfw.MOUSE_BUTTON_LEFT: False,
            glfw.MOUSE_BUTTON_MIDDLE: False,
            glfw.MOUSE_BUTTON_RIGHT: False,
        }
        self.last_x = self.last_y = 0.0

    def mouse_button(self, window, _button, _action, _mods) -> None:
        for button in self.buttons:
            self.buttons[button] = glfw.get_mouse_button(window, button) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def cursor_position(self, window, xpos, ypos) -> None:
        if not any(self.buttons.values()):
            return
        dx, dy = xpos - self.last_x, ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos
        _, height = glfw.get_window_size(window)
        if height <= 0:
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self.buttons[glfw.MOUSE_BUTTON_RIGHT]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.buttons[glfw.MOUSE_BUTTON_LEFT]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.camera
        )

    def scroll(self, _window, _xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0, -0.05 * yoffset, self.scene, self.camera,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="X/Y/Z와 LEFT/RIGHT 키로 MuJoCo Cartesian IK 한계를 탐색합니다."
    )
    parser.add_argument(
        "--start-q-deg", nargs=5, type=float,
        default=[0, 20, -40, 20, 0],
        help="시작 관절각 5개(기본: 0 20 -40 20 0 deg READY 자세)",
    )
    parser.add_argument(
        "--step-mm", type=float, default=10.0,
        help="키 1회당 TCP 이동거리(mm, 기본 10, 허용 5~20)",
    )
    parser.add_argument("--move-time", type=float, default=0.8, help="허용된 IK 자세의 이동시간")
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--path-steps", type=int, default=101)
    args = parser.parse_args()
    if not 5 <= args.step_mm <= 20:
        raise ValueError("step-mm는 5 이상 20 이하로 지정해야 합니다.")
    if args.move_time <= 0 or args.tolerance_mm <= 0:
        raise ValueError("move-time과 tolerance-mm는 양수여야 합니다.")
    if args.path_steps < 2:
        raise ValueError("path-steps는 2 이상이어야 합니다.")

    solver = SafeIK()
    model = solver.model
    data = mujoco.MjData(model)
    qpos_addresses = solver.fk.qpos_addresses
    start_q = np.deg2rad(np.asarray(args.start_q_deg, dtype=float))
    start_result = solver.fk.evaluate(start_q)
    if not start_result.safe:
        raise RuntimeError(f"시작 자세가 안전하지 않습니다: {start_result.reasons}")

    data.qpos[qpos_addresses] = start_q
    mujoco.mj_forward(model, data)

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패: 데스크톱 DISPLAY에서 실행하세요.")
    window = glfw.create_window(900, 700, "Manual Cartesian IK", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW 창 생성 실패")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    camera.lookat[:], camera.distance = [0.12, 0.0, 0.20], 0.75
    camera.azimuth, camera.elevation = 145.0, -25.0
    scene = mujoco.MjvScene(model, maxgeom=10_000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera_controller = CameraController(model, scene, camera)

    state = {
        "axis": 0,
        "request": None,
        "status": "READY",
        "moving": False,
        "move_start": 0.0,
        "start_q": start_q.copy(),
        "goal_q": start_q.copy(),
    }

    def on_key(_window, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_window, True)
        elif key in (glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z):
            state["axis"] = {glfw.KEY_X: 0, glfw.KEY_Y: 1, glfw.KEY_Z: 2}[key]
            state["status"] = f"SELECT {'XYZ'[state['axis']]}"
            print("선택축:", "XYZ"[state["axis"]])
        elif key == glfw.KEY_LEFT:
            state["request"] = -1
        elif key == glfw.KEY_RIGHT:
            state["request"] = 1

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, camera_controller.mouse_button)
    glfw.set_cursor_pos_callback(window, camera_controller.cursor_position)
    glfw.set_scroll_callback(window, camera_controller.scroll)
    print(f"X/Y/Z: 축 선택, LEFT/RIGHT: {args.step_mm:g} mm 이동, ESC: 종료")
    print("마우스: 좌클릭 드래그=회전, 우클릭 드래그=이동, 휠/중클릭=확대·축소")
    print("시작 TCP(m):", np.round(start_result.tcp_position, 6))

    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            request = state["request"]
            state["request"] = None

            if request is not None and not state["moving"]:
                current_q = data.qpos[qpos_addresses].copy()
                current_result = solver.fk.evaluate(current_q)
                axis = int(state["axis"])
                target_xyz = current_result.tcp_position.copy()
                target_xyz[axis] += int(request) * args.step_mm / 1000.0
                axis_label = "XYZ"[axis]
                direction_label = "+" if request > 0 else "-"
                try:
                    ik = solver.solve(
                        current_q, target_xyz,
                        tolerance_m=args.tolerance_mm / 1000.0,
                    )
                    if not ik.converged:
                        raise RuntimeError(f"IK 미수렴, FK 오차={ik.error_m*1000:.2f} mm")
                    path = solver.check_path(
                        current_q, ik.q_rad, steps=args.path_steps,
                    )
                    if not path.safe:
                        raise RuntimeError(
                            f"unsafe step={path.unsafe_step}, reason={path.unsafe_result.reasons}"
                        )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    state["status"] = f"REJECT {axis_label}{direction_label}"
                    print(
                        f"거부 {axis_label}{direction_label}{args.step_mm:g}mm: {exc}; "
                        f"현재 TCP={np.round(current_result.tcp_position, 5)}"
                    )
                else:
                    state["start_q"] = current_q
                    state["goal_q"] = ik.q_rad
                    state["move_start"] = now
                    state["moving"] = True
                    state["status"] = f"MOVE {axis_label}{direction_label}"
                    print(
                        f"허용 {axis_label}{direction_label}{args.step_mm:g}mm: "
                        f"target={np.round(target_xyz, 5)}, "
                        f"FK error={ik.error_m*1000:.3f}mm, "
                        f"q={np.round(np.rad2deg(ik.q_rad), 2)}"
                    )

            if state["moving"]:
                progress = min(1.0, (now - float(state["move_start"])) / args.move_time)
                alpha = smoothstep(progress)
                command_q = (
                    (1.0 - alpha) * state["start_q"] + alpha * state["goal_q"]
                )
                # 이 도구의 목적은 IK 도달영역 탐색입니다. 물리 actuator로 적분하면
                # 중력과 제한된 torque 때문에 명령 자세에서 처지고, unsafe 시작점에
                # 갇힐 수 있으므로 검증된 관절 경로를 정확히 재생합니다.
                data.qpos[qpos_addresses] = command_q
                if progress >= 1.0:
                    state["moving"] = False
                    state["status"] = "READY"

            mujoco.mj_forward(model, data)

            actual_q = data.qpos[qpos_addresses].copy()
            actual_tcp = data.site_xpos[solver.site_id].copy()
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                model, data, option, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            glfw.set_window_title(
                window,
                f"Cartesian IK | axis={'XYZ'[state['axis']]} | step={args.step_mm:g}mm | "
                f"TCP={np.round(actual_tcp, 3)} | {state['status']} | ESC: close",
            )
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(0.001)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
