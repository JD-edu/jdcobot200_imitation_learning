#!/usr/bin/env python3
"""Run fixed MINK FK poses in a MuJoCo GLFW simulation.

Keys: Space=pause, R=restart, Esc=quit. Mouse: left=orbit,
right=pan, middle/wheel=zoom.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mujoco.glfw import glfw
import mujoco
import numpy as np

from jdcobot200_mink_fk_solver import SO101MinkFKSolver


XML_PATH = Path(__file__).resolve().parent / "scene.xml"
END_EFFECTOR_SITE = "graspframe"
ARM_POSES = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00],
    [0.30, -0.20, 0.40, -0.30, 0.00],
    [-0.45, 0.15, -0.30, 0.40, 0.10],
    [0.10, -0.55, 0.55, -0.20, -0.30],
], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="고정 관절 자세를 MuJoCo GLFW에서 순차 재생합니다."
    )
    parser.add_argument("--move-time", type=float, default=2.0,
                        help="자세 사이 이동 시간(초)")
    parser.add_argument("--hold-time", type=float, default=1.0,
                        help="각 자세 유지 시간(초)")
    parser.add_argument("--once", action="store_true",
                        help="한 사이클 재생 후 종료")
    return parser.parse_args()


def require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise ValueError(f"MuJoCo model does not contain '{name}'.")
    return int(object_id)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class CameraController:
    def __init__(self, model: mujoco.MjModel, scene: mujoco.MjvScene) -> None:
        self.model = model
        self.scene = scene
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.option)
        self.camera.lookat[:] = [0.0, 0.0, 0.25]
        self.camera.distance = 1.1
        self.camera.azimuth = 145.0
        self.camera.elevation = -22.0
        self.buttons = {button: False for button in (
            glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_MIDDLE,
            glfw.MOUSE_BUTTON_RIGHT
        )}
        self.last_x = self.last_y = 0.0

    def mouse_button(self, window, button, action, mods) -> None:
        del button, action, mods
        for mouse_button in self.buttons:
            self.buttons[mouse_button] = (
                glfw.get_mouse_button(window, mouse_button) == glfw.PRESS
            )
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def cursor_position(self, window, xpos, ypos) -> None:
        if not any(self.buttons.values()):
            return
        dx, dy = xpos - self.last_x, ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos
        _, height = glfw.get_window_size(window)
        if height <= 0:
            return
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self.buttons[glfw.MOUSE_BUTTON_RIGHT]:
            mouse_action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift
                            else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        elif self.buttons[glfw.MOUSE_BUTTON_LEFT]:
            mouse_action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift
                            else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        else:
            mouse_action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model, mouse_action, dx / height, dy / height,
            self.scene, self.camera
        )

    def scroll(self, window, xoffset, yoffset) -> None:
        del window, xoffset
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0,
            -0.05 * yoffset, self.scene, self.camera
        )


def main() -> int:
    args = parse_args()
    if args.move_time <= 0.0 or args.hold_time < 0.0:
        raise ValueError("--move-time은 양수, --hold-time은 0 이상이어야 합니다.")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    fk = SO101MinkFKSolver(model, END_EFFECTOR_SITE)
    joint_ids = np.array([
        require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in fk.ARM_JOINT_NAMES
    ])
    qpos_addresses = model.jnt_qposadr[joint_ids].astype(int)
    actuator_ids = np.array([
        require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in fk.ARM_JOINT_NAMES
    ])
    gripper_id = require_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor"
    )
    site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)
    poses = np.clip(
        ARM_POSES,
        model.actuator_ctrlrange[actuator_ids, 0],
        model.actuator_ctrlrange[actuator_ids, 1],
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.ctrl[actuator_ids] = data.qpos[qpos_addresses]
    data.ctrl[gripper_id] = 0.0

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패: DISPLAY 설정을 확인하세요.")
    window = glfw.create_window(1200, 900, "JDCobot200 MINK FK demo", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW 창 생성에 실패했습니다.")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mujoco.MjvScene(model, maxgeom=10_000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera = CameraController(model, scene)
    glfw.set_mouse_button_callback(window, camera.mouse_button)
    glfw.set_cursor_pos_callback(window, camera.cursor_position)
    glfw.set_scroll_callback(window, camera.scroll)

    state = {"paused": False, "restart": False}

    def key_callback(window, key, scancode, action, mods) -> None:
        del scancode, mods
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif action == glfw.PRESS and key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif action == glfw.PRESS and key == glfw.KEY_R:
            state["restart"] = True

    glfw.set_key_callback(window, key_callback)
    pose_index, cycles = 0, 0
    start_q = data.qpos[qpos_addresses].copy()
    phase_start = time.perf_counter()
    pause_start = 0.0
    holding = False
    print("GLFW FK 시뮬레이션 시작")
    print("조작: Space=일시정지, R=재시작, Esc=종료")

    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            if state["restart"]:
                pose_index, cycles = 0, 0
                start_q = data.qpos[qpos_addresses].copy()
                phase_start, holding = now, False
                state["restart"] = False

            if state["paused"]:
                if pause_start == 0.0:
                    pause_start = now
            else:
                if pause_start != 0.0:
                    phase_start += now - pause_start
                    pause_start = 0.0
                target_q = poses[pose_index]
                if not holding:
                    progress = (now - phase_start) / args.move_time
                    blend = smoothstep(progress)
                    data.ctrl[actuator_ids] = (1.0 - blend) * start_q + blend * target_q
                    if progress >= 1.0:
                        data.ctrl[actuator_ids] = target_q
                        holding, phase_start = True, now
                elif now - phase_start >= args.hold_time:
                    start_q = data.ctrl[actuator_ids].copy()
                    pose_index += 1
                    if pose_index == len(poses):
                        pose_index, cycles = 0, cycles + 1
                        if args.once:
                            break
                    holding, phase_start = False, now

                frame_end = data.time + 1.0 / 60.0
                while data.time < frame_end:
                    mujoco.mj_step(model, data)

            reference_qpos = data.qpos.copy()
            reference_qpos[qpos_addresses] = data.ctrl[actuator_ids]
            fk_result = fk.solve_step(reference_qpos)
            actual_xyz = data.site_xpos[site_id].copy()
            error_mm = np.linalg.norm(fk_result.position - actual_xyz) * 1000.0
            actual_q = data.qpos[qpos_addresses].copy()
            status = "PAUSED" if state["paused"] else ("HOLD" if holding else "MOVE")
            glfw.set_window_title(
                window, f"JDCobot200 FK | pose {pose_index + 1}/{len(poses)} | "
                f"{status} | TCP error {error_mm:.2f} mm"
            )

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                model, data, camera.option, None, camera.camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, scene
            )
            mujoco.mjr_render(viewport, scene, context)
            overlay = (
                f"Pose: {pose_index + 1}/{len(poses)} [{status}]\n"
                f"Target q: {np.array2string(data.ctrl[actuator_ids], precision=3)}\n"
                f"Actual q: {np.array2string(actual_q, precision=3)}\n"
                f"MINK FK xyz: {np.array2string(fk_result.position, precision=4)}\n"
                f"MuJoCo xyz: {np.array2string(actual_xyz, precision=4)}\n"
                f"TCP error: {error_mm:.2f} mm"
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport, overlay, "Space: pause | R: restart | Esc: quit", context
            )
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        context.free()
        scene.free()
        glfw.destroy_window(window)
        glfw.terminate()

    print(f"시뮬레이션 종료 (완료 사이클: {cycles})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise
