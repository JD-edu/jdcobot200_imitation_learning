#!/usr/bin/env python3
"""GLFW 키보드로 stable MJCF의 arm과 gripper를 제어합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/02_mujoco_basics/04_manual_control.py
# GLFW 창에서 1~5, LEFT/RIGHT, H, O, C, ESC 키를 사용합니다.

from pathlib import Path
import time

import mujoco
from mujoco.glfw import glfw
import numpy as np


LESSON_DIR = Path(__file__).resolve().parent
SCENE_PATH = LESSON_DIR / "scene.xml"
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
STEP_RAD = np.deg2rad(5.0)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.45, -0.20


def main() -> None:
    if not (LESSON_DIR / "generated/jdcobot200_stable.xml").exists():
        raise SystemExit("먼저 01_urdf_to_mjcf.py와 02_add_dynamics.py를 실행하세요.")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    actuator_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in JOINT_NAMES
    }
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    selected = 0
    data.ctrl[:] = 0.0
    data.ctrl[gripper_id] = GRIPPER_OPEN

    print("1~5: 관절 선택, LEFT/RIGHT: 5도 이동")
    print("H: HOME, O: gripper open, C: gripper close, ESC: 종료")
    if not glfw.init():
        raise SystemExit("GLFW 초기화 실패: 데스크톱 화면에서 실행하세요.")
    window = glfw.create_window(900, 700, "JDCobot200 Manual Control", None, None)
    if window is None:
        glfw.terminate()
        raise SystemExit("GLFW 창을 만들 수 없습니다.")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.12, 0.0, 0.12]
    camera.distance, camera.azimuth, camera.elevation = 0.75, 145, -25
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=2000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    def show_target() -> None:
        name = JOINT_NAMES[selected]
        print(f"선택: {name}, 목표: {np.rad2deg(data.ctrl[actuator_ids[name]]):.1f} deg")

    def on_key(_window, key, _scancode, action, _mods):
        nonlocal selected
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif glfw.KEY_1 <= key <= glfw.KEY_5:
            selected = key - glfw.KEY_1
            show_target()
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            actuator_id = actuator_ids[JOINT_NAMES[selected]]
            direction = -1.0 if key == glfw.KEY_LEFT else 1.0
            lower, upper = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(data.ctrl[actuator_id] + direction * STEP_RAD, lower, upper)
            show_target()
        elif key == glfw.KEY_H:
            for actuator_id in actuator_ids.values():
                data.ctrl[actuator_id] = 0.0
            print("HOME")
        elif key == glfw.KEY_O:
            data.ctrl[gripper_id] = GRIPPER_OPEN
            print("gripper OPEN")
        elif key == glfw.KEY_C:
            data.ctrl[gripper_id] = GRIPPER_CLOSED
            print("gripper CLOSE")

    glfw.set_key_callback(window, on_key)
    show_target()
    try:
        while not glfw.window_should_close(window):
            frame_start = data.time
            while data.time - frame_start < 1.0 / 60.0:
                mujoco.mj_step(model, data)
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(0.001)
    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()
