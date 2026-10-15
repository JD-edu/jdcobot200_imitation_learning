#!/usr/bin/env python3
"""안전 검사에 통과한 관절 명령만 GLFW 시뮬레이션에 적용합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/04_safe_manual_control.py
# GLFW 창에서 1~5, LEFT/RIGHT, ESC 키를 사용합니다.

import time
import mujoco
from mujoco.glfw import glfw
import numpy as np

from fk_safety import FKSafety, JOINT_NAMES, SCENE_PATH


safety = FKSafety()
# 화면의 물리 상태와 FK 검사 상태를 분리한다.
model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)
actuator_ids = [model.actuator(name).id for name in JOINT_NAMES]
selected = 0
data.ctrl[:] = 0.0

if not glfw.init():
    raise SystemExit("GLFW 초기화 실패")
window = glfw.create_window(900, 700, "Safe FK Control", None, None)
if window is None:
    glfw.terminate()
    raise SystemExit("GLFW 창 생성 실패")
glfw.make_context_current(window)
camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = [0.12, 0, 0.12], 0.75, 145, -25
scene = mujoco.MjvScene(model, maxgeom=2000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
print("1~5: 관절 선택, LEFT/RIGHT: 안전할 때만 5도 이동, ESC: 종료")


def on_key(_window, key, _scancode, action, _mods):
    global selected
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif glfw.KEY_1 <= key <= glfw.KEY_5:
        selected = key - glfw.KEY_1
        print("선택:", JOINT_NAMES[selected])
    elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
        current = data.ctrl[actuator_ids].copy()
        goal = current.copy()
        goal[selected] += np.deg2rad(-5 if key == glfw.KEY_LEFT else 5)
        safe, _, result = safety.check_path(current, goal, 21)
        if safe:
            data.ctrl[actuator_ids] = goal
            print("허용:", JOINT_NAMES[selected], f"{np.rad2deg(goal[selected]):.1f} deg")
        else:
            print("거부:", ", ".join(result.reasons),
                  f"self={result.self_clearance*1000:.1f}mm",
                  f"floor={result.floor_clearance*1000:.1f}mm")


glfw.set_key_callback(window, on_key)
try:
    while not glfw.window_should_close(window):
        start = data.time
        while data.time - start < 1 / 60:
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
