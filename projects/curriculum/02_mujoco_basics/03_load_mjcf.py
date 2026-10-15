#!/usr/bin/env python3
"""raw/stable/canonical MJCF를 GLFW 창에 표시합니다."""

# 실행 예(인자 하나를 선택):
# /usr/bin/python3 projects/curriculum/02_mujoco_basics/03_load_mjcf.py raw
# /usr/bin/python3 projects/curriculum/02_mujoco_basics/03_load_mjcf.py stable
# /usr/bin/python3 projects/curriculum/02_mujoco_basics/03_load_mjcf.py canonical

import argparse
from pathlib import Path
import time

import mujoco
from mujoco.glfw import glfw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LESSON_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "raw": LESSON_DIR / "generated/jdcobot200_raw.xml",
    "stable": LESSON_DIR / "scene.xml",
    "canonical": PROJECT_ROOT / "robot_description/mjcf/scenes/empty.xml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_PATHS)
    args = parser.parse_args()
    model_path = MODEL_PATHS[args.model]
    if not model_path.exists():
        raise SystemExit(f"모델이 없습니다: {model_path}\n먼저 01과 02를 실행하세요.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    print(f"로드: {args.model} ({model_path})")
    print(f"자유도: {model.nv}, actuator: {model.nu}, ESC: 종료")

    if not glfw.init():
        raise SystemExit("GLFW 초기화 실패: 데스크톱 화면에서 실행하세요.")
    window = glfw.create_window(900, 700, f"JDCobot200 - {args.model}", None, None)
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

    def on_key(_window, key, _scancode, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, on_key)
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
