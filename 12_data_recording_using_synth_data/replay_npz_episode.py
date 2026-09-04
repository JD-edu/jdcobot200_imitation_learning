#!/usr/bin/env python3
"""Replay a saved synthetic NPZ episode in the MuJoCo GLFW viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco.glfw import glfw

from generate_synthetic_trajectories import DEFAULT_XML, ROOT
from jdcobot200_mujoco_glfw import GLFWCameraController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        default=ROOT / "synthetic_dataset" / "episode_0000.npz",
        help="NPZ path (default: synthetic_dataset/episode_0000.npz)",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Validate and replay numerically without opening a window",
    )
    return parser.parse_args()


def load_episode(path: Path, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode does not exist: {path}")
    with np.load(path) as source:
        required = {"timestamp", "qpos"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"NPZ is missing fields: {sorted(missing)}")
        episode = {key: np.asarray(source[key]).copy() for key in source.files}

    timestamps = episode["timestamp"]
    qpos = episode["qpos"]
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("timestamp must be a non-empty 1-D array")
    if qpos.shape != (len(timestamps), model.nq):
        raise ValueError(
            f"qpos shape must be ({len(timestamps)}, {model.nq}), got {qpos.shape}"
        )
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(qpos)):
        raise ValueError("timestamp/qpos contains NaN or Inf")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return episode


def set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    episode: dict[str, np.ndarray],
    frame: int,
) -> None:
    data.qpos[:] = episode["qpos"][frame]
    if "qvel" in episode and episode["qvel"].shape == (len(episode["qpos"]), model.nv):
        data.qvel[:] = episode["qvel"][frame]
    else:
        data.qvel[:] = 0.0
    data.time = float(episode["timestamp"][frame])
    mujoco.mj_forward(model, data)


def run_headless(
    model: mujoco.MjModel, data: mujoco.MjData, episode: dict[str, np.ndarray]
) -> None:
    for frame in range(len(episode["timestamp"])):
        set_frame(model, data, episode, frame)
    print(f"Headless replay passed: {len(episode['timestamp'])} frames")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    episode: dict[str, np.ndarray],
    path: Path,
    speed: float,
    loop: bool,
    width: int,
    height: int,
) -> None:
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed (is a display available?)")
    window = glfw.create_window(width, height, f"NPZ Replay - {path.name}", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera = GLFWCameraController(model, scene)
    camera.camera.azimuth = 150.0
    camera.camera.elevation = -25.0
    camera.camera.distance = 0.9
    camera.camera.lookat[:] = [0.15, 0.0, 0.15]
    glfw.set_mouse_button_callback(window, camera.mouse_button)
    glfw.set_cursor_pos_callback(window, camera.cursor_position)
    glfw.set_scroll_callback(window, camera.scroll)

    state = {"frame": 0, "paused": False, "restart": True}

    def key_callback(window, key, scancode, action, mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            state["paused"] = not state["paused"]
            state["restart"] = True
        elif key == glfw.KEY_R and action == glfw.PRESS:
            state["frame"] = 0
            state["paused"] = False
            state["restart"] = True
        elif key == glfw.KEY_RIGHT:
            state["paused"] = True
            state["frame"] = min(state["frame"] + 1, len(episode["timestamp"]) - 1)
            state["restart"] = True
        elif key == glfw.KEY_LEFT:
            state["paused"] = True
            state["frame"] = max(state["frame"] - 1, 0)
            state["restart"] = True

    glfw.set_key_callback(window, key_callback)
    timestamps = episode["timestamp"] - episode["timestamp"][0]
    wall_start = time.perf_counter()
    episode_start = 0.0

    try:
        while not glfw.window_should_close(window):
            if state["restart"]:
                episode_start = float(timestamps[state["frame"]])
                wall_start = time.perf_counter()
                state["restart"] = False

            if not state["paused"]:
                elapsed = (time.perf_counter() - wall_start) * speed + episode_start
                state["frame"] = int(np.searchsorted(timestamps, elapsed, side="right") - 1)
                if state["frame"] >= len(timestamps) - 1:
                    if loop:
                        state["frame"] = 0
                        state["restart"] = True
                    else:
                        state["frame"] = len(timestamps) - 1
                        state["paused"] = True

            frame = state["frame"]
            set_frame(model, data, episode, frame)
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(
                model, data, camera.option, None, camera.camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            phase_text = ""
            if "phase" in episode:
                phase_text = f" | phase {int(episode['phase'][frame])}"
            status = "PAUSED" if state["paused"] else f"PLAY {speed:g}x"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                f"{path.name}\n{status}",
                f"frame {frame + 1}/{len(timestamps)}{phase_text}\n"
                "Space pause | Left/Right step | R restart | Esc quit",
                context,
            )
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    args = parse_args()
    if args.speed <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("speed, width, and height must be positive")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    episode = load_episode(args.episode, model)
    duration = float(episode["timestamp"][-1] - episode["timestamp"][0])
    print(
        f"Loaded {args.episode}: {len(episode['timestamp'])} frames, "
        f"{duration:.2f} s, replay speed {args.speed:g}x"
    )
    if args.headless:
        run_headless(model, data, episode)
    else:
        run_viewer(
            model, data, episode, args.episode, args.speed, args.loop,
            args.width, args.height,
        )


if __name__ == "__main__":
    main()
