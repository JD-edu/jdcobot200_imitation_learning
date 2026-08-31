#!/usr/bin/env python3
"""Replay 10 random safe JDCobot200 episodes in MuJoCo by default."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco.glfw import glfw


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "synthetic_dataset_safe"
GOAL_COLOR = "#2ca02c"
BLOCK_COLOR = "#d62728"
EE_COLOR = "#1f77b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="100개 합성 에피소드 중 무작위 에피소드를 시각화합니다."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "random_10_trajectories.png"
    )
    parser.add_argument("--show", action="store_true", help="그래프 창도 표시")
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument(
        "--replay", dest="replay", action="store_true",
        help="선택된 에피소드를 MuJoCo로 재생(기본 동작)",
    )
    replay_group.add_argument(
        "--plot-only", dest="replay", action="store_false",
        help="MuJoCo를 열지 않고 경로 PNG만 생성",
    )
    parser.set_defaults(replay=True)
    parser.add_argument(
        "--speed", type=float, default=2.0, help="MuJoCo 재생 배속"
    )
    return parser.parse_args()


def select_episodes(dataset_dir: Path, count: int, seed: int) -> list[Path]:
    files = sorted(dataset_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"에피소드가 없습니다: {dataset_dir}")
    if not 1 <= count <= len(files):
        raise ValueError(f"count는 1~{len(files)} 범위여야 합니다.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(files), size=count, replace=False))
    return [files[int(index)] for index in indices]


def load_goal(dataset_dir: Path) -> np.ndarray:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return np.asarray(manifest["goal_xy"], dtype=float)
    return np.array([0.22, 0.10])


def plot_episodes(files: list[Path], goal_xy: np.ndarray, output: Path) -> None:
    columns = min(5, len(files))
    rows = int(np.ceil(len(files) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.6 * columns, 3.35 * rows), squeeze=False,
        sharex=True, sharey=True,
    )
    all_xy: list[np.ndarray] = []
    for axis, episode_path in zip(axes.flat, files):
        with np.load(episode_path) as episode:
            ee = episode["ee_position"]
            block = episode["block_position"]
            all_xy.extend([ee[:, :2], block[:, :2]])
            axis.plot(ee[:, 0], ee[:, 1], color=EE_COLOR, lw=1.3,
                      alpha=0.85, label="end effector")
            axis.plot(block[:, 0], block[:, 1], color=BLOCK_COLOR, lw=2.0,
                      label="red block")
            axis.scatter(block[0, 0], block[0, 1], color=BLOCK_COLOR,
                         marker="o", s=38, zorder=4, label="block start")
            axis.scatter(goal_xy[0], goal_xy[1], color=GOAL_COLOR,
                         marker="*", s=110, zorder=5, label="fixed goal")
            error_mm = np.linalg.norm(block[-1, :2] - goal_xy) * 1000
            axis.set_title(f"{episode_path.stem}  |  error {error_mm:.1f} mm")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
            axis.set_xlabel("world X (m)")
            axis.set_ylabel("world Y (m)")

    for axis in axes.flat[len(files):]:
        axis.set_visible(False)
    combined = np.concatenate(all_xy + [goal_xy[None, :]], axis=0)
    padding = 0.025
    for axis in axes.flat[:len(files)]:
        axis.set_xlim(combined[:, 0].min() - padding, combined[:, 0].max() + padding)
        axis.set_ylim(combined[:, 1].min() - padding, combined[:, 1].max() + padding)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(
        "Randomly selected JDCobot200 pick-and-place trajectories", y=0.995
    )
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=4, frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"시각화 저장: {output.resolve()}")


def replay_episodes(files: list[Path], xml_path: Path, speed: float) -> None:
    if speed <= 0:
        raise ValueError("speed는 0보다 커야 합니다.")
    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    data = mujoco.MjData(model)
    if not glfw.init():
        raise RuntimeError("GLFW 초기화에 실패했습니다. 그래픽 화면을 확인하세요.")
    window = glfw.create_window(1200, 900, "JDCobot200 trajectory replay", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("MuJoCo 재생 창을 만들 수 없습니다.")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    camera.azimuth = 150
    camera.elevation = -25
    camera.distance = 0.9
    camera.lookat[:] = [0.15, 0.0, 0.15]

    def on_key(window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, on_key)
    try:
        for number, episode_path in enumerate(files, start=1):
            if glfw.window_should_close(window):
                break
            print(f"재생 [{number}/{len(files)}]: {episode_path.name}")
            with np.load(episode_path) as episode:
                qpos = episode["qpos"]
                timestamps = episode["timestamp"]
                for frame in range(len(timestamps)):
                    if glfw.window_should_close(window):
                        break
                    tick = time.perf_counter()
                    data.qpos[:] = qpos[frame]
                    data.time = float(timestamps[frame])
                    mujoco.mj_forward(model, data)
                    width, height = glfw.get_framebuffer_size(window)
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    mujoco.mjv_updateScene(
                        model, data, option, None, camera,
                        mujoco.mjtCatBit.mjCAT_ALL.value, scene,
                    )
                    mujoco.mjr_render(viewport, scene, context)
                    title = f"EPISODE {number} / {len(files)}"
                    detail = (
                        f"{episode_path.stem}   |   frame {frame + 1} / {len(qpos)}"
                        f"   |   speed {speed:g}x\nESC: close"
                    )
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        viewport,
                        title,
                        detail,
                        context,
                    )
                    glfw.set_window_title(
                        window, f"JDCobot200 | Episode {number}/{len(files)}"
                    )
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    if frame + 1 < len(timestamps):
                        frame_dt = timestamps[frame + 1] - timestamps[frame]
                        time.sleep(max(0.0, frame_dt / speed - (time.perf_counter() - tick)))
            transition_until = time.perf_counter() + 0.5
            while (
                time.perf_counter() < transition_until
                and not glfw.window_should_close(window)
            ):
                glfw.poll_events()
                time.sleep(0.01)
    finally:
        context.free()
        scene.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    args = parse_args()
    files = select_episodes(args.dataset_dir, args.count, args.seed)
    print("선택된 에피소드:", ", ".join(path.stem for path in files))
    goal_xy = load_goal(args.dataset_dir)
    plot_episodes(files, goal_xy, args.output)
    if args.show:
        plt.show()
    else:
        plt.close("all")
    if args.replay:
        replay_episodes(files, ROOT / "scene.xml", args.speed)


if __name__ == "__main__":
    main()
