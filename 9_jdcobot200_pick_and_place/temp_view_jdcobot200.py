#!/usr/bin/env python3
"""Interactively tune the two fingertip collision pads in jdcobot200.xml."""

from __future__ import annotations

import math
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import mujoco
import mujoco.viewer


XML_PATH = Path(__file__).resolve().with_name("jdcobot200.xml")
PAD_NAMES = ("left_finger_pad", "right_finger_pad")
GRIPPER_OPEN = 0.45
GRIPPER_CLOSED = -0.40


def quat_to_euler_deg(quat: list[float]) -> tuple[float, float, float]:
    """Convert MuJoCo's wxyz quaternion to intrinsic XYZ Euler angles."""
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def euler_deg_to_quat(angles: list[float]) -> tuple[float, float, float, float]:
    """Convert intrinsic XYZ Euler angles to a MuJoCo wxyz quaternion."""
    roll, pitch, yaw = (math.radians(v) / 2 for v in angles)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def replace_geom_attribute(xml: str, geom_name: str, attribute: str, value: str) -> str:
    """Replace one attribute without reformatting the rest of the XML file."""
    pattern = re.compile(rf'(<geom\s+name="{re.escape(geom_name)}"[^>]*?/>)', re.DOTALL)
    match = pattern.search(xml)
    if not match:
        raise ValueError(f"Cannot find geom {geom_name!r}")
    tag = match.group(1)
    attr_pattern = re.compile(rf'({re.escape(attribute)}=")[^"]*(")')
    if not attr_pattern.search(tag):
        raise ValueError(f"Geom {geom_name!r} has no {attribute!r} attribute")
    new_tag = attr_pattern.sub(rf'\g<1>{value}\g<2>', tag, count=1)
    return xml[:match.start(1)] + new_tag + xml[match.end(1):]


class PadTuner:
    def __init__(self, root: tk.Tk, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.root, self.model, self.data = root, model, data
        self.pad_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in PAD_NAMES
        }
        if any(geom_id < 0 for geom_id in self.pad_ids.values()):
            raise ValueError("The left/right finger pad geoms were not found")
        self.gripper_qpos = {}
        for name in ("gripper_left", "gripper_right"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint {name!r} was not found")
            self.gripper_qpos[name] = int(model.jnt_qposadr[joint_id])
        self.initial = {
            name: (model.geom_pos[geom_id].copy(), model.geom_quat[geom_id].copy())
            for name, geom_id in self.pad_ids.items()
        }
        self.variables: dict[str, dict[str, list[tk.DoubleVar]]] = {}
        # Start in the pose actually used by jdcobot200_pick_and_place.py.
        self.gripper = tk.DoubleVar(value=GRIPPER_CLOSED)
        self.parallel_angle = tk.StringVar()
        self.status = tk.StringVar(value="Adjust the sliders; changes appear immediately.")
        self.closed = False
        root.title("JDCobot200 fingertip pad tuner")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        for column, name in enumerate(PAD_NAMES):
            geom_id = self.pad_ids[name]
            frame = ttk.LabelFrame(main, text=name, padding=8)
            frame.grid(row=0, column=column, padx=5, sticky="nsew")
            pos = self.model.geom_pos[geom_id].tolist()
            rot = list(quat_to_euler_deg(self.model.geom_quat[geom_id].tolist()))
            pos_vars = [tk.DoubleVar(value=v) for v in pos]
            rot_vars = [tk.DoubleVar(value=v) for v in rot]
            self.variables[name] = {"pos": pos_vars, "rot": rot_vars}
            for row, (label, var, center) in enumerate(zip("XYZ", pos_vars, pos)):
                ttk.Label(frame, text=f"Position {label} (m)").grid(row=row, column=0, sticky="w")
                tk.Scale(frame, variable=var, from_=center - 0.04, to=center + 0.04,
                         resolution=0.0001, orient="horizontal", length=310).grid(row=row, column=1)
            for row, (label, var) in enumerate(zip(("Roll X", "Pitch Y", "Yaw Z"), rot_vars), 3):
                ttk.Label(frame, text=f"{label} (deg)").grid(row=row, column=0, sticky="w")
                tk.Scale(frame, variable=var, from_=-180, to=180, resolution=0.1,
                         orient="horizontal", length=310).grid(row=row, column=1)
        gripper_frame = ttk.LabelFrame(main, text="Gripper pose used for alignment", padding=8)
        gripper_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=(10, 0), sticky="ew")
        tk.Scale(gripper_frame, variable=self.gripper, from_=-0.60, to=0.60,
                 resolution=0.01, orient="horizontal", length=500,
                 label="Left joint qpos (right joint uses the opposite value)").pack(side="left")
        ttk.Button(gripper_frame, text="Open 0.45", command=self.set_open).pack(side="left", padx=4)
        ttk.Button(gripper_frame, text="Closed -0.40", command=self.set_closed).pack(side="left", padx=4)
        ttk.Label(gripper_frame, textvariable=self.parallel_angle).pack(side="right", padx=8)

        buttons = ttk.Frame(main, padding=(0, 10, 0, 0))
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(buttons, text="Reset", command=self.reset).pack(side="left", padx=4)
        ttk.Button(buttons, text="Save to XML", command=self.save).pack(side="left", padx=4)
        ttk.Button(buttons, text="Close", command=self.close).pack(side="right", padx=4)
        ttk.Label(main, textvariable=self.status).grid(row=3, column=0, columnspan=2, sticky="w")

    def set_open(self) -> None:
        self.gripper.set(GRIPPER_OPEN)

    def set_closed(self) -> None:
        self.gripper.set(GRIPPER_CLOSED)

    def apply(self) -> None:
        left_qpos = self.gripper.get()
        self.data.qpos[self.gripper_qpos["gripper_left"]] = left_qpos
        self.data.qpos[self.gripper_qpos["gripper_right"]] = -left_qpos
        for name, geom_id in self.pad_ids.items():
            values = self.variables[name]
            self.model.geom_pos[geom_id] = [var.get() for var in values["pos"]]
            self.model.geom_quat[geom_id] = euler_deg_to_quat([var.get() for var in values["rot"]])
        mujoco.mj_forward(self.model, self.data)
        normals = []
        for name in PAD_NAMES:
            matrix = self.data.geom_xmat[self.pad_ids[name]]
            normals.append((matrix[0], matrix[3], matrix[6]))
        dot = abs(sum(a * b for a, b in zip(normals[0], normals[1])))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        self.parallel_angle.set(f"Pad face angle: {angle:.2f}°")

    def reset(self) -> None:
        for name in PAD_NAMES:
            pos, quat = self.initial[name]
            for var, value in zip(self.variables[name]["pos"], pos):
                var.set(float(value))
            for var, value in zip(self.variables[name]["rot"], quat_to_euler_deg(quat.tolist())):
                var.set(float(value))
        self.apply()
        self.status.set("Reset to the values loaded from XML.")

    def save(self) -> None:
        self.apply()
        try:
            xml = XML_PATH.read_text(encoding="utf-8")
            for name, geom_id in self.pad_ids.items():
                pos = " ".join(f"{v:.8f}" for v in self.model.geom_pos[geom_id])
                quat = " ".join(f"{v:.8f}" for v in self.model.geom_quat[geom_id])
                xml = replace_geom_attribute(xml, name, "pos", pos)
                xml = replace_geom_attribute(xml, name, "quat", quat)
            # Parse before writing. The active model remains unchanged if validation fails.
            mujoco.MjModel.from_xml_string(xml, assets={
                path.name: path.read_bytes() for path in XML_PATH.parent.glob("*.stl")
            })
            XML_PATH.write_text(xml, encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            self.status.set("Save failed; XML was not changed.")
            return
        self.status.set(f"Saved pad pos/quat to {XML_PATH.name}")
        messagebox.showinfo("Saved", f"Saved both pads to:\n{XML_PATH}", parent=self.root)

    def close(self) -> None:
        self.closed = True


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = tk.Tk()
    tuner = PadTuner(root, model, data)
    print(f"Loaded: {XML_PATH}")
    print("Use the slider window to adjust the pads and save them to XML.")
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        while viewer.is_running() and not tuner.closed:
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                break
            tuner.apply()
            viewer.sync()
    if root.winfo_exists():
        root.destroy()


if __name__ == "__main__":
    main()
