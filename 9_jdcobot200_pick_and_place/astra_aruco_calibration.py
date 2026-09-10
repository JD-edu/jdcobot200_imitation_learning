#!/usr/bin/env python3
"""Calibrate Orbbec Astra RGB coordinates to the JDCobot200 IK base frame.

The fixed ArUco marker pose must be measured in the robot base/IK frame.  The
result maps camera-frame metres to base-frame metres::

    p_base = R_base_camera @ p_camera + t_base_camera
"""

from __future__ import annotations

import argparse
import ctypes as C
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "camera_to_robot_calibration.json"


def _load_astra_viewer():
    path = ROOT.parent / "depth_cam" / "astra_viewer.py"
    if not path.is_file():
        raise RuntimeError(f"Astra OpenNI helper not found: {path}")
    spec = importlib.util.spec_from_file_location("jdcobot_astra_viewer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rpy_matrix(degrees: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def rotation_mean(rotations: list[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((4, 4))
    for rotation in rotations:
        quaternion = rotation_matrix_to_quaternion(rotation)
        if quaternion[0] < 0:
            quaternion = -quaternion
        accumulator += np.outer(quaternion, quaternion)
    quaternion = np.linalg.eigh(accumulator)[1][:, -1]
    return quaternion_to_rotation_matrix(quaternion)


def rotation_matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    # Eigen-decomposition is stable even close to 180 degrees. Output is wxyz.
    k = np.array([
        [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0],
         r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
        [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2],
         r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
        [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1],
         r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0],
         r[1, 0] - r[0, 1], r[0, 0] + r[1, 1] + r[2, 2]],
    ]) / 3.0
    q_xyzw = np.linalg.eigh(k)[1][:, -1]
    return q_xyzw[[3, 0, 1, 2]]


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


class AstraColor:
    """Small context manager around the workspace's OpenNI2 runtime."""

    # OniStreamProperty enum (OniProperties.h).
    FOV_HORIZONTAL = 1
    FOV_VERTICAL = 2

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.av = _load_astra_viewer()
        self.width, self.height, self.fps = width, height, fps
        self.lib = self.av.load_openni()
        self.device = C.c_void_p()
        self.stream = C.c_void_p()
        self.initialized = False

    def __enter__(self):
        av, lib = self.av, self.lib
        if lib.oniInitialize(av.ONI_API_VERSION) != av.ONI_OK:
            raise av.error(lib, "OpenNI2 initialize", -1)
        self.initialized = True
        devices, count = C.POINTER(av.DeviceInfo)(), C.c_int()
        status = lib.oniGetDeviceList(C.byref(devices), C.byref(count))
        if status != av.ONI_OK or count.value == 0:
            raise RuntimeError("No Astra camera found (check USB permission).")
        uri = bytes(devices[0].uri).split(b"\0", 1)[0]
        status = lib.oniDeviceOpen(uri, C.byref(self.device))
        lib.oniReleaseDeviceList(devices)
        if status != av.ONI_OK:
            raise av.error(lib, "Open camera", status)
        status = lib.oniDeviceCreateStream(self.device, av.SENSORS["color"],
                                           C.byref(self.stream))
        if status != av.ONI_OK:
            raise av.error(lib, "Create color stream", status)
        mode = av.VideoMode(av.FORMATS["color"], self.width, self.height, self.fps)
        status = lib.oniStreamSetProperty(self.stream, av.ONI_STREAM_PROPERTY_VIDEO_MODE,
                                          C.byref(mode), C.sizeof(mode))
        if status != av.ONI_OK:
            raise av.error(lib, "Set color mode", status)
        lib.oniStreamGetProperty.argtypes = [C.c_void_p, C.c_int, C.c_void_p,
                                             C.POINTER(C.c_int)]
        lib.oniStreamGetProperty.restype = C.c_int
        status = lib.oniStreamStart(self.stream)
        if status != av.ONI_OK:
            raise av.error(lib, "Start color stream", status)
        return self

    def fov_intrinsics(self) -> tuple[np.ndarray, tuple[float, float]]:
        fovs = []
        for prop in (self.FOV_HORIZONTAL, self.FOV_VERTICAL):
            value, size = C.c_float(), C.c_int(C.sizeof(C.c_float))
            status = self.lib.oniStreamGetProperty(
                self.stream, prop, C.byref(value), C.byref(size))
            if status != self.av.ONI_OK or not 0.1 < value.value < math.pi:
                raise RuntimeError("Camera FOV query failed; use --fx/--fy/--cx/--cy.")
            fovs.append(float(value.value))
        hfov, vfov = fovs
        fx = self.width / (2.0 * math.tan(hfov / 2.0))
        fy = self.height / (2.0 * math.tan(vfov / 2.0))
        return np.array([[fx, 0, (self.width - 1) / 2],
                         [0, fy, (self.height - 1) / 2], [0, 0, 1.0]]), (hfov, vfov)

    def read(self) -> np.ndarray:
        frame_ptr = C.POINTER(self.av.Frame)()
        status = self.lib.oniStreamReadFrame(self.stream, C.byref(frame_ptr))
        if status != self.av.ONI_OK:
            raise self.av.error(self.lib, "Read color frame", status)
        try:
            return self.av.frame_image(frame_ptr.contents, "color", 5000)
        finally:
            self.lib.oniFrameRelease(frame_ptr)

    def __exit__(self, *_):
        if self.stream.value:
            self.lib.oniStreamStop(self.stream)
            self.lib.oniStreamDestroy(self.stream)
        if self.device.value:
            self.lib.oniDeviceClose(self.device)
        if self.initialized:
            self.lib.oniShutdown()


def marker_pose(corners: np.ndarray, size_m: float, camera: np.ndarray,
                distortion: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    half = size_m / 2
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners.reshape(4, 2), camera,
                                  distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or tvec[2, 0] <= 0:
        raise RuntimeError("Marker pose solve failed.")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera, distortion)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) -
                                          corners.reshape(4, 2)) ** 2, axis=1))))
    return cv2.Rodrigues(rvec)[0], tvec[:, 0], error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Astra RGB -> JDCobot IK base calibration")
    p.add_argument("--marker-size-mm", type=float, required=True)
    p.add_argument("--marker-id", type=int, default=None,
                   help="required if more than one marker is visible")
    p.add_argument("--marker-in-base-mm", type=float, nargs=3, required=True,
                   metavar=("X", "Y", "Z"), help="marker centre relative to physical baseframe")
    p.add_argument("--marker-rpy-deg", type=float, nargs=3, required=True,
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="marker axes orientation in physical baseframe (Rz*Ry*Rx)")
    p.add_argument("--ik-origin-from-base-mm", type=float, nargs=3,
                   default=(100.0, 0.0, 200.0), metavar=("X", "Y", "Z"),
                   help="scene.xml IK/world translation of physical baseframe")
    p.add_argument("--dictionary", default="AUTO",
                   help="OpenCV dictionary name, or AUTO (default)")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--max-seconds", type=float, default=20)
    p.add_argument("--max-reprojection-error", type=float, default=1.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fx", type=float); p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float); p.add_argument("--cy", type=float)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--snapshot", type=Path, default=None,
                   help="save the last annotated RGB frame (also on detection failure)")
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.marker_size_mm <= 0 or args.samples < 3:
        raise ValueError("marker size must be positive and samples must be >= 3")
    # Keep each family together from the smallest to largest dictionary.  A
    # marker with a high ID (the supplied marker is 4X4_1000, ID 871) cannot be
    # decoded by the smaller 50/100/250 dictionaries.
    auto_names = tuple(
        f"DICT_{bits}X{bits}_{count}"
        for bits in (4, 5, 6, 7)
        for count in (50, 100, 250, 1000)
    ) + ("DICT_ARUCO_ORIGINAL",)
    dictionary_names = auto_names if args.dictionary.upper() == "AUTO" else (args.dictionary,)
    try:
        detectors = [(name, cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)),
            cv2.aruco.DetectorParameters())) for name in dictionary_names]
    except AttributeError as exc:
        raise ValueError(f"Unknown ArUco dictionary: {args.dictionary}") from exc
    custom = [args.fx, args.fy, args.cx, args.cy]
    if any(v is not None for v in custom) and not all(v is not None for v in custom):
        raise ValueError("Provide all of --fx --fy --cx --cy, or none.")
    rotations, translations, errors = [], [], []
    selected_id = args.marker_id
    selected_dictionary = None
    last_frame = None
    with AstraColor(args.width, args.height, args.fps) as stream:
        if all(v is not None for v in custom):
            camera = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1.]])
            fov = None
        else:
            camera, fov = stream.fov_intrinsics()
        distortion = np.zeros(5)  # FOV-derived fallback; calibrated intrinsics are preferable.
        start = time.monotonic()
        while len(rotations) < args.samples and time.monotonic() - start < args.max_seconds:
            frame = stream.read()
            last_frame = frame
            corners, ids = [], None
            active_detectors = detectors if selected_dictionary is None else [
                item for item in detectors if item[0] == selected_dictionary]
            for dictionary_name, detector in active_detectors:
                candidate_corners, candidate_ids, _ = detector.detectMarkers(frame)
                if candidate_ids is not None:
                    corners, ids = candidate_corners, candidate_ids
                    selected_dictionary = dictionary_name
                    break
            if ids is not None:
                visible = ids[:, 0].tolist()
                if selected_id is None:
                    if len(visible) == 1:
                        selected_id = visible[0]
                    else:
                        raise RuntimeError(f"Multiple markers visible {visible}; set --marker-id.")
                if selected_id in visible:
                    index = visible.index(selected_id)
                    r, t, error = marker_pose(corners[index], args.marker_size_mm / 1000,
                                              camera, distortion)
                    if error <= args.max_reprojection_error:
                        rotations.append(r); translations.append(t); errors.append(error)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if args.preview:
                cv2.putText(frame, f"samples {len(rotations)}/{args.samples}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                cv2.imshow("Astra ArUco calibration", frame)
                if cv2.waitKey(1) & 0xff in (27, ord('q')):
                    break
    cv2.destroyAllWindows()
    if args.snapshot is not None and last_frame is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), last_frame):
            raise RuntimeError(f"Could not save snapshot: {args.snapshot}")
        print(f"snapshot: {args.snapshot.resolve()}")
    if len(rotations) < 3:
        raise RuntimeError(f"Only {len(rotations)} valid samples; marker was not detected reliably.")

    r_camera_marker = rotation_mean(rotations)
    t_camera_marker = np.median(np.asarray(translations), axis=0)
    t_ik_base = transform(np.eye(3), np.asarray(args.ik_origin_from_base_mm) / 1000)
    t_base_marker = transform(rpy_matrix(args.marker_rpy_deg),
                              np.asarray(args.marker_in_base_mm) / 1000)
    t_ik_marker = t_ik_base @ t_base_marker
    t_camera_marker_4 = transform(r_camera_marker, t_camera_marker)
    t_base_camera = t_ik_marker @ np.linalg.inv(t_camera_marker_4)
    translation_spread_mm = np.std(np.asarray(translations), axis=0) * 1000
    payload = {
        "schema": "jdcobot200-camera-to-base-v1",
        "units": "metres",
        "camera_frame": "OpenCV: +x right, +y down, +z forward",
        "base_frame": "MuJoCo/IK scene.xml world frame (despite T_base_camera legacy key)",
        "marker_id": int(selected_id), "aruco_dictionary": selected_dictionary,
        "marker_size_mm": args.marker_size_mm,
        "marker_in_base_mm": args.marker_in_base_mm,
        "marker_rpy_deg": args.marker_rpy_deg,
        "ik_origin_from_base_mm": args.ik_origin_from_base_mm,
        "camera_matrix": camera.tolist(), "distortion": distortion.tolist(),
        "fov_radians": list(fov) if fov else None,
        "sample_count": len(rotations),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "camera_marker_translation_std_mm": translation_spread_mm.tolist(),
        "T_camera_marker": t_camera_marker_4.tolist(),
        "T_base_camera": t_base_camera.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"marker id: {selected_id}, samples: {len(rotations)}")
    print(f"mean reprojection error: {np.mean(errors):.3f} px")
    print(f"translation std: {translation_spread_mm.round(2)} mm")
    print("T_base_camera:\n", np.array2string(t_base_camera, precision=7, suppress_small=True))
    print(f"saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
