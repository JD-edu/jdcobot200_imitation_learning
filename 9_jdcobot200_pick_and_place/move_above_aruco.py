#!/usr/bin/env python3
"""Measure the fixed base ArUco and optionally move the TCP 100 mm above it.

Default mode is measurement-only. Real motion is delegated to the existing
READY -> Cartesian IK controller, which checks joint limits, safe-zone and
singularities and asks the operator to type MOVE before enabling torque.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from astra_aruco_calibration import AstraColor, marker_pose
from camera_robot_transform import camera_to_base


ROOT = Path(__file__).resolve().parent
READY_IK = ROOT.parent / "8_jdcobot200_IK" / "jdcobot200_ready_ik_sync.py"
SAFE_IK = ROOT.parent / "8_jdcobot200_IK" / "jdcobot200_safe_mink_ik.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArUco 중심의 IK/world +Z 100 mm 위치를 측정/검증/이동합니다."
    )
    parser.add_argument(
        "--mode", choices=("measure", "simulate", "move"), default="measure",
        help="measure=좌표만 출력(기본), simulate=MuJoCo 검증, move=실제 로봇 이동",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=ROOT / "camera_to_robot_calibration.json",
    )
    parser.add_argument("--height-mm", type=float, default=100.0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-reprojection-error", type=float, default=1.5)
    parser.add_argument(
        "--max-marker-deviation-mm", type=float, default=20.0,
        help="캘리브레이션 당시 마커 좌표와 현재 측정값의 최대 차이",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "aruco_target_check.png")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--ready-time", type=float, default=6.0)
    parser.add_argument("--move-time", type=float, default=5.0)
    parser.add_argument(
        "--ready-q-deg", type=float, nargs=5,
        default=(0.0, -20.0, 40.0, -20.0, 0.0),
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="양의 elbow IK 분기의 안전 READY 자세",
    )
    return parser.parse_args()


def load_calibration(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "T_base_camera", "camera_matrix", "distortion", "marker_id",
        "aruco_dictionary", "marker_size_mm", "marker_in_base_mm",
        "ik_origin_from_base_mm",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"캘리브레이션 파일에 필드가 없습니다: {missing}")
    return payload


def expected_marker_ik(payload: dict) -> np.ndarray:
    """Marker centre expressed in the position coordinates consumed by IK."""
    return (
        np.asarray(payload["ik_origin_from_base_mm"], dtype=float)
        + np.asarray(payload["marker_in_base_mm"], dtype=float)
    ) / 1000.0


def measure_marker(args: argparse.Namespace, payload: dict) -> tuple[np.ndarray, dict]:
    dictionary_name = str(payload["aruco_dictionary"])
    try:
        dictionary_id = getattr(cv2.aruco, dictionary_name)
    except AttributeError as exc:
        raise ValueError(f"지원하지 않는 ArUco 사전: {dictionary_name}") from exc
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id),
        cv2.aruco.DetectorParameters(),
    )
    marker_id = int(payload["marker_id"])
    marker_size_m = float(payload["marker_size_mm"]) / 1000.0
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=float)
    distortion = np.asarray(payload["distortion"], dtype=float)
    transform_matrix = np.asarray(payload["T_base_camera"], dtype=float)
    translations, errors = [], []
    last_frame = None

    with AstraColor(args.width, args.height, args.fps) as stream:
        started = time.monotonic()
        while len(translations) < args.samples and time.monotonic() - started < args.max_seconds:
            frame = stream.read()
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None and marker_id in ids[:, 0]:
                index = ids[:, 0].tolist().index(marker_id)
                _, translation, error = marker_pose(
                    corners[index], marker_size_m, camera_matrix, distortion
                )
                if error <= args.max_reprojection_error:
                    translations.append(translation)
                    errors.append(error)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(
                frame, f"marker {marker_id}  samples {len(translations)}/{args.samples}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
            last_frame = frame
            if args.preview:
                cv2.imshow("ArUco target check", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    cv2.destroyAllWindows()
    if last_frame is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), last_frame):
            raise RuntimeError(f"스냅샷 저장 실패: {args.snapshot}")
    if len(translations) < max(3, args.samples // 2):
        raise RuntimeError(
            f"유효 마커 샘플이 부족합니다: {len(translations)}/{args.samples}. "
            f"스냅샷을 확인하세요: {args.snapshot}"
        )
    camera_xyz = np.median(np.asarray(translations), axis=0)
    marker_ik = camera_to_base(camera_xyz, transform_matrix)
    stats = {
        "samples": len(translations),
        "reprojection_px": float(np.mean(errors)),
        "camera_std_mm": (np.std(np.asarray(translations), axis=0) * 1000.0),
        "camera_xyz": camera_xyz,
    }
    return marker_ik, stats


def controller_command(args: argparse.Namespace, target: np.ndarray) -> list[str]:
    xyz = [f"{value:.9f}" for value in target]
    if args.mode == "simulate":
        return [
            sys.executable, str(SAFE_IK), "--target-xyz", *xyz,
            "--simulation-only", "--max-joint-change-deg", "180",
            "--start-q-deg", *(str(value) for value in args.ready_q_deg),
            "--move-time", str(args.move_time),
        ]
    return [
        sys.executable, str(READY_IK), "--target-xyz", *xyz,
        "--port", args.port, "--baudrate", str(args.baudrate),
        "--ready-time", str(args.ready_time), "--move-time", str(args.move_time),
        "--ready-q-deg", *(str(value) for value in args.ready_q_deg),
        "--return-to", "stay",
    ]


def main() -> int:
    args = parse_args()
    if args.height_mm <= 0 or args.samples < 3 or args.max_seconds <= 0:
        raise ValueError("height/samples/max-seconds 값을 확인하세요.")
    if args.mode != "measure":
        controller = SAFE_IK if args.mode == "simulate" else READY_IK
        if not controller.is_file():
            raise FileNotFoundError(f"IK 제어기를 찾을 수 없습니다: {controller}")

    payload = load_calibration(args.calibration)
    marker_ik, stats = measure_marker(args, payload)
    expected = expected_marker_ik(payload)
    deviation_mm = (marker_ik - expected) * 1000.0
    deviation_norm = float(np.linalg.norm(deviation_mm))
    target = marker_ik + np.array([0.0, 0.0, args.height_mm / 1000.0])

    print("\nArUco 측정 완료")
    print(f"유효 샘플: {stats['samples']}  평균 재투영 오차: {stats['reprojection_px']:.3f} px")
    print("카메라 측정 표준편차(mm):", np.round(stats["camera_std_mm"], 3))
    print("카메라 marker XYZ(m):", np.round(stats["camera_xyz"], 6))
    print("측정 marker IK/world XYZ(m):", np.round(marker_ik, 6))
    print("캘리브레이션 기준 marker XYZ(m):", np.round(expected, 6))
    print("기준 대비 XYZ 차이(mm):", np.round(deviation_mm, 3),
          f"거리={deviation_norm:.2f} mm")
    print(f"최종 TCP 목표(+Z {args.height_mm:.1f} mm):", np.round(target, 6), "m")
    print("스냅샷:", args.snapshot.resolve())

    if deviation_norm > args.max_marker_deviation_mm:
        raise RuntimeError(
            f"마커 위치 차이 {deviation_norm:.2f} mm가 허용값 "
            f"{args.max_marker_deviation_mm:.2f} mm를 초과했습니다. "
            "카메라/마커가 움직였거나 캘리브레이션 입력값이 잘못됐습니다."
        )
    if args.mode == "measure":
        print("MEASURE ONLY: 로봇에 연결하거나 모터 명령을 보내지 않았습니다.")
        print("다음 단계: --mode simulate, 확인 후 --mode move")
        return 0

    command = controller_command(args, target)
    print(f"\n{args.mode.upper()} IK 제어기 실행:", " ".join(command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1)
