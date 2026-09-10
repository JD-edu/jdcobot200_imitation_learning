#!/usr/bin/env python3
"""ROS 없이 Orbbec Astra의 Color/Depth/IR 영상을 표시한다."""

from __future__ import annotations

import argparse
import ctypes as C
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np


ONI_OK = 0
ONI_API_VERSION = 2003
ONI_STREAM_PROPERTY_VIDEO_MODE = 3

SENSORS = {"ir": 1, "color": 2, "depth": 3}
FORMATS = {"depth": 100, "color": 200, "ir": 203}


class VideoMode(C.Structure):
    _fields_ = [
        ("pixel_format", C.c_int),
        ("width", C.c_int),
        ("height", C.c_int),
        ("fps", C.c_int),
    ]


class Frame(C.Structure):
    _fields_ = [
        ("data_size", C.c_int),
        ("data", C.c_void_p),
        ("sensor_type", C.c_int),
        ("timestamp", C.c_uint64),
        ("frame_index", C.c_int),
        ("width", C.c_int),
        ("height", C.c_int),
        ("video_mode", VideoMode),
        ("cropping_enabled", C.c_int),
        ("crop_origin_x", C.c_int),
        ("crop_origin_y", C.c_int),
        ("stride", C.c_int),
    ]


class DeviceInfo(C.Structure):
    _fields_ = [
        ("uri", C.c_char * 256),
        ("vendor", C.c_char * 256),
        ("name", C.c_char * 256),
        ("usb_vendor_id", C.c_uint16),
        ("usb_product_id", C.c_uint16),
    ]


def runtime_dir() -> Path:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "arm" if machine.startswith("arm") else "x64"
    return Path(__file__).resolve().parent / "src/ros2_astra_camera/astra_camera/openni2_redist" / arch


def load_openni() -> C.CDLL:
    root = runtime_dir()
    library = root / "libOpenNI2.so"
    drivers = root / "OpenNI2/Drivers"
    if not library.is_file() or not drivers.is_dir():
        raise RuntimeError(f"Orbbec OpenNI2 런타임을 찾을 수 없습니다: {root}")

    # OpenNI2가 센서 플러그인을 확실히 찾도록 실행 전에 경로를 지정한다.
    os.environ["OPENNI2_REDIST"] = str(root)
    os.environ["OPENNI2_DRIVERS_PATH"] = str(drivers)
    lib = C.CDLL(str(library), mode=C.RTLD_GLOBAL)

    handle = C.c_void_p
    lib.oniInitialize.argtypes = [C.c_int]
    lib.oniInitialize.restype = C.c_int
    lib.oniShutdown.argtypes = []
    lib.oniGetExtendedError.restype = C.c_char_p
    lib.oniGetDeviceList.argtypes = [C.POINTER(C.POINTER(DeviceInfo)), C.POINTER(C.c_int)]
    lib.oniGetDeviceList.restype = C.c_int
    lib.oniReleaseDeviceList.argtypes = [C.POINTER(DeviceInfo)]
    lib.oniDeviceOpen.argtypes = [C.c_char_p, C.POINTER(handle)]
    lib.oniDeviceOpen.restype = C.c_int
    lib.oniDeviceClose.argtypes = [handle]
    lib.oniDeviceCreateStream.argtypes = [handle, C.c_int, C.POINTER(handle)]
    lib.oniDeviceCreateStream.restype = C.c_int
    lib.oniStreamSetProperty.argtypes = [handle, C.c_int, C.c_void_p, C.c_int]
    lib.oniStreamSetProperty.restype = C.c_int
    lib.oniStreamStart.argtypes = [handle]
    lib.oniStreamStart.restype = C.c_int
    lib.oniStreamStop.argtypes = [handle]
    lib.oniStreamDestroy.argtypes = [handle]
    lib.oniStreamReadFrame.argtypes = [handle, C.POINTER(C.POINTER(Frame))]
    lib.oniStreamReadFrame.restype = C.c_int
    lib.oniFrameRelease.argtypes = [C.POINTER(Frame)]
    return lib


def error(lib: C.CDLL, action: str, status: int) -> RuntimeError:
    detail = lib.oniGetExtendedError()
    message = detail.decode(errors="replace") if detail else "상세 오류 없음"
    return RuntimeError(f"{action} 실패 (OpenNI status {status}): {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 없는 Orbbec Astra RGB/Depth/IR 뷰어")
    parser.add_argument(
        "--stream", choices=("color", "depth", "ir", "color-depth", "depth-ir"),
        default="color-depth", help="표시할 스트림 (기본값: color-depth)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=5000, metavar="MM",
                        help="Depth 컬러맵의 최대 거리 mm (기본값: 5000)")
    return parser.parse_args()


def frame_image(frame: Frame, kind: str, max_depth: int) -> np.ndarray:
    if kind == "color":
        raw = np.ctypeslib.as_array(C.cast(frame.data, C.POINTER(C.c_uint8)),
                                    shape=(frame.data_size,)).copy()
        rgb = raw.reshape(frame.height, frame.stride // 3, 3)[:, :frame.width]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    count = frame.data_size // C.sizeof(C.c_uint16)
    raw16 = np.ctypeslib.as_array(C.cast(frame.data, C.POINTER(C.c_uint16)),
                                  shape=(count,)).copy()
    image16 = raw16.reshape(frame.height, frame.stride // 2)[:, :frame.width]
    if kind == "ir":
        return cv2.applyColorMap(cv2.convertScaleAbs(image16, alpha=255.0 / max(1, int(image16.max()))),
                                 cv2.COLORMAP_BONE)
    clipped = np.clip(image16, 0, max_depth)
    display = cv2.convertScaleAbs(clipped, alpha=255.0 / max_depth)
    display[image16 == 0] = 0
    return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)


def main() -> int:
    args = parse_args()
    kinds = args.stream.split("-")
    lib = load_openni()
    device = C.c_void_p()
    streams: dict[str, C.c_void_p] = {}
    initialized = False
    try:
        status = lib.oniInitialize(ONI_API_VERSION)
        if status != ONI_OK:
            raise error(lib, "OpenNI2 초기화", status)
        initialized = True

        devices = C.POINTER(DeviceInfo)()
        count = C.c_int()
        status = lib.oniGetDeviceList(C.byref(devices), C.byref(count))
        if status != ONI_OK:
            raise error(lib, "장치 검색", status)
        if count.value == 0:
            raise RuntimeError("연결된 OpenNI2 카메라가 없습니다. USB 연결과 udev 권한을 확인하세요.")
        info = devices[0]
        uri = bytes(info.uri).split(b"\0", 1)[0]
        print(f"카메라: {info.vendor.decode(errors='replace')} {info.name.decode(errors='replace')} ({uri.decode()})")
        status = lib.oniDeviceOpen(uri, C.byref(device))
        lib.oniReleaseDeviceList(devices)
        if status != ONI_OK:
            raise error(lib, "카메라 열기", status)

        for kind in kinds:
            stream = C.c_void_p()
            status = lib.oniDeviceCreateStream(device, SENSORS[kind], C.byref(stream))
            if status != ONI_OK:
                raise error(lib, f"{kind} 스트림 생성", status)
            streams[kind] = stream
            mode = VideoMode(FORMATS[kind], args.width, args.height, args.fps)
            status = lib.oniStreamSetProperty(stream, ONI_STREAM_PROPERTY_VIDEO_MODE,
                                              C.byref(mode), C.sizeof(mode))
            if status != ONI_OK:
                raise error(lib, f"{kind} {args.width}x{args.height}@{args.fps} 설정", status)
            status = lib.oniStreamStart(stream)
            if status != ONI_OK:
                raise error(lib, f"{kind} 스트림 시작", status)

        print("영상 표시 중 — q 또는 ESC: 종료")
        while True:
            for kind, stream in streams.items():
                frame_ptr = C.POINTER(Frame)()
                status = lib.oniStreamReadFrame(stream, C.byref(frame_ptr))
                if status != ONI_OK:
                    raise error(lib, f"{kind} 프레임 읽기", status)
                try:
                    image = frame_image(frame_ptr.contents, kind, args.max_depth)
                finally:
                    lib.oniFrameRelease(frame_ptr)
                cv2.imshow(f"Astra {kind.capitalize()}", image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        return 0
    finally:
        for stream in streams.values():
            lib.oniStreamStop(stream)
            lib.oniStreamDestroy(stream)
        if device.value:
            lib.oniDeviceClose(device)
        if initialized:
            lib.oniShutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1)
