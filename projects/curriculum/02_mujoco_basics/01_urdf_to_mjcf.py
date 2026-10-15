#!/usr/bin/env python3
"""URDF를 MuJoCo가 읽을 수 있는 기본 MJCF로 변환합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/02_mujoco_basics/01_urdf_to_mjcf.py

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = PROJECT_ROOT / "robot_description/urdf/jdcobot200.urdf"
MESH_DIR = PROJECT_ROOT / "robot_description/urdf/assets"
LESSON_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LESSON_DIR / "generated"
RAW_MJCF_PATH = OUTPUT_DIR / "jdcobot200_raw.xml"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
    mujoco.mj_saveLastXML(str(RAW_MJCF_PATH), model)

    # artifacts 폴더에서도 mesh를 찾도록 생성된 MJCF의 경로를 명시한다.
    tree = ET.parse(RAW_MJCF_PATH)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(MESH_DIR))
    tree.write(RAW_MJCF_PATH, encoding="utf-8", xml_declaration=True)

    print("URDF -> MJCF 변환 완료")
    print(f"  입력: {URDF_PATH}")
    print(f"  출력: {RAW_MJCF_PATH}")
    print(f"  자유도(nv): {model.nv}")
    print(f"  actuator 수: {model.nu}")
    print("\n아직 servo damping, frictionloss, armature, actuator가 없습니다.")
    print("다음으로 02_add_dynamics.py를 실행하세요.")


if __name__ == "__main__":
    main()
