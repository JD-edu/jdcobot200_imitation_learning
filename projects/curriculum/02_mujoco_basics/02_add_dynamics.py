#!/usr/bin/env python3
"""단순 변환한 MJCF에 servo 동역학과 position actuator를 추가합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/02_mujoco_basics/02_add_dynamics.py

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LESSON_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LESSON_DIR / "generated"
RAW_MJCF_PATH = OUTPUT_DIR / "jdcobot200_raw.xml"
STABLE_MJCF_PATH = OUTPUT_DIR / "jdcobot200_stable.xml"

ARM_JOINTS = ("base", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
ALL_JOINTS = ARM_JOINTS + ("gripper_left", "gripper_right")
CONTROL_RANGES = {
    "base": (-3.14159, 3.14159),
    "shoulder": (-1.5708, 1.5708),
    "elbow": (-1.5708, 1.5708),
    "wrist_pitch": (-1.5708, 1.5708),
    "wrist_roll": (-3.14159, 3.14159),
    "gripper_left": (-0.57, 0.57),
}

# STS3215를 출발점으로 삼은 교육용 값이다. 실제 응답에 맞춰 식별해야 한다.
DAMPING = "0.60"
FRICTION_LOSS = "0.052"
ARMATURE = "0.028"
KP = "100"
KV = "2.0"
FORCE_RANGE = "-2.94 2.94"


def main() -> None:
    if not RAW_MJCF_PATH.exists():
        raise SystemExit("먼저 01_urdf_to_mjcf.py를 실행하세요.")

    tree = ET.parse(RAW_MJCF_PATH)
    root = tree.getroot()
    joints = {joint.get("name"): joint for joint in root.findall(".//joint")}
    missing = [name for name in ALL_JOINTS if name not in joints]
    if missing:
        raise SystemExit(f"MJCF에서 관절을 찾지 못했습니다: {', '.join(missing)}")

    # 점성 저항, 정지 마찰, motor와 감속기의 등가 관성을 추가한다.
    for name in ALL_JOINTS:
        joint = joints[name]
        joint.set("damping", DAMPING)
        joint.set("frictionloss", FRICTION_LOSS)
        joint.set("armature", ARMATURE)

    # 상세 STL collision mesh가 겹치면 특히 base 관절이 접촉력에 묶인다.
    # 이 기초 제어 과목에서는 시각화만 하고, 단순 collision geom은 이후에 만든다.
    geoms = root.findall(".//geom")
    for geom in geoms:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    old_actuator = root.find("actuator")
    if old_actuator is not None:
        root.remove(old_actuator)
    actuator = ET.SubElement(root, "actuator")

    for name in ARM_JOINTS + ("gripper_left",):
        lower, upper = CONTROL_RANGES[name]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": "gripper_motor" if name == "gripper_left" else name,
                "joint": name,
                "kp": KP,
                "kv": KV,
                "ctrllimited": "true",
                "ctrlrange": f"{lower} {upper}",
                "forcerange": FORCE_RANGE,
            },
        )

    # URDF에는 mimic 관계가 없으므로 두 finger를 반대 방향으로 묶는다.
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "joint1": "gripper_right",
            "joint2": "gripper_left",
            "polycoef": "0 -1 0 0 0",
        },
    )

    tree.write(STABLE_MJCF_PATH, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(STABLE_MJCF_PATH))
    print("동역학 보강 MJCF 생성 완료")
    print(f"  출력: {STABLE_MJCF_PATH}")
    print(f"  damping: {DAMPING}")
    print(f"  frictionloss: {FRICTION_LOSS}")
    print(f"  armature: {ARMATURE}")
    print(f"  position actuator 수: {model.nu}")
    print(f"  비활성화한 상세 mesh collision: {len(geoms)}개")
    print("\n이 값은 정답이 아니라 안정적인 시뮬레이션을 위한 시작점입니다.")


if __name__ == "__main__":
    main()
