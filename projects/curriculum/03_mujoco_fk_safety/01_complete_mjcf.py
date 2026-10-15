#!/usr/bin/env python3
"""과목 02 stable MJCF에 TCP와 안전검사용 collision proxy를 추가합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/01_complete_mjcf.py

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "projects/curriculum/02_mujoco_basics/generated/jdcobot200_stable.xml"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
OUTPUT_PATH = OUTPUT_DIR / "jdcobot200_complete.xml"
TCP_POSITION = "0 0 -0.08"  # gripper_assembly 원점에서 실제 집기 중심까지, 단위 m

PROXIES = {
    "base_assembly": ("collision_base", "cylinder", {"pos": "0 0 0.045", "size": "0.055 0.04"}),
    "shoulder_assmebly": ("collision_shoulder", "sphere", {"pos": "0 0 -0.015", "size": "0.045"}),
    "arm_link_assembly": ("collision_upper_arm", "capsule", {"fromto": "0 0 0 0.1352 0 0", "size": "0.025"}),
    "arm_link_assembly_2": ("collision_forearm", "capsule", {"fromto": "0 0 0 0.1352 0 0", "size": "0.025"}),
    "wrist_assembly": ("collision_wrist", "capsule", {"fromto": "0 0 0 0.0575 0 0", "size": "0.026"}),
    "gripper_assembly": ("collision_gripper", "capsule", {"fromto": f"0 0 0 {TCP_POSITION}", "size": "0.025"}),
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit("과목 02의 01, 02 스크립트를 먼저 실행하세요.")
    tree = ET.parse(INPUT_PATH)
    root = tree.getroot()

    # include하는 scene의 위치와 무관하게 후보 모델을 열 수 있게 절대경로를 쓴다.
    # 공식 XML에는 저장소 이동이 가능하도록 기존의 ../meshes 경로를 유지한다.
    mesh_dir = PROJECT_ROOT / "robot_description/urdf/assets"
    root.find("compiler").set("meshdir", str(mesh_dir))

    bodies = {body.get("name"): body for body in root.findall(".//body")}
    missing = [name for name in PROXIES if name not in bodies]
    if missing:
        raise SystemExit("body를 찾지 못했습니다: " + ", ".join(missing))

    gripper = bodies["gripper_assembly"]
    for old_site in list(gripper.findall("site")):
        if old_site.get("name") == "graspframe":
            gripper.remove(old_site)
    gripper.insert(1, ET.Element("site", {
        "name": "graspframe", "pos": TCP_POSITION,
        "size": "0.006", "rgba": "0 1 0 1",
    }))

    for body_name, (geom_name, geom_type, attributes) in PROXIES.items():
        body = bodies[body_name]
        for old_geom in list(body.findall("geom")):
            if old_geom.get("name") == geom_name:
                body.remove(old_geom)
        geom = ET.Element("geom", {
            "name": geom_name, "type": geom_type,
            "contype": "0", "conaffinity": "0", "group": "3",
            "rgba": "1 0.2 0.2 0.15", **attributes,
        })
        body.insert(1, geom)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_PATH, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_PATH))
    print("FK용 완성 후보 MJCF 생성:", OUTPUT_PATH)
    print("TCP graspframe:", TCP_POSITION, "m")
    print("collision proxy:", len(PROXIES), "개")
    print("검증: nq=", model.nq, "nu=", model.nu)
    print("\n주의: TCP 실측과 안전 검사를 마친 뒤에만 공식 XML로 승격하세요.")


if __name__ == "__main__":
    main()
