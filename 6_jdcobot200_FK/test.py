import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os


# ==========================================
# Standard DH 변환 행렬
# T = Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)
# ==========================================
def get_dh_matrix(a, alpha, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


# ==========================================
# 교육용 DH 기반 FK
# 주의: MuJoCo/URDF와 100% 일치 목적이 아니라
# T01 @ T12 @ T23 @ T34 @ T45 구조 설명용
# ==========================================
def jdcobot200_dh_fk(joint_angles):
    q1, q2, q3, q4, q5 = joint_angles

    dh_table = [
        # a,       alpha,      d,       theta_offset
        [0.0,      np.pi/2,   0.0537,  0.0],       # base
        [0.1352,   0.0,       0.0,     np.pi/2],   # shoulder
        [0.1352,   0.0,       0.0,     0.0],       # elbow
        [0.0,     -np.pi/2,   0.0,    -np.pi/2],   # wrist_pitch
        [0.0575,   0.0,       0.0,     0.0],       # wrist_roll / tool direction
    ]

    T_total = np.eye(4)

    for i, (a, alpha, d, theta_offset) in enumerate(dh_table):
        theta = joint_angles[i] + theta_offset
        T_i = get_dh_matrix(a, alpha, d, theta)
        T_total = T_total @ T_i

    return T_total


# ==========================================
# MuJoCo 모델 로딩
# ==========================================
xml_path = os.path.join(os.path.dirname(__file__), "jdcobot200.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)


# ==========================================
# 비교할 MuJoCo body 이름
# 먼저 gripper_right_assembly보다 wrist_roll 이후 중심 body를 추천
# 실제 XML에 존재하는 이름으로 바꾸세요.
# ==========================================
COMPARE_BODY_NAME = "gripper_assembly"

body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, COMPARE_BODY_NAME)

if body_id == -1:
    print(f"[경고] body '{COMPARE_BODY_NAME}'를 찾을 수 없습니다.")
    print("대신 'gripper_right_assembly'를 사용합니다.")
    COMPARE_BODY_NAME = "gripper_right_assembly"
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, COMPARE_BODY_NAME)

if body_id == -1:
    raise ValueError("비교할 MuJoCo body를 찾지 못했습니다. XML의 body name을 확인하세요.")


# ==========================================
# 테스트 포즈
# gripper까지 포함해서 6축이면 마지막 gripper는 비교에서 제외
# ==========================================
pose_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
pose_B = np.array([0.4, 0.3, -0.5, 0.2, 0.6, 0.0])

kp, kd = 500.0, 10.0
switch_interval = 2.0


# ==========================================
# GLFW / Viewer 설정
# ==========================================
glfw.init()
window = glfw.create_window(
    800,
    600,
    "DH FK vs MuJoCo FK Check",
    None,
    None
)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.lookat[:] = [0.0, 0.0, 0.1]
cam.distance = 0.8
cam.azimuth = 135.0
cam.elevation = -20.0

print_counter = 0


# ==========================================
# 시뮬레이션 루프
# ==========================================
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev) < (1.0 / 60.0):
        if (data.time // switch_interval) % 2 == 0:
            target_qpos = pose_A
        else:
            target_qpos = pose_B

        # actuator 수와 qpos 수가 다를 수 있으므로 model.nu 기준 사용
        position_error = target_qpos[:model.nu] - data.qpos[:model.nu]
        velocity_error = 0.0 - data.qvel[:model.nu]

        data.ctrl[:model.nu] = kp * position_error + kd * velocity_error

        mj.mj_step(model, data)

    # 현재 주관절 5개만 DH에 사용
    current_joints = data.qpos[:5]

    # DH FK 계산
    T_dh = jdcobot200_dh_fk(current_joints)
    dh_position = T_dh[:3, 3]
    dh_rotation = T_dh[:3, :3]

    # MuJoCo FK 결과
    mujoco_position = data.xpos[body_id].copy()
    mujoco_rotation = data.xmat[body_id].reshape(3, 3).copy()

    pos_error_mm = np.linalg.norm(dh_position - mujoco_position) * 1000.0

    print_counter += 1

    if print_counter % 90 == 0:
        print("\n" + "=" * 80)
        print(" DH FK vs MuJoCo FK 비교")
        print("=" * 80)
        print(f"비교 MuJoCo body: {COMPARE_BODY_NAME}")
        print(f"현재 관절각(rad): {np.round(current_joints, 4)}")
        print("-" * 80)
        print(f"[DH FK]     position XYZ: {np.round(dh_position, 5)}")
        print(f"[MuJoCo FK] position XYZ: {np.round(mujoco_position, 5)}")
        print(f"위치 오차: {pos_error_mm:.3f} mm")
        print("-" * 80)
        print("[DH FK] rotation:")
        print(np.round(dh_rotation, 3))
        print("[MuJoCo FK] rotation:")
        print(np.round(mujoco_rotation, 3))
        print("=" * 80)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(
        model,
        data,
        mj.MjvOption(),
        None,
        cam,
        mj.mjtCatBit.mjCAT_ALL,
        scene
    )
    mj.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()