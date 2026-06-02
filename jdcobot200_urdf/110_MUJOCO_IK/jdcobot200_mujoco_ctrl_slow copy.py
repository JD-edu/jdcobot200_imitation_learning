import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# [핵심 추가] 메인 시뮬레이션 data를 오염시키지 않을 IK 수치해석 전용 가상 데이터 객체 생성
virtual_data = mj.MjData(model)

# ===== IK 설정 =====
ee_body_name = "gripper_assembly"
ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)

joint_names = ["base", "shoulder", "elbow", "wrist", "gripper_base"]
joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]
dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]

# xyz 목표점 2개
target_A = np.array([0.18,  0.08, 0.10])
target_B = np.array([0.18, -0.08, 0.16])

switch_interval = 2.0

ik_iters = 15
ik_alpha = 0.5        # 보간이 추가되었으므로 IK 스텝 크기를 안정적인 0.5로 소폭 하향
ik_damping = 0.05     # 싱귤러리티 방지를 위해 대칭 댐핑 소폭 상향
ik_tol = 1e-3

# actuator는 position actuator이므로 ctrl에는 목표 joint angle을 넣음
q_des = data.qpos[qpos_ids].copy()

# --- 선형 보간을 위한 가상 제어 변수 ---
INTERPOLATION_STEP = 0.5  # 실제 로봇처럼 매우 끈적하고 천천히 움직이게 유도 (필요시 조절)
mj.mj_forward(model, data)
current_target = data.xpos[ee_id].copy()

# ===== GLFW =====
glfw.init()
window = glfw.create_window(800, 600, "MuJoCo Numerical IK Safe Control", None, None)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.lookat[:] = [0.0, 0.0, 0.08]
cam.distance = 0.8
cam.azimuth = 45.0
cam.elevation = -25.0


def numerical_ik(target_pos):
    global q_des

    # [핵심] 실제 data가 아닌 virtual_data 상에서 포워드 연산을 진행합니다.
    q = data.qpos.copy()
    q[qpos_ids] = q_des

    for _ in range(ik_iters):
        virtual_data.qpos[:] = q
        virtual_data.qvel[:] = 0
        mj.mj_forward(model, virtual_data)  # 가상 메모리 전방 기구학 업데이트

        ee_pos = virtual_data.xpos[ee_id].copy()
        err = target_pos - ee_pos

        if np.linalg.norm(err) < ik_tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, virtual_data, jacp, jacr, ee_id)

        J = jacp[:, dof_ids]

        # Damped Least Squares
        A = J @ J.T + (ik_damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        q_new = q[qpos_ids] + ik_alpha * dq

        # joint limit 적용
        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q_new[i] = np.clip(q_new[i], low, high)

        q[qpos_ids] = q_new

    q_des = q[qpos_ids].copy()
    return q_des


while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev) < (1.0 / 60.0):
        # 1. 시간에 따라 최종 타겟 결정
        if int(data.time // switch_interval) % 2 == 0:
            final_destination = target_A
        else:
            final_destination = target_B

        # 2. 3차원 목표점을 미세 스텝단위로 선형 보간 이동
        direction_err = final_destination - current_target
        dist = np.linalg.norm(direction_err)
        
        if dist > INTERPOLATION_STEP:
            current_target += (direction_err / dist) * INTERPOLATION_STEP
        else:
            current_target = final_destination.copy()

        # 3. 오염되지 않은 가상 타겟으로 IK 연산 후 실제 액추에이터 주입
        target_q = numerical_ik(current_target)
        data.ctrl[:len(target_q)] = target_q

        # 실제 물리 엔진 스텝 진행
        mj.mj_step(model, data)

    # 60Hz 화면 렌더링 영역
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(
        model, data, mj.MjvOption(), None, cam,
        mj.mjtCatBit.mjCAT_ALL, scene
    )
    mj.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()