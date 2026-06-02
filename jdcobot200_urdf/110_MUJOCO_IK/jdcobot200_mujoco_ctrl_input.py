import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import threading

xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 메인 시뮬레이션 data를 오염시키지 않을 IK 수치해석 전용 가상 데이터 객체
virtual_data = mj.MjData(model)

# ===== IK 설정 =====
ee_body_name = "gripper_assembly"
ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)

joint_names = ["base", "shoulder", "elbow", "wrist", "gripper_base"]
joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]
dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]

ik_iters = 15
ik_alpha = 0.5        
ik_damping = 0.05     
ik_tol = 1e-3

# actuator는 position actuator이므로 ctrl에는 목표 joint angle을 넣음
q_des = data.qpos[qpos_ids].copy()

# ===== 실시간 입력 및 타겟 제어 변수 =====
mj.mj_forward(model, data)
# 최초 시작 시 로봇의 초기 엔드이펙터 위치를 목표점으로 설정
initial_pos = data.xpos[ee_id].copy()
target_pos = initial_pos.copy()

print(f"=== 로봇 현재 초기 위치 ===")
print(f"X: {target_pos[0]:.4f}, Y: {target_pos[1]:.4f}, Z: {target_pos[2]:.4f}\n")

# 사용자가 입력한 새로운 좌표를 비동기적으로 전달하기 위한 락(Lock)
target_lock = threading.Lock()


def numerical_ik(target_pos_ik):
    global q_des

    q = data.qpos.copy()
    q[qpos_ids] = q_des

    for _ in range(ik_iters):
        virtual_data.qpos[:] = q
        virtual_data.qvel[:] = 0
        mj.mj_forward(model, virtual_data)  

        ee_pos = virtual_data.xpos[ee_id].copy()
        err = target_pos_ik - ee_pos

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


# ----- [핵심] 실시간 터미널 입력을 받는 백그라운드 스레드 함수 -----
def console_input_thread():
    global target_pos
    print("--------------------------------------------------")
    print(" 원하는 3차원 XYZ 목표 좌표를 입력하세요.")
    print(" 입력 예시 -> 0.18 -0.05 0.15 (공백으로 구분)")
    print("--------------------------------------------------")
    
    while not glfw.window_should_close(window):
        try:
            user_input = input("\n[Input XYZ] >> ")
            coords = list(map(float, user_input.split()))
            
            if len(coords) == 3:
                new_target = np.array(coords)
                # 스레드 안전하게 글로벌 목표 좌표 갱신
                with target_lock:
                    target_pos = new_target.copy()
                print(f"🎯 목표 좌표 변경 완료 -> X: {coords[0]}, Y: {coords[1]}, Z: {coords[2]}")
            else:
                print("❌ 에러: 반드시 3개의 숫자(X Y Z)를 공백으로 구분하여 입력해야 합니다.")
        except ValueError:
            print("❌ 에러: 올바른 숫자를 입력해 주세요.")
        except Exception as e:
            print(f"❌ 에러 발생: {e}")


# ===== GLFW 초기화 =====
glfw.init()
window = glfw.create_window(800, 600, "MuJoCo Real-time XYZ Console Control", None, None)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.lookat[:] = [0.0, 0.0, 0.08]
cam.distance = 0.8
cam.azimuth = 45.0
cam.elevation = -25.0

# 백그라운드 콘솔 입력 스레드 시작
input_thread = threading.Thread(target=console_input_thread, daemon=True)
input_thread.start()

# ===== 메인 시뮬레이션 및 렌더링 루프 =====
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev) < (1.0 / 60.0):
        # 스레드 락을 걸고 현재 타겟 좌표를 안전하게 가져옴
        with target_lock:
            current_loop_target = target_pos.copy()

        # 실시간 변경된 target 좌표로 즉시 IK 계산
        target_q = numerical_ik(current_loop_target)
        
        # position actuator 제어 주입
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