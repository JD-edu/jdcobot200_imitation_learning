import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# 1. 모델 및 데이터 로딩
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 2. GLFW 윈도우 생성 및 설정
glfw.init()
window = glfw.create_window(800, 600, "Python-level PD Control (jdcobot200)", None, None)
glfw.make_context_current(window)

# 3. 시각화 및 카메라 객체 생성
scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 초기 카메라 위치 설정
cam.lookat[:] = [0.0, 0.0, 0.1]
cam.distance = 1.2
cam.azimuth = 135.0
cam.elevation = -15.0

# --- [파이썬 전용 PD 제어 파라미터 세팅] ---
# 목표 고정 포즈 (Radian 단위) -> [base, shoulder, elbow, wrist, gripper_base]
target_pose = np.array([0.0, 0.5, -0.5, 0.0, 0.0])

# 각 관절별 파이썬 비례 게인(Kp) 및 미분 게인(Kd) 설정
# 하위 가벼운 링크로 갈수록 게인을 유동적으로 낮추어 떨림을 방지합니다.
Kp = np.array([150.0, 150.0, 100.0, 50.0, 30.0])
Kd = np.array([10.0,  12.0,  8.0,   4.0,  2.0])

# 4. 메인 시뮬레이션 및 렌더링 루프
while not glfw.window_should_close(window):
    time_prev = data.time

    # 1/60초 타임스텝 동기화 제어 루프
    while (data.time - time_prev) < (1.0 / 60.0):
        
        # [핵심] 파이썬 레벨에서 직접 5개 관절의 PD 제어 토크 연산
        for i in range(5):
            current_qpos = data.qpos[i]  # 현재 관절 각도
            current_qvel = data.qvel[i]  # 현재 관절 속도
            
            # 오차 계산 (위치 오차, 속도 오차)
            error_pos = target_pose[i] - current_qpos
            error_vel = 0.0 - current_qvel  # 목표 속도는 0 (정지 상태 지향)
            
            # PD 제어 수식 적용: Torque = Kp * error_pos + Kd * error_vel
            target_torque = (Kp[i] * error_pos) + (Kd[i] * error_vel)
            
            # 연산된 토크 값을 액추에이터 입력에 다이렉트로 매핑
            data.ctrl[i] = target_torque
        
        # 물리 엔진 계산 수행 (입력된 토크를 기반으로 dynamic 전개)
        mj.mj_step(model, data)

    # 60Hz 화면 렌더링
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

# 종료 처리
glfw.terminate()