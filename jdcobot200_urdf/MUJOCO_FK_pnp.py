import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

def jdcobot200_forward_kinematics(joint_angles):
    """
    앞서 검증을 마친 jdcobot200 표준 DH 순방향 기구학 수식 (실시간 로그 인쇄용)
    """
    def get_dh_matrix(a, alpha, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [ 0,       sa,       ca,      d],
            [ 0,        0,        0,      1]
        ])
    
    dh_table = [
        [0.0,      np.pi/2,  0.0537,  0.0],
        [0.1352,   0.0,      0.06146, np.pi/2],
        [0.1352,   0.0,      0.0,     0.0],
        [0.0,      -np.pi/2, 0.0,     -np.pi/2],
        [0.0,      0.0,      0.0575,  0.0]
    ]
    T_total = np.eye(4)
    for i, (a, alpha, d, theta_offset) in enumerate(dh_table):
        T_i = get_dh_matrix(a, alpha, d, joint_angles[i] + theta_offset)
        T_total = np.dot(T_total, T_i)
    return T_total

def main():
    # 1. MuJoCo 모델 로딩 (동일 폴더의 scene.xml 경로 참조)
    xml_name = "scene.xml"
    xml_path = os.path.join(os.path.dirname(__file__), xml_name) if "__file__" in globals() else xml_name
    if not os.path.exists(xml_path):
        print(f"에러: [{xml_path}] 파일을 찾을 수 없습니다.")
        return
        
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    # 2. GLFW 그래픽스 인프라 초기화
    if not glfw.init():
        return
    window = glfw.create_window(1024, 768, "jdcobot200 Physical Pick & Place Sequence", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    scene = mj.MjvScene(model, maxgeom=2000)
    cam = mj.MjvCamera()
    vopt = mj.MjvOption()
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # 편안한 관찰 구도 설정
    cam.lookat[:] = [0.0, 0.0, 0.15]
    cam.distance = 0.65
    cam.azimuth = 135.0
    cam.elevation = -25.0
    
    # 3. 픽앤플레이스 핵심 시퀀스 타겟 조인트 포즈 정의 (라디안 단위)
    # [base, shoulder, elbow, wrist_pitch, wrist_roll]
    poses = {
        "HOME":         np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "A_APPROACH":   np.array([0.5, 0.3, 0.4, -0.3, 0.0]),  # A지점 상공 대기
        "A_PICK":       np.array([0.5, 0.6, 0.8, -0.6, 0.0]),  # A지점 하강 (집기)
        "B_APPROACH":   np.array([-0.5, 0.3, 0.4, -0.3, 0.0]), # B지점 상공 대기
        "B_PLACE":      np.array([-0.5, 0.6, 0.8, -0.6, 0.0]), # B지점 하강 (놓기)
    }
    
    # 상태 머신 관리 변수들
    sequence = ["A_APPROACH", "A_PICK", "A_APPROACH", "B_APPROACH", "B_PLACE", "B_APPROACH"]
    current_step_idx = 0
    state_duration = 1.5  # 각 동작 단계별 유지 시간 (1.5초 동안 제어 유지)
    last_state_switch_time = 0.0
    
    # XML 내부 액추에이터 주소 바인딩
    actuator_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
    actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
    
    print("\n" + "="*60)
    print(" [Pick & Place 시퀀스 물리 시뮬레이션 가동]")
    print(" - mj_step()을 활용한 순방향 다이내믹스 제어가 활성화되었습니다.")
    print(" - 각 단계별로 실제 모터의 토크와 질량 제어 프로파일이 반영됩니다.")
    print("="*60 + "\n")

    while not glfw.window_should_close(window):
        time_sim = data.time
        
        # 4. [상태 머신] 정해진 시간이 지나면 다음 시퀀스 단계로 전환
        if time_sim - last_state_switch_time > state_duration:
            current_step_idx = (current_step_idx + 1) % len(sequence)
            last_state_switch_time = time_sim
            print(f">> State 변경 -> 현재 단계: [{sequence[current_step_idx]}] (Sim Time: {time_sim:.1f}s)")
            
        # 5. 현재 단계에 맞는 목표 포즈 획득
        current_state_name = sequence[current_step_idx]
        target_joint_angles = poses[current_state_name]
        
        # 6. [중요] XML에 선언된 <position kp="998.22"...> 액추에이터 플러그인에 목표 각도 주입
        for idx, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = target_joint_angles[idx]
            
        # 7. [핵심] 1/60초 그래픽 프레임 주기 동안 물리 엔진을 정밀 하위 스텝(dt) 단위로 전진
        # 타임스텝 오차를 완벽히 억제하기 위해 루프로 시간 적분 처리
        time_prev = data.time
        while (data.time - time_prev) < (1.0 / 60.0):
            mj.mj_step(model, data)
            
        # 8. 실시간 DH 기구학 수식 교차 검증용 현재 모터 피드백 각도 추출
        present_angles = [data.qpos[model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]] for n in actuator_names]
        T_dh = jdcobot200_forward_kinematics(present_angles)
        dh_xyz = T_dh[0:3, 3]
        
        # 9. 화면 버퍼 갱신 및 오버레이 렌더링
        viewport = glfw.get_framebuffer_size(window)
        mj_viewport = mj.MjrRect(0, 0, viewport[0], viewport[1])
        
        mj.mjv_updateScene(model, data, vopt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        
        # 실시간 DH 손끝 계산 위치에 반투명 녹색 마커 구체 오버레이 생성
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.012, 0.012, 0.012],
                pos=dh_xyz,
                mat=np.eye(3).flatten(),
                rgba=[0.0, 1.0, 0.0, 0.5] # 50% 반투명 그린 구체
            )
            scene.ngeom += 1
            
        mj.mjr_render(mj_viewport, scene, ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
    glfw.terminate()

if __name__ == "__main__":
    main()