import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

def jdcobot200_forward_kinematics(joint_angles):
    """
    앞서 검증을 마친 jdcobot200 표준 DH 순방향 기구학 수식 (실시간 마커용)
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

def smoothstep(t):
    """
    3차 다항식 기반 가감속 프로파일 수식 (t: 0.0 ~ 1.0 입력 -> 0.0 ~ 1.0 출력)
    시작과 끝 지점에서 부드럽게 속도가 0으로 수렴합니다.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def main():
    xml_name = "scene.xml"
    xml_path = os.path.join(os.path.dirname(__file__), xml_name) if "__file__" in globals() else xml_name
    if not os.path.exists(xml_path):
        print(f"에러: [{xml_path}] 파일을 찾을 수 없습니다.")
        return
        
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    if not glfw.init():
        return
    window = glfw.create_window(1024, 768, "jdcobot200 Smooth Acceleration Pick & Place", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    scene = mj.MjvScene(model, maxgeom=2000)
    cam = mj.MjvCamera()
    vopt = mj.MjvOption()
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    cam.lookat[:] = [0.0, 0.0, 0.15]
    cam.distance = 1.2
    cam.azimuth = 135.0
    cam.elevation = -25.0
    
    # 픽앤플레이스 핵심 시퀀스 타겟 포즈
    poses = {
        "HOME":         np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "A_APPROACH":   np.array([0.5, 0.3, 0.4, -0.3, 0.0]),  # A지점 상공
        "A_PICK":       np.array([0.5, 0.6, 0.8, -0.6, 0.0]),  # A지점 하강
        "B_APPROACH":   np.array([-0.5, 0.3, 0.4, -0.3, 0.0]), # B지점 상공
        "B_PLACE":      np.array([-0.5, 0.6, 0.8, -0.6, 0.0]), # B지점 하강
    }
    
    sequence = ["A_APPROACH", "A_PICK", "A_APPROACH", "B_APPROACH", "B_PLACE", "B_APPROACH"]
    current_step_idx = 0
    
    # 💡 [가감속 튜닝 변수] 
    # 포즈 이동에 걸리는 주기를 설정합니다. 이 시간을 늘릴수록 로봇이 천천히 부드럽게 움직입니다.
    state_duration = 2.0  # 한 포즈당 도달 및 대기 유지 시간 (초)
    move_duration = 1.2   # 순수하게 포즈 이동(가감속 보간)이 일어나는 시간 (초)
    
    last_state_switch_time = 0.0
    
    # 이전 단계의 시작 포즈를 기억하기 위한 변수 (초기값은 HOME)
    start_joint_angles = poses["HOME"].copy()
    
    actuator_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
    actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
    
    print("\n" + "="*60)
    print(" [가감속 보간(Smoothstep) 기반 픽앤플레이스 시작]")
    print(f" - 전체 주행 주기: {state_duration}초 | 순수 이동(가감속) 시간: {move_duration}초")
    print(" - 급격한 토크 튐이 사라지고 실제 프리미엄 서보모터처럼 구동됩니다.")
    print("="*60 + "\n")

    while not glfw.window_should_close(window):
        time_sim = data.time
        
        # 1. 상태 머신 전환 타이밍 체크
        if time_sim - last_state_switch_time > state_duration:
            # 다음 포즈로 넘어가기 전, 현재의 실제 각도를 다음 출발 각도로 고정
            start_joint_angles = np.array([data.qpos[model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]] for n in actuator_names])
            
            current_step_idx = (current_step_idx + 1) % len(sequence)
            last_state_switch_time = time_sim
            print(f">> State 변경 -> [{sequence[current_step_idx]}] (Time: {time_sim:.1f}s)")
            
        # 2. 현재 도달해야 할 최종 목표 포즈 획득
        current_state_name = sequence[current_step_idx]
        final_target_angles = poses[current_state_name]
        
        # 3. 💡 [핵심: 실시간 궤적 가감속 생성]
        # 현재 단계가 시작된 후 흘러간 시간 측정
        elapsed_time = time_sim - last_state_switch_time
        
        # 이동 시간(move_duration) 동안은 가감속 보간을 하고, 그 이후에는 목표점에 고정대기
        if elapsed_time < move_duration:
            # 진행도 계산 (0.0 ~ 1.0)
            progress = elapsed_time / move_duration
            # Smoothstep 삼차곡선 가중치 변환
            smooth_factor = smoothstep(progress)
            # 출발지에서 목적지까지 부드럽게 중간 징검다리 각도 보간(Linear Interpolation + Smoothstep)
            current_target_angles = start_joint_angles + (final_target_angles - start_joint_angles) * smooth_factor
        else:
            # 지정된 가감속 시간이 끝나면 완전한 최종 목표 각도 고정 홀딩
            current_target_angles = final_target_angles
        
        # 4. 부드럽게 쪼개진 제어 각도를 액추에이터 주입
        for idx, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = current_target_angles[idx]
            
        # 5. 하위 물리 타임스텝(dt) 루프 작동
        time_prev = data.time
        while (data.time - time_prev) < (1.0 / 60.0):
            mj.mj_step(model, data)
            
        # 6. 실시간 DH 연산 및 렌더링 오버레이
        present_angles = [data.qpos[model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]] for n in actuator_names]
        T_dh = jdcobot200_forward_kinematics(present_angles)
        dh_xyz = T_dh[0:3, 3]
        
        viewport = glfw.get_framebuffer_size(window)
        mj_viewport = mj.MjrRect(0, 0, viewport[0], viewport[1])
        mj.mjv_updateScene(model, data, vopt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.012, 0.012, 0.012],
                pos=dh_xyz,
                mat=np.eye(3).flatten(),
                rgba=[0.0, 1.0, 0.0, 0.5]
            )
            scene.ngeom += 1
            
        mj.mjr_render(mj_viewport, scene, ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
    glfw.terminate()

if __name__ == "__main__":
    main()