import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

def get_dh_matrix(a, alpha, d, theta):
    """
    표준 DH 파라미터 4가지 변수를 받아 4x4 동차 변환 행렬을 반환하는 함수
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [ 0,       sa,       ca,      d],
        [ 0,        0,        0,      1]
    ])
    return T

def jdcobot200_forward_kinematics(joint_angles):
    """
    jdcobot200의 5개 관절 각도(라디안)를 받아 말단(End-Effector)의 위치와 자세를 계산
    """
    q1, q2, q3, q4, q5 = joint_angles
    
    # [제공된 XML 구조와 축 관계가 100% 동기화된 정밀 Standard DH 테이블]
    # 각 행의 구조: [a, alpha, d, theta_offset]
    dh_table = [
        [0.0,      np.pi/2,  0.0537,  0.0],       # 1. Base (수직축에서 수평축으로 전환)
        [0.1352,   0.0,      0.06146, np.pi/2],   # 2. Shoulder (+90도 수평 정렬 오프셋 강제)
        [0.1352,   0.0,      0.0,     0.0],       # 3. Elbow 
        [0.0,      -np.pi/2, 0.0,     -np.pi/2],  # 4. Wrist Pitch (손목 정렬을 위한 오프셋)
        [0.0,      0.0,      0.0575,  0.0]        # 5. Wrist Roll -> End-Effector Tip (gripper_assembly)
    ]
    
    T_total = np.eye(4)
    
    for i, (a, alpha, d, theta_offset) in enumerate(dh_table):
        current_theta = joint_angles[i] + theta_offset
        T_i = get_dh_matrix(a, alpha, d, current_theta)
        T_total = np.dot(T_total, T_i)
        
    return T_total

def main():
    # 1. MuJoCo 모델 로딩 (동일 폴더의 xml 이름 확인 후 매칭)
    # 제공해주신 XML 내용을 'scene.xml' 또는 'jdcobot200.xml'로 저장한 후 아래 이름을 맞춰주세요.
    xml_name = "scene.xml" 
    xml_path = os.path.join(os.path.dirname(__file__), xml_name) if "__file__" in globals() else xml_name
    
    if not os.path.exists(xml_path):
        print(f"에러: [{xml_path}] 파일을 찾을 수 없습니다. 파일명을 확인하고 소스코드와 같은 폴더에 놓아주세요.")
        return
        
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    # 2. GLFW 그래픽 디스플레이 창 생성
    if not glfw.init():
        return
    window = glfw.create_window(1024, 768, "jdcobot200 DH FK Real-time Overlay Verification", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # 3. 시각화 씬 및 카메라 객체 파이프라인 초기화
    scene = mj.MjvScene(model, maxgeom=2000)
    cam = mj.MjvCamera()
    vopt = mj.MjvOption()
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # 로봇암 관찰에 가장 직관적이고 편안한 카메라 각도 셋팅
    cam.lookat[:] = [0.0, 0.0, 0.15]
    cam.distance = 0.55
    cam.azimuth = 135.0
    cam.elevation = -20.0
    
    # XML 내부 조인트 명칭 바인딩 및 주소 파싱
    joint_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
    joint_qpos_adr = []
    for n in joint_names:
        try:
            j_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)
            joint_qpos_adr.append(model.jnt_qposadr[j_id])
        except Exception:
            print(f"경고: XML 내에 '{n}' 조인트가 존재하지 않거나 이름이 일치하지 않습니다.")
            
    # 기구학 검증 기준이 될 말단(End-Effector) 바디의 ID 추출
    ee_body_name = "gripper_assembly"
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
    
    start_time = glfw.get_time()

    while not glfw.window_should_close(window):
        # 액추에이터 고정 부하에 제어 신호가 굳는 현상을 차단하기 위해 GLFW 절대 시간 흐름을 모션 틱으로 활용
        time_now = glfw.get_time() - start_time
        
        # 4. 실시간 좌표 변화 검증을 위해 5축에 위상이 다른 다이내믹 사인파 각도 생성
        q1 = 0.8 * np.sin(time_now * 1.0)
        q2 = 0.5 * np.sin(time_now * 1.3)
        q3 = 0.6 * np.sin(time_now * 1.1)
        q4 = 0.4 * np.sin(time_now * 0.7)
        q5 = 1.0 * np.sin(time_now * 1.8)
        current_angles = [q1, q2, q3, q4, q5]
        
        # 5. [핵심] 모터 역토크 고정 간섭을 우회하기 위해 qpos 직접 대입 후 
        # mj_step() 대신 순수 링크 기하학 배열만 정렬하는 mj_kinematics 단독 호출
        for idx, adr in enumerate(joint_qpos_adr):
            data.qpos[adr] = current_angles[idx]
        mj.mj_kinematics(model, data)
        
        # 6. 파이썬 상에 작성된 DH 파라미터 공식을 풀어서 (X, Y, Z) 좌표 획득
        T_dh = jdcobot200_forward_kinematics(current_angles)
        dh_xyz = T_dh[0:3, 3]
        
        # 7. MuJoCo 시뮬레이터 커널 엔진이 물리적으로 갱신한 실제 gripper_assembly 바디의 원점 좌표 획득
        mujoco_xyz = data.xpos[ee_id]
        
        # 수학적 공식 결과와 가상 모델 손끝 좌표 간의 변위 오차 연산 (mm 단위)
        error_dist = np.linalg.norm(mujoco_xyz - dh_xyz) * 1000
        
        # 콘솔창에 0.5초마다 실시간 오차 상태 브리핑 출력
        if int(time_now * 20) % 10 == 0:
            print(f"Time: {time_now:.1f}s | DH 계산: {np.round(dh_xyz, 3)} | MuJoCo 감지: {np.round(mujoco_xyz, 3)} | 실시간 오차: {error_dist:.3f} mm")
            
        # 8. 백그라운드 뷰포트 그래픽스 초기화 및 갱신
        viewport = glfw.get_framebuffer_size(window)
        mj_viewport = mj.MjrRect(0, 0, viewport[0], viewport[1])
        
        # 로봇 외형 정보 업데이트
        mj.mjv_updateScene(model, data, vopt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        
        # 9. [비주얼 검증 코어] DH 연산 공식 결과 자리에 불투명도 50%의 '빨간색 반투명 추적 구체' 실시간 생성 및 오버레이
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.015, 0.015, 0.015], # 지름 1.5cm 오버레이 볼 크기
                pos=dh_xyz,                 # DH 공식 결과 좌표
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 0.5]   # 반투명 레드 마커
            )
            scene.ngeom += 1 # 렌더링 큐 카운트 증가
            
        # 화면에 그리기 버퍼 교체
        mj.mjr_render(mj_viewport, scene, ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
    glfw.terminate()

if __name__ == "__main__":
    main()