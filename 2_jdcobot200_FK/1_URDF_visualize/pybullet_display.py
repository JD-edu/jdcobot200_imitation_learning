import pybullet as p
import pybullet_data
import time
import os

# 1. PyBullet 시뮬레이터 초기화 (GUI 모드)
physicsClient = p.connect(p.GUI)

# 데이터 경로 설정 (바닥 판 등을 불러오기 위함)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 중력 설정 및 바닥 추가
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# 2. 내 로봇 URDF 로드하기
# 코드와 같은 폴더에 URDF 파일과 assets(stl) 폴더가 있다고 가정합니다.
urdf_path = "jdcobot200.urdf"  # 파일명에 맞게 수정하세요

try:
    # flags=p.URDF_USE_INERTIA_FROM_FILE를 주면 작성한 mass/inertia가 정상 작동하는지 검증 가능합니다.
    robotId = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    print(f"성공적으로 {urdf_path} 모델을 로드했습니다.")
except Exception as e:
    print(f"URDF 로드 실패: {e}\n파일 경로 및 STL 매쉬 파일 경로를 확인하세요.")
    p.disconnect()
    exit()

# 3. 조인트 정보 파악 및 GUI 슬라이더 생성
numJoints = p.getNumJoints(robotId)
joint_indices = []

print("\n--- [조인트 정보 목록] ---")
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    jointName = jointInfo[1].decode("utf-8")
    jointType = jointInfo[2]
    
    # 회전축(revolute)이나 프리즘(prismatic) 관절만 제어 슬라이더 생성
    if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        lowerLimit = jointInfo[8]
        upperLimit = jointInfo[9]
        
        # 리미트가 안 잡혀있을 경우 기본값 세팅
        if lowerLimit >= upperLimit:
            lowerLimit = -3.14159
            upperLimit = 3.14159
            
        # GUI 우측에 슬라이더 바 추가
        paramId = p.addUserDebugParameter(jointName, lowerLimit, upperLimit, 0.0)
        joint_indices.append((i, paramId, jointName))
        print(f"ID: {i} | 이름: {jointName} | 범위: {lowerLimit:.2f} ~ {upperLimit:.2f}")

# 4. 시뮬레이션 및 슬라이더 동기화 루프
print("\n시뮬레이터가 실행 중입니다. 우측 패널의 슬라이더를 움직여 축을 확인하세요.")

# 카메라 초점 조정 (로봇이 작으므로 가깝게 줌인)
p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.1])

# 마우스로 링크 좌표축을 볼 수 있도록 설정 켬
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

try:
    while True:
        # 1) 우측 GUI 슬라이더 값을 읽어서 가상 로봇 관절에 주입
        for jointIndex, paramId, name in joint_indices:
            targetAngle = p.readUserDebugParameter(paramId)
            p.resetJointState(robotId, jointIndex, targetAngle)
            
        # 2) 물리 엔진 1스텝 전진
        p.stepSimulation()
        time.sleep(1./240.)
        
except KeyboardInterrupt:
    print("시뮬레이션을 종료합니다.")
finally:
    p.disconnect()