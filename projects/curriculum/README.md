# JDCobot200 커리큘럼 프로젝트

각 서브프로젝트는 `hardware_setup`에서 확인한 기본 기능을 조합하여 하나의 로봇
기능을 완성합니다. 공통 통신·설정 코드는 최상위 `jdcobot200` 패키지를 사용하고,
각 프로젝트 폴더에는 학습할 실행 코드와 설명만 둡니다.

## 서브프로젝트

1. [`01_servo_homing_and_gripper`](01_servo_homing_and_gripper/README.md)
   - ID 1~6 전체 homing
   - gripper 열기, 닫기, 원점 및 jog 제어
2. [`02_mujoco_basics`](02_mujoco_basics/README.md)
   - canonical MJCF 모델과 관절 순서 검사
   - MuJoCo position actuator 및 gripper 수동 제어
3. [`03_mujoco_fk_safety`](03_mujoco_fk_safety/README.md)
   - MuJoCo FK로 TCP 위치와 방향 계산
   - 자기충돌·바닥 침범 및 전체 관절 경로 검사
4. [`04_sim2real_fk`](04_sim2real_fk/README.md)
   - 실제 관절과 MuJoCo 관절의 회전 방향을 비교하여 설정 보정
   - 안전검사 후 단일 관절 또는 키보드 수동 동기 제어
5. [`05_mujoco_ik`](05_mujoco_ik/README.md)
   - MuJoCo Jacobian으로 position IK 계산
   - FK 재검증과 전체 관절 경로 안전검사
6. [`06_sim2real_ik`](06_sim2real_ik/README.md)
   - 실제 encoder 자세에서 Cartesian 목표 계산
   - 검증된 경로만 작업자 승인 후 실제 로봇에 적용
7. [`07_sim2real_pick_and_place`](07_sim2real_pick_and_place/README.md)
   - 로봇을 손으로 움직여 pick/place Cartesian 좌표 티칭
   - FK 안전검사를 통과한 TCP·관절·encoder 자세 기록 및 실제 루틴 실행
8. [`08_synthetic_pick_and_place`](08_synthetic_pick_and_place/README.md)
   - MuJoCo IK expert로 고정 블록의 합성 pick-and-place 실행
   - 전체 IK/FK 경로와 최종 place 성공 여부 자동 검증
9. [`09_act_dataset_generation`](09_act_dataset_generation/README.md)
   - MuJoCo RGB·관절 상태·expert action을 동기화하여 episode로 기록
   - 무작위 합성 데이터 생성과 manifest/배열 검사 및 영상 재생
10. [`10_act_training_and_mujoco_eval`](10_act_training_and_mujoco_eval/README.md)
    - 09 원본을 LeRobot Dataset v3 형식으로 변환
    - ACT 학습과 expert 정보 없는 MuJoCo closed-loop 평가
