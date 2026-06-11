import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull  # 맨 바깥 표면을 묶어주는 라이브러리

# 1. 로봇 링크 길이 설정 (이전과 동일)
L1, L2, L3, L4 = 0.080, 0.120, 0.120, 0.060

# 조인트 가동 범위 설정
joint_limits = [
    [-np.deg2rad(150), np.deg2rad(150)], # q1 (Base)
    [-np.deg2rad(60),  np.deg2rad(90)],  # q2 (Shoulder)
    [-np.deg2rad(90),  np.deg2rad(90)],  # q3 (Elbow)
    [-np.deg2rad(90),  np.deg2rad(90)]   # q4 (Wrist)
]

def compute_forward_kinematics(q1, q2, q3, q4):
    angle_shoulder = q2
    angle_elbow    = q2 + q3
    angle_wrist    = q2 + q3 + q4
    
    z = L1 + L2 * np.sin(angle_shoulder) + L3 * np.sin(angle_elbow) + L4 * np.sin(angle_wrist)
    r = L2 * np.cos(angle_shoulder) + L3 * np.cos(angle_elbow) + L4 * np.cos(angle_wrist)
    x = r * np.cos(q1)
    y = r * np.sin(q1)
    return [x, y, z, r]

# 2. 샘플링 연산
num_samples = 10000
points = []
r_z_profile = [] # 수직 단면용 배열

for _ in range(num_samples):
    q1 = np.random.uniform(joint_limits[0][0], joint_limits[0][1])
    q2 = np.random.uniform(joint_limits[1][0], joint_limits[1][1])
    q3 = np.random.uniform(joint_limits[2][0], joint_limits[2][1])
    q4 = np.random.uniform(joint_limits[3][0], joint_limits[3][1])
    
    x, y, z, r = compute_forward_kinematics(q1, q2, q3, q4)
    points.append([x, y, z])
    r_z_profile.append([r, z])

points = np.array(points)
r_z_profile = np.array(r_z_profile)

# =====================================================================
# 3. 개선된 직관적 시각화 창 (3D 덩어리 표면 + 2D 수직 단면 프로필)
# =====================================================================
fig = plt.figure(figsize=(15, 7))

# --- [왼쪽 화면] 3D 메쉬 표면 시각화 ---
ax1 = fig.add_subplot(121, projection='3d')
# 맨 바깥 표면의 외곽 점들을 수학적으로 찾아 삼각형 면으로 이어줌
hull = ConvexHull(points)
ax1.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices, 
                 cmap='coolwarm', alpha=0.6, edgecolor='none')
ax1.scatter([0], [0], [0], color='black', s=100, label='Base')
ax1.set_title("1. 3D Solid Surface Workspace", fontsize=13)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.set_box_aspect([1, 1, 0.8])

# --- [오른쪽 화면] 2D 로봇 옆면 단면 가동 영역 ---
ax2 = fig.add_subplot(122)
# 수직 단면의 외곽선(껍질)만 추출
hull_2d = ConvexHull(r_z_profile)
# 모든 데이터 점들을 연하게 깔아주고
ax2.scatter(r_z_profile[:, 0], r_z_profile[:, 1], c='lightgray', s=1, alpha=0.5)
# 도달 가능한 한계선 경계를 빨간선으로 명확히 표현
for simplex in hull_2d.simplices:
    ax2.plot(r_z_profile[simplex, 0], r_z_profile[simplex, 1], 'r-', lw=2)

ax2.plot(r_z_profile[hull_2d.vertices, 0], r_z_profile[hull_2d.vertices, 1], 'ro', markersize=4)
ax2.axvline(0, color='black', linestyle='--', alpha=0.5) # 로봇 중심선
ax2.scatter([0], [L1], color='blue', s=80, label='Shoulder Joint')
ax2.set_title("2. Robot Side Profile (Reach & Dead zone Limit)", fontsize=13)
ax2.set_xlabel("Horizontal Reach (R, meters)")
ax2.set_ylabel("Vertical Height (Z, meters)")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()