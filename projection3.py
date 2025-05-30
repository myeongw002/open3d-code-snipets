import cv2
import numpy as np
import os
import open3d as o3d
from utils import pcd_projection

# 경로 지정
image_path = "0424-0515/2025-05-17/cam1/2025-05-17_13-03-50.png"
pointcloud_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/roi/1748617011836370.pcd"
result_path = "/home/myungw00/ROS/gm/Code/data/scripts/results/projection images/sample"
intrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/intrinsic1.csv"
extrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/iou_optimized_transform1.txt"

# 상대 변환 행렬 정의 (쉼표로 구분)
relative_transform = np.eye(4)
R = np.array([
    [ 9.99987855e-01,  9.69194734e-04,  4.83218624e-03],
    [-1.00576085e-03,  9.99970837e-01,  7.57051987e-03],
    [-4.82470801e-03, -7.57528795e-03,  9.99959668e-01]
])
t = np.array([-0.01413879, -0.00463732, 0.04843542])
relative_transform[0:3, 0:3] = R
relative_transform[0:3, 3] = t
print("Relative Transform:\n", relative_transform)
relative_transform_inv = np.linalg.inv(relative_transform)

# 이미지 파일 1개만 사용
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# PCD 파일 1개 불러오기
pcd = o3d.io.read_point_cloud(pointcloud_path)
# PCD에 상대 변환 적용
pcd.transform(relative_transform)

# 카메라 파라미터 로딩
intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
intrinsic = np.array([
    [intrinsic_param[0]/2, intrinsic_param[1], intrinsic_param[2]/2],
    [0.0, intrinsic_param[3]/2, intrinsic_param[4]/2],
    [0.0, 0.0, 1.0]
])
distortion = np.array(intrinsic_param[5:])

extrinsic = np.loadtxt(extrinsic_path, delimiter=',')
if extrinsic.shape == (3, 4):
    # 3x4 행렬인 경우 4x4로 변환
    ext_tmp = np.eye(4)
    ext_tmp[:3, :] = extrinsic
    extrinsic = ext_tmp

print("Camera Matrix:\n", intrinsic)
print("Distortion Coefficients:\n", distortion)
print("Extrinsic Matrix:\n", extrinsic)

optimized_image, _ = pcd_projection(
    image.copy(),
    pcd,
    intrinsic,
    distortion,
    extrinsic,
    point_size=1,
    color=None,
    num_bins=10,
    colormap=cv2.COLORMAP_TURBO,
    gamma=1
)
# 결과 이미지 저장
os.makedirs(result_path, exist_ok=True)
save_path = os.path.join(result_path, "image_pcd.jpg")
cv2.imwrite(save_path, optimized_image)
print(f"Projected image saved at: {save_path}")
