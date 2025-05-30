import numpy as np
import open3d as o3d
import cv2
from copy import deepcopy
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy.spatial import distance
import itertools

iou_cost_history = []
reprojection_cost_history = []



def perform_icp(camera_frame, pcd_frame, source, target, threshold, init_transform, visualize):
    # ICP 수행
    print("Performing ICP...")

    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print("ICP fitness:", reg_icp.fitness)
    print("ICP inlier RMSE:", reg_icp.inlier_rmse)
    print("ICP transformation:\n", reg_icp.transformation)
    source_cp = deepcopy(source)  # 원본 포인트 클라우드 복사
    source_cp.transform(reg_icp.transformation)  # 변환 적용
    pcd_frame.transform(reg_icp.transformation)  # 포인트 클라우드 변환
    if visualize:
        o3d.visualization.draw_geometries([camera_frame, pcd_frame, source_cp, target], window_name="ICP Result")
    # 변환 행렬 반환
    # print("Transformation matrix:\n", reg_icp.transformation)
    return reg_icp.transformation



def pcd_projection(img,
                   pcd,
                   intrinsic,
                   distortion,
                   transform,
                   point_size=3,
                   color=None,
                   num_bins=6,
                   colormap=cv2.COLORMAP_TURBO,
                   gamma=0.9):
    """
    LiDAR 포인트 클라우드를 카메라 영상에 투영하고,
    거리별(깊이별) 색상을 계단식으로 부여한다.

    Parameters
    ----------
    img : np.ndarray (H, W, 3, BGR)
    pcd : open3d.geometry.PointCloud
    intrinsic : (3,3) np.ndarray
    distortion : (N,) np.ndarray
    transform : (4,4) np.ndarray
    point_size : int, plotted circle radius
    color : None 혹은 (3,) 또는 (N,3) uint8  [B,G,R]
    num_bins : int, 거리 구간 수 (색 구분 단계)
    colormap : OpenCV colormap ID (ex. cv2.COLORMAP_TURBO)
    gamma : float, 0 < gamma ≤ 3, 1보다 작으면 가까운 거리 강조

    Returns
    -------
    img_out : np.ndarray
    valid_points : (M,2) np.ndarray  이미지 좌표
    """
    if not pcd.has_points():
        return img, []

    # ── 1. 클라우드 변환 ───────────────────────────────────────────────
    pts = np.asarray(pcd.points)
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    pts_cam = (transform @ pts_h.T).T[:, :3]

    # 카메라 앞쪽 Z>0
    mask_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask_front]
    if pts_cam.size == 0:
        return img, []

    # ── 2. 거리 계산 및 계단식 정규화 ─────────────────────────────────
    dist = np.linalg.norm(pts_cam, axis=1)
    # gamma 보정으로 근거리 분해능 향상
    dist_gamma = dist ** gamma

    # 0~255 정규화 후 binning
    dist_norm = cv2.normalize(dist_gamma, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # 계단식 효과: 0~255를 num_bins 구간으로 나눠 중앙값으로 스냅
    step = 256 // num_bins
    dist_quant = (dist_norm // step) * step + step // 2
    colors_map = cv2.applyColorMap(dist_quant, colormap)

    # ── 3. 사용자 지정 색상 우선 적용 ────────────────────────────────
    if color is None:
        colors = colors_map
    else:
        color = np.asarray(color, dtype=np.uint8)
        colors = np.tile(color, (len(pts_cam), 1)) if color.ndim == 1 else color

    # ── 4. 3‑D → 2‑D 투영 ────────────────────────────────────────────
    proj, _ = cv2.projectPoints(pts_cam,
                                rvec=np.zeros((3, 1), np.float32),
                                tvec=np.zeros((3, 1), np.float32),
                                cameraMatrix=intrinsic,
                                distCoeffs=distortion)
    proj = proj.reshape(-1, 2)

    h, w = img.shape[:2]
    in_img = (proj[:, 0] >= 0) & (proj[:, 0] < w) & \
             (proj[:, 1] >= 0) & (proj[:, 1] < h)

    pts_2d = proj[in_img].astype(int)
    pts_color = colors[in_img]

    # ── 5. 렌더링 ────────────────────────────────────────────────────
    for (x, y), c in zip(pts_2d, pts_color):
        bgr = tuple(int(v) for v in c.squeeze())   # c: (1,3) → (3,)
        cv2.circle(img, (x, y), point_size, bgr, -1)

    return img, pts_2d


def draw_contour(img, corners, rvec, tvec, intrinsic, distortion):
    # 체커보드 코너를 2D 이미지 평면으로 투영
    projected_corners, _ = cv2.projectPoints(corners, rvec, tvec, intrinsic, distortion)
    projected_corners = projected_corners.reshape(-1, 2)  # (N, 1, 2) -> (N, 2)

    # 이미지에 체커보드 코너 그리기
    for pt_idx, pt in enumerate(projected_corners):
        # print(f"Point {pt_idx}: {pt}, type: {type(pt)}")
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)  # 초록색 점

    return img, projected_corners

def filter_points_inside_chessboard(lidar_pcd, radius=0.8):
    """
    체스보드 영역 내에 포함되지 않는 LiDAR 포인트를 필터링하는 함수.
    
    Args:
        lidar_pcd: LiDAR 포인트 클라우드 (Open3D PointCloud 객체)
    
    Returns:
        filtered_pcd: 체스보드 영역 내부의 포인트 클라우드
    """
    pcd_centroid = np.mean(np.asarray(lidar_pcd.points), axis=0)  # 포인트 클라우드의 중심 계산
    # filter 기준: 체스보드 영역의 중심을 기준으로 반지름 1.5m 이내의 포인트
    
    filtered_points = []
    for point in lidar_pcd.points:
        distance = np.linalg.norm(point - pcd_centroid)
        if distance < radius:
            filtered_points.append(point)
    # 필터링된 포인트로 새로운 포인트 클라우드 생성
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.paint_uniform_color([0, 1, 0])

    return filtered_pcd

def sort_points_by_angle(points, center):
    """점들을 중심에서 방위각으로 정렬"""
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def reprojection_error(params, pcd_list, corner_list, intrinsic, distortion, image_T_list):
    residuals = []
    for pcd_pts, corner_pts, image_T in zip(pcd_list, corner_list, image_T_list):
        # 파라미터 설정
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)
        image_R = image_T[:3, :3]
        image_rvec = cv2.Rodrigues(image_R)[0]
        image_tvec = image_T[:3, 3].reshape(3, 1)

        # LiDAR 포인트 투영
        pcd_np = np.asarray(pcd_pts.points)
        projected_pcd, _ = cv2.projectPoints(pcd_np, rvec, tvec, intrinsic, distortion)
        projected_pcd = projected_pcd.reshape(-1, 2)

        # 체스보드 꼭지점 투영
        projected_corner, _ = cv2.projectPoints(corner_pts, image_rvec, image_tvec, intrinsic, distortion)
        projected_corner = projected_corner.reshape(-1, 2)

        # Convex Hull 계산
        hull = cv2.convexHull(projected_pcd.astype(np.float32)).reshape(-1, 2)

        # Convex Hull 중심과 점들 정렬
        hull_center = np.mean(hull, axis=0)
        sorted_hull = sort_points_by_angle(hull, hull_center)

        # 체스보드 꼭지점 정렬
        corner_center = np.mean(projected_corner, axis=0)
        sorted_corners = sort_points_by_angle(projected_corner, corner_center)

        # Convex Hull에서 체스보드 꼭지점에 대응하는 점 찾기
        matched_pts = []
        for corner in sorted_corners:
            dists = np.linalg.norm(sorted_hull - corner, axis=1)
            nearest_idx = np.argmin(dists)
            matched_pts.append(sorted_hull[nearest_idx])
        matched_pts = np.array(matched_pts)

        # 오차 계산
        distances = np.linalg.norm(sorted_corners - matched_pts, axis=1) / 4
        residuals.extend(distances)

    residuals = np.array(residuals)
    reprojection_cost_history.append(np.sum(residuals**2))

    return residuals

def compute_iou(lidar_poly, camera_poly):
    area_lidar = cv2.contourArea(lidar_poly)
    area_camera = cv2.contourArea(camera_poly)
    ret, intersection = cv2.intersectConvexConvex(lidar_poly, camera_poly)
    area_intersection = cv2.contourArea(intersection) if intersection is not None and len(intersection) > 0 else 0
    union = area_lidar + area_camera - area_intersection
    return area_intersection / (union + 1e-6)

def joint_iou_loss(params, pcd_list, corner_list, intrinsic, distortion, image_T_list):
    residuals = []
    for pcd_pts, corner_pts, imgae_T in zip(pcd_list, corner_list, image_T_list):
        # params: 6차원 extrinsic 파라미터 (회전, 이동)
        rvec = params[:3].reshape(3,1)
        tvec = params[3:6].reshape(3,1)

        image_R = imgae_T[:3,:3]
        image_rvec = cv2.Rodrigues(image_R)[0]
        image_tvec = imgae_T[:3,3].reshape(3,1)
        # 여기서는 각 이미지에 대해 이미 계산된 projected points(투영 결과)를 활용하거나,
        # 만약 재투영이 필요하다면 해당 LiDAR 포인트 클라우드를 params로 재투영하는 과정을 포함해야 합니다.
        # 예: projected_pts, _ = cv2.projectPoints(lidar_pts, rvec, tvec, intrinsic, distortion)
        pcd_np = np.asarray(pcd_pts.points)
        # corner_np = np.asarray(corner_pts.points)
        projected_pcd, _ = cv2.projectPoints(pcd_np, rvec, tvec, intrinsic, distortion)
        projected_corner, _ = cv2.projectPoints(corner_pts, image_rvec, image_tvec, intrinsic, distortion)
        
        # 예시로, convex hull을 구하고 IoU를 계산한다고 가정하면:
        hull = cv2.convexHull(projected_pcd.astype(np.float32)).reshape(-1,2)
        # print("Hull length:", len(hull))    
        iou = compute_iou(hull, projected_corner)
        iou_error = 1 - iou
        # print("IOU error:", iou_error)
        residuals.append(iou_error)

    residuals = np.array(residuals)
    iou_cost_history.append(np.sum(residuals**2))

    return residuals

def draw_iou(img, projected_points, contour_points):
    """
    IoU를 시각화하기 위한 함수
    Args:
        img: 이미지 (numpy 배열)
        lidar_poly: LiDAR 다각형 (numpy 배열)
        camera_poly: 카메라 다각형 (numpy 배열)
    Returns:
        img: IoU가 시각화된 이미지
    """
    if projected_points is None:
        return
    
    hull = cv2.convexHull(projected_points.astype(np.float32))
    if hull is None:
        return
    hull = hull.reshape(-1,2).astype(np.int32)
    # LiDAR 다각형 그리기
    contour_points = contour_points.astype(np.int32)
    cv2.polylines(img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록색
    # 카메라 다각형 그리기
    cv2.polylines(img, [contour_points], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨간색
    return img


def plot_cost_history(residuals, title="Residuals over Iterations"):
    """
    Residuals를 시각화하는 함수
    Args:
        residuals: Residual 값 리스트
        title: 그래프 제목
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(residuals, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.grid()
    plt.show()