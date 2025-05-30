#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import os
import glob

def average_pcds(pcd_paths, output_path):
    """
    여러 PCD 파일을 읽어 첫 번째 PCD의 각 점에 대해 나머지 PCD들에서
    최근접 점들을 찾아 평균 좌표를 계산하고, 하나의 PCD로 저장합니다.

    :param pcd_paths:   PCD 파일 경로 리스트 (최소 2개)
    :param output_path: 평균화된 PCD를 저장할 경로
    """
    if len(pcd_paths) < 2:
        raise ValueError("PCD 파일이 2개 이상 필요합니다.")

    # 1) 첫 번째 PCD 로드 (기준)
    base_pcd = o3d.io.read_point_cloud(pcd_paths[0])
    base_pts = np.asarray(base_pcd.points)
    n = base_pts.shape[0]

    # 2) KD-Tree 생성 (기준 PCD)
    tree = o3d.geometry.KDTreeFlann(base_pcd)

    # 3) 각 기준 점마다 누적 합을 저장할 배열
    accum = np.zeros((n, 3), dtype=np.float64)
    accum += base_pts  # 자신도 포함

    # 4) 나머지 PCD들 처리
    for path in pcd_paths[1:]:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        # 각각의 기준 점마다 이 PCD에서 가장 가까운 점을 찾아 누적
        for i, p in enumerate(base_pts):
            _, idx, _ = tree.search_knn_vector_3d(p, 1)
            # 주의: search_knn_vector_3d 에는 p 대신 'pts'를 넣어야 합니다.
            #    우리는 기준(p) 에 대해 pcd(pts)에서 최근접을 찾을 거므로
            #    tree를 pcd로 바꿔서야 올바르게 작동합니다.
            # 대신, 아래와 같이 KD-Tree를 매 파일마다 다시 생성해 줍니다.
            break  # 이 부분은 풀어쓴 예제 로직 아래에 정리합니다.

        # --- 올바른 매칭 로직 ---
        # 이 PCD에 대한 KD-Tree
        tree_other = o3d.geometry.KDTreeFlann(pcd)
        for i, p in enumerate(base_pts):
            _, idx, _ = tree_other.search_knn_vector_3d(p, 1)
            nearest = pts[idx[0]]
            accum[i] += nearest

    # 5) 최종 평균 계산
    avg_pts = accum / len(pcd_paths)

    # 6) 평균 점으로 새 PCD 생성
    avg_pcd = o3d.geometry.PointCloud()
    avg_pcd.points = o3d.utility.Vector3dVector(avg_pts)

    # 7) (선택) 색상 정보도 동일한 방식으로 평균 가능
    # 모두 컬러가 있다면:
    # if all(o3d.io.read_point_cloud(p).has_colors() for p in pcd_paths):
    #     accum_color = np.zeros((n, 3), dtype=np.float64)
    #     base_color = np.asarray(base_pcd.colors)
    #     accum_color += base_color
    #     for path in pcd_paths[1:]:
    #         c = np.asarray(o3d.io.read_point_cloud(path).colors)
    #         tree_c = o3d.geometry.KDTreeFlann(
    #             o3d.io.read_point_cloud(path))
    #         for i, p in enumerate(base_pts):
    #             _, idx, _ = tree_c.search_knn_vector_3d(p, 1)
    #             accum_color[i] += c[idx[0]]
    #     avg_color = accum_color / len(pcd_paths)
    #     avg_pcd.colors = o3d.utility.Vector3dVector(avg_color)

    # 8) 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, avg_pcd)
    print(f"✅ 평균 화된 PCD 저장 완료: {output_path}")

if __name__ == "__main__":
    # 처리할 PCD들이 들어있는 폴더와 출력 경로 설정
    INPUT_DIR = "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points"
    OUTPUT_FILE = "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/average.pcd"

    # 폴더 내 모든 PCD 파일 수집
    pcd_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pcd")))

    average_pcds(pcd_files, OUTPUT_FILE)

