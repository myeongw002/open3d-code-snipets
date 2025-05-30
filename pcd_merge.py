#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import open3d as o3d
import os

def merge_pcds(pcd_paths, output_path, voxel_downsample=None):
    """
    여러 PCD 파일을 읽어 하나로 합친 뒤 저장합니다.

    :param pcd_paths: 합칠 PCD 파일 경로 리스트
    :param output_path: 합쳐진 PCD를 저장할 파일 경로
    :param voxel_downsample: (선택) 합친 뒤 voxel downsampling 할 voxel size (float)
    """
    merged = o3d.geometry.PointCloud()
    for path in pcd_paths:
        if not os.path.exists(path):
            print(f"경고: 파일이 없습니다: {path}")
            continue
        pcd = o3d.io.read_point_cloud(path)
        merged += pcd

    # 필요하다면 중복 점 제거/균질화
    if voxel_downsample is not None and voxel_downsample > 0:
        merged = merged.voxel_down_sample(voxel_downsample)

    # Normals가 필요한 경우 재계산
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_downsample*2 if voxel_downsample else 0.1,
                                                          max_nn=30)
    )

    # 저장
    o3d.visualization.draw_geometries([merged])
    o3d.io.write_point_cloud(output_path, merged)
    print(f"✅ 병합된 PCD 저장 완료: {output_path}")


if __name__ == "__main__":
    # 합칠 PCD 파일 목록
    pcd_files = [
        "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/2025-05-15_000000866136.pcd",
        "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/2025-05-15_000002796104.pcd",
        "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/2025-05-15_000004570003.pcd",
        "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/2025-05-15_000006106315.pcd",
        "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/2025-05-15_000007748046.pcd"
        # 추가 경로 …
    ]

    # 출력 파일
    output_pcd = "/home/myungw00/ROS/gm/Code/data/scripts/0510-0515/2025-05-15/ouster1_points/merged.pcd"

    # 예: voxel downsampling 크기를 0.02로 지정하고 싶다면 아래처럼 설정
    voxel_size = None  # 또는 0.02

    merge_pcds(pcd_files, output_pcd, voxel_downsample=voxel_size)

