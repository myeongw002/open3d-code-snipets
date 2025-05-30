#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np

def visualize_two_pcds(pcd1_path, pcd2_path, z_offset=0.0):
    # 첫 번째 PCD (gray)
    pcd1 = o3d.io.read_point_cloud(pcd1_path)
    pcd1.paint_uniform_color([0.5, 0.5, 0.5])

    # 두 번째 PCD (green)
    pcd2 = o3d.io.read_point_cloud(pcd2_path)
    pcd2.paint_uniform_color([0, 1, 0])
    
    # z축 오프셋 적용
    pcd2_points = np.asarray(pcd2.points)
    pcd2_points[:, 2] += z_offset
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)

    # 시각화
    o3d.visualization.draw_geometries(
        [pcd1, pcd2],
        window_name="Red vs Green PCDs",
        width=800, height=600,
        left=50, top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    # 실제 파일 경로로 수정하세요
    PCD1_PATH = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-28/os1/2025-05-28_07-00-02.pcd"
    PCD2_PATH = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-28/os1/2025-05-28_08-00-02.pcd"
    Z_OFFSET = 0  # 원하는 만큼 z축 오프셋(m 단위 등)을 입력

    visualize_two_pcds(PCD1_PATH, PCD2_PATH, Z_OFFSET)

