#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import open3d as o3d

def visualize_two_pcds(pcd1_path, pcd2_path):
    # 첫 번째 PCD (red)
    pcd1 = o3d.io.read_point_cloud(pcd1_path)
    pcd1.paint_uniform_color([0.5, 0.5, 0.5])  # 빨간색

    # 두 번째 PCD (green)
    pcd2 = o3d.io.read_point_cloud(pcd2_path)
    pcd2.paint_uniform_color([0, 1, 0])  # 초록색

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
    
    PCD1_PATH = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/roi/ouster1_roi3.pcd"
    PCD2_PATH = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/roi/1748617011836370.pcd"
    visualize_two_pcds(PCD1_PATH, PCD2_PATH)

