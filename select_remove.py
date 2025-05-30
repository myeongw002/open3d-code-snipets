#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d

# ───────────────── 사용자 설정 ──────────────────
INPUT_PCD  = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-30/sample/1748587859358440.pcd"
OUTPUT_PCD = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-30/sample/1748587859358440.pcd"
# ─────────────────────────────────────────────

def main():
    # 1) PCD 로드
    pcd = o3d.io.read_point_cloud(INPUT_PCD)
    print(f"Loaded '{INPUT_PCD}' with {len(pcd.points)} points")

    # 2) VisualizerWithEditing 실행
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick points to remove", width=800, height=600)
    vis.add_geometry(pcd)
    print(">> Shift+클릭으로 제거할 점을 선택하세요. 완료되면 창을 닫으세요.")
    vis.run()           # 여기서 사용자 입력 대기
    vis.destroy_window()

    # 선택된 점 인덱스 가져오기
    picked_indices = vis.get_picked_points()
    print("Picked indices:", picked_indices)
    if not picked_indices:
        print("No points were picked. Exiting without changes.")
        return

    # 3) 선택된 점 제거
    cleaned = pcd.select_by_index(picked_indices, invert=True)
    print(f"After removal: {len(cleaned.points)} points remain")

    # 4) 결과 저장
    o3d.io.write_point_cloud(OUTPUT_PCD, cleaned)
    print(f"Saved cleaned point cloud to '{OUTPUT_PCD}'")

    # (선택사항) 결과 시각화
    # o3d.visualization.draw_geometries([cleaned], window_name="Cleaned PCD")

if __name__ == "__main__":
    main()
