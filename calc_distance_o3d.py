import open3d as o3d
import numpy as np

def load_and_color_pcds(pcd_paths, colors):
    """주어진 경로 리스트의 PCD를 불러와 각 색으로 페인트"""
    pcds = []
    for path, color in zip(pcd_paths, colors):
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise FileNotFoundError(f"PCD 파일을 불러올 수 없습니다: {path}")
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
    return pcds

def measure_distance(merged_pcd):
    """
    VisualizerWithEditing을 띄워 두 점을 선택하게 한 뒤,
    선택 점 인덱스 리스트와 계산된 거리를 반환.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Select 2 Points")
    vis.add_geometry(merged_pcd)
    print("Shift + 좌클릭으로 두 점을 선택한 뒤 창을 닫으세요.")
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    if len(picked) != 2:
        print(f"선택된 점 개수: {len(picked)} (2점을 선택해야 합니다.)")
        return None, None, None

    pts = np.asarray(merged_pcd.points)
    p0 = pts[picked[0]]
    p1 = pts[picked[1]]
    dist = np.linalg.norm(p1 - p0)
    return picked, (p0, p1), dist

def main():
    # 1) PCD 파일 경로 및 색 지정
    PCD_PATHS = [
        "/home/myungw00/ROS/gm/Code/data/sample/sample1.pcd",
        "/home/myungw00/ROS/gm/Code/data/sample/sample2.pcd",
        "/home/myungw00/ROS/gm/Code/data/sample/sample3.pcd",
        "/home/myungw00/ROS/gm/Code/data/2025-05-07_20-59-35_1.pcd"
    ]
    COLORS = [
        [1.0, 0.0, 0.0],   # 빨강
        [0.0, 1.0, 0.0],   # 초록
        [0.0, 0.0, 1.0],   # 파랑
        [0.5, 0.5, 0.5]    # 회색
    ]

    # 2) 로드 및 병합
    pcds = load_and_color_pcds(PCD_PATHS, COLORS)
    merged_pcd = pcds[0]
    for pcd in pcds[1:]:
        merged_pcd += pcd

    # 3) 반복 측정 루프
    while True:
        picked, (p0, p1), dist = measure_distance(merged_pcd)
        if dist is None:
            break  # 2점을 선택하지 않았으면 종료
        print("=== 거리 계산 결과 ===")
        print(f"Point A (index={picked[0]}): x={p0[0]:.3f}, y={p0[1]:.3f}, z={p0[2]:.3f}")
        print(f"Point B (index={picked[1]}): x={p1[0]:.3f}, y={p1[1]:.3f}, z={p1[2]:.3f}")
        print(f"Distance = {dist:.5f} m")
        inp = input("다시 측정하려면 Enter, 종료하려면 'q' 입력: ")
        if inp.lower() == 'q':
            break

if __name__ == "__main__":
    main()

