import open3d as o3d
from matplotlib import cm


def visualize_pcds(pcd_paths):
    """
    주어진 PCD 파일 경로 리스트를 서로 다른 색으로 시각화합니다.
    """
    # 컬러맵에서 파일 개수만큼 색상 추출
    cmap = cm.get_cmap('tab10', len(pcd_paths))
    geometries = []
    for idx, path in enumerate(pcd_paths):
        pcd = o3d.io.read_point_cloud(path)
        # RGB 값으로 변환
        color = cmap(idx)[:3]
        pcd.paint_uniform_color(color)
        geometries.append(pcd)
        print(f"Index : {idx}, Color : {color}")
        print(f"Points: {len(pcd.points)}")
    # 시각화 실행
    o3d.visualization.draw_geometries(
        geometries,
        window_name='Multi-PCD Viewer',
        width=800, height=600,
        left=50, top=50,
        point_show_normal=False
    )


if __name__ == '__main__':
    # 시각화할 PCD 파일 경로를 직접 여기에서 입력하세요
    pcd_paths = [
        '/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/roi/1748617011836370.pcd'
    ]
    visualize_pcds(pcd_paths)
