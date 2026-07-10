import open3d as o3d
from matplotlib import cm


def visualize_pcds(pcd_paths, axis_size=1.0):
    """
    주어진 PCD 파일 경로 리스트를 서로 다른 색으로 시각화합니다.
    좌표축도 함께 표시합니다.
    """
    cmap = cm.get_cmap('tab10', len(pcd_paths))
    geometries = []

    for idx, path in enumerate(pcd_paths):
        pcd = o3d.io.read_point_cloud(path)

        color = cmap(idx)[:3]
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        print(f"Index : {idx}, Color : {color}")
        print(f"Points: {len(pcd.points)}")

    # 좌표축 추가
    # 빨간색: X축
    # 초록색: Y축
    # 파란색: Z축
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name='Multi-PCD Viewer',
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False
    )


if __name__ == '__main__':
    pcd_paths = [
        '/media/myungw00/2TB_SSD/kitti/kitti_object/testing/velodyne_pcd/000000.pcd',
        '/home/myungw00/ROS2/upsample_ws/depth_any_ws/Pseudo_Lidar_V2/results/sdn_kitti_train_set/pseudo_lidar_gdc_64ch_from_4beam_pcd/test/000000.pcd'
    ]

    visualize_pcds(pcd_paths, axis_size=2.0)
