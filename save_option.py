import open3d as o3d
from pathlib import Path


def save_view_options(
    pcd_path,
    camera_json="camera.json",
    render_json="render_option.json",
    width=1600,
    height=900,
):
    pcd_path = Path(pcd_path)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Adjust view, then press Q",
        width=width,
        height=height,
        visible=True,
    )

    vis.add_geometry(pcd)

    # 기본 렌더 옵션
    opt = vis.get_render_option()
    opt.background_color = [1.0, 1.0, 1.0]  # 흰 배경
    opt.point_size = 1.0                    # 점 크기
    opt.show_coordinate_frame = False

    print("뷰를 원하는 각도로 맞춘 뒤 Q를 누르세요.")
    vis.run()

    # 카메라 저장
    cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(camera_json, cam_param)

    # 렌더 옵션 저장
    vis.get_render_option().save_to_json(render_json)

    vis.destroy_window()

    print(f"Saved camera: {camera_json}")
    print(f"Saved render option: {render_json}")


if __name__ == "__main__":
    save_view_options(
        pcd_path="original_pcd\\000000.pcd",
        camera_json="camera.json",
        render_json="render_option.json",
        width=1600,
        height=900,
    )