import open3d as o3d
from pathlib import Path
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

def colorize_by_z_fixed(
    pcd,
    z_min=-3.0,
    z_max=3.0,
    cmap_name="turbo",
):
    points = np.asarray(pcd.points)

    if len(points) == 0:
        return pcd

    z = points[:, 2]

    norm = Normalize(vmin=z_min, vmax=z_max, clip=True)
    cmap = cm.get_cmap(cmap_name)

    colors = cmap(norm(z))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def apply_camera(vis, camera_param):
    ctr = vis.get_view_control()

    # Open3D 버전별 호환
    try:
        ctr.convert_from_pinhole_camera_parameters(
            camera_param,
            allow_arbitrary=True,
        )
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(camera_param)


def batch_capture_pcds(
    input_dir,
    output_dir,
    camera_json="camera.json",
    render_json="render_option.json",
    ext=".pcd",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_param = o3d.io.read_pinhole_camera_parameters(camera_json)

    width = camera_param.intrinsic.width
    height = camera_param.intrinsic.height

    pcd_paths = sorted(input_dir.glob(f"*{ext}"))

    if len(pcd_paths) == 0:
        raise RuntimeError(f"No {ext} files found in {input_dir}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Batch Render",
        width=width,
        height=height,
        visible=False,
    )

    # 렌더 옵션 불러오기
    vis.get_render_option().load_from_json(render_json)

    for idx, pcd_path in enumerate(pcd_paths):
        print(f"[{idx+1}/{len(pcd_paths)}] Rendering {pcd_path.name}")

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if pcd.is_empty():
            print(f"Skip empty point cloud: {pcd_path}")
            continue
        
        pcd = colorize_by_z_fixed(pcd, z_min=-3.0, z_max=3.0, cmap_name="turbo")

        vis.clear_geometries()
        vis.add_geometry(pcd)

        # 중요: geometry 추가 후 카메라 다시 적용해야 함
        apply_camera(vis, camera_param)

        vis.poll_events()
        vis.update_renderer()

        out_path = output_dir / f"{pcd_path.stem}.png"
        vis.capture_screen_image(str(out_path), do_render=True)

    vis.destroy_window()
    print(f"Saved images to {output_dir}")


if __name__ == "__main__":
    batch_capture_pcds(
        input_dir="./sample",
        output_dir="./renders",
        camera_json="camera.json",
        render_json="render_option.json",
        ext=".pcd",
    )