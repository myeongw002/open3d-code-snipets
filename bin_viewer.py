# visualize_bin_open3d.py

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_bin(bin_path: str) -> np.ndarray:
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"File not found: {bin_path}")

    data = np.fromfile(str(bin_path), dtype=np.float32)

    if data.size % 4 == 0:
        points = data.reshape(-1, 4)
    elif data.size % 3 == 0:
        xyz = data.reshape(-1, 3)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        points = np.hstack([xyz, intensity])
    else:
        raise ValueError(
            f"Invalid .bin size: {data.size}. "
            "Expected Nx4 or Nx3 float32 format."
        )

    return points


def make_colors(points: np.ndarray, mode: str = "intensity") -> np.ndarray:
    xyz = points[:, :3]
    intensity = points[:, 3]

    if mode == "intensity":
        v = intensity.copy()
        if np.max(v) > np.min(v):
            v = (v - np.min(v)) / (np.max(v) - np.min(v))
        else:
            v = np.zeros_like(v)

        colors = np.stack([v, v, v], axis=1)

    elif mode == "z":
        z = xyz[:, 2]
        v = np.clip((z - np.percentile(z, 1)) / (np.percentile(z, 99) - np.percentile(z, 1) + 1e-6), 0, 1)
        colors = np.stack([v, 1.0 - v, 0.5 * np.ones_like(v)], axis=1)

    elif mode == "distance":
        d = np.linalg.norm(xyz, axis=1)
        v = np.clip((d - np.percentile(d, 1)) / (np.percentile(d, 99) - np.percentile(d, 1) + 1e-6), 0, 1)
        colors = np.stack([v, 0.5 * np.ones_like(v), 1.0 - v], axis=1)

    elif mode == "white":
        colors = np.ones((points.shape[0], 3), dtype=np.float32)

    else:
        raise ValueError(f"Unknown color mode: {mode}")

    return colors.astype(np.float64)


def visualize(
    bin_path: str,
    color_mode: str = "intensity",
    point_size: float = 1.5,
    max_range: float | None = None,
    min_range: float | None = None,
    voxel_size: float | None = None,
    show_axis: bool = True,
):
    points = load_bin(bin_path)

    xyz = points[:, :3]
    finite_mask = np.isfinite(xyz).all(axis=1)

    dist = np.linalg.norm(xyz, axis=1)
    range_mask = np.ones(points.shape[0], dtype=bool)

    if min_range is not None:
        range_mask &= dist >= min_range
    if max_range is not None:
        range_mask &= dist <= max_range

    mask = finite_mask & range_mask
    points = points[mask]

    print(f"Loaded: {bin_path}")
    print(f"Points: {points.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(make_colors(points, color_mode))

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsample: {len(pcd.points)}")

    geometries = [pcd]

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
        geometries.append(axis)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(Path(bin_path).name), width=1280, height=720)

    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0.02, 0.02, 0.02])

    view = vis.get_view_control()
    view.set_front([0.0, -1.0, 0.3])
    view.set_lookat([15.0, 0.0, 0.0])
    view.set_up([0.0, 0.0, 1.0])
    view.set_zoom(0.35)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_path", required=True, help="Path to KITTI-style .bin point cloud")
    parser.add_argument(
        "--color",
        default="intensity",
        choices=["intensity", "z", "distance", "white"],
        help="Point color mode",
    )
    parser.add_argument("--point_size", type=float, default=1.5)
    parser.add_argument("--min_range", type=float, default=None)
    parser.add_argument("--max_range", type=float, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--no_axis", action="store_true")
    args = parser.parse_args()

    visualize(
        bin_path=args.bin_path,
        color_mode=args.color,
        point_size=args.point_size,
        min_range=args.min_range,
        max_range=args.max_range,
        voxel_size=args.voxel_size,
        show_axis=not args.no_axis,
    )


if __name__ == "__main__":
    main()
