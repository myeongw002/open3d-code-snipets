import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# 64e_s3-xiesc.yaml에서 추출한 non-uniform HDL-64E vertical angles
# row 0 = lowest ring, row 63 = highest ring
HDL64E_S2_VERT_ANGLES_DEG = np.array([
    -24.845081, -24.419271, -23.853561, -23.183969,
    -22.727167, -22.357510, -21.885101, -21.307959,
    -20.859316, -20.119377, -19.570175, -19.184532,
    -18.723572, -18.217821, -17.762243, -17.111965,
    -16.554010, -16.111792, -15.689306, -15.177824,
    -14.598064, -14.081364, -13.407262, -12.974276,
    -12.416959, -12.070212, -11.520303, -10.945655,
    -10.860783, -10.538668, -10.362929,  -9.994348,
     -9.818008,  -9.619062,  -9.339722,  -9.071708,
     -8.768623,  -8.356335,  -7.782272,  -7.264264,
     -6.860538,  -6.259461,  -5.668596,  -5.192702,
     -4.704456,  -4.168925,  -3.714313,  -3.189181,
     -2.593387,  -2.055519,  -1.587509,  -1.154410,
     -0.568941,  -0.217594,   0.508544,   0.976948,
      1.445222,   1.971796,   2.486347,   2.977136,
      3.502495,   4.003956,   4.493163,   4.970090,
], dtype=np.float64)


HDL64E_S2_LASER_IDS_BY_RING = np.array([
    38, 39, 42, 43, 32, 33, 36, 37,
    40, 41, 46, 47, 50, 51, 54, 55,
    44, 45, 48, 49, 52, 53, 58, 59,
    62, 63, 34, 35, 56, 57, 60, 61,
     6,  7, 10, 11,  0,  1,  4,  5,
     8,  9, 14, 15, 18, 19, 22, 23,
    12, 13, 16, 17, 20, 21, 26, 27,
    30, 31,  2,  3, 24, 25, 28, 29,
], dtype=np.int32)


def load_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    if pcd.is_empty():
        raise ValueError(f"Empty or invalid PCD file: {pcd_path}")

    xyz = np.asarray(pcd.points)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid point shape: {xyz.shape}")

    return xyz


def compute_elevation_deg(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    horizontal_dist = np.sqrt(x * x + y * y)
    elevation_rad = np.arctan2(z, horizontal_dist)
    elevation_deg = np.rad2deg(elevation_rad)

    return elevation_deg


def assign_ring_by_hardcoded_angles(
    xyz,
    ring_angles_deg=HDL64E_S2_VERT_ANGLES_DEG,
    vmin_deg=None,
    vmax_deg=None,
    max_angle_diff_deg=0.4,
):
    """
    각 point의 elevation angle을 계산한 뒤,
    hard-coded HDL-64E non-uniform vertical angle 중 가장 가까운 ring에 할당.

    반환:
        ring_ids:
            0 = 가장 아래 ring
            63 = 가장 위 ring
            -1 = vertical FOV 밖 또는 angle 차이가 너무 큰 point

        elevation_deg:
            각 point의 elevation angle

        nearest_diff:
            가장 가까운 ring angle과의 차이
    """
    elevation_deg = compute_elevation_deg(xyz)

    valid = np.isfinite(elevation_deg)

    if vmin_deg is not None:
        valid &= elevation_deg >= vmin_deg

    if vmax_deg is not None:
        valid &= elevation_deg <= vmax_deg

    ring_ids = np.full(len(xyz), -1, dtype=np.int32)
    nearest_diff = np.full(len(xyz), np.nan, dtype=np.float64)

    valid_indices = np.where(valid)[0]
    elev_valid = elevation_deg[valid_indices]

    if len(elev_valid) == 0:
        return ring_ids, elevation_deg, nearest_diff

    diff = np.abs(elev_valid[:, None] - ring_angles_deg[None, :])
    nearest_ring = np.argmin(diff, axis=1)
    nearest_ring_diff = diff[np.arange(len(elev_valid)), nearest_ring]

    assigned = np.ones(len(elev_valid), dtype=bool)

    if max_angle_diff_deg is not None:
        assigned &= nearest_ring_diff <= max_angle_diff_deg

    assigned_indices = valid_indices[assigned]

    ring_ids[assigned_indices] = nearest_ring[assigned]
    nearest_diff[valid_indices] = nearest_ring_diff

    return ring_ids, elevation_deg, nearest_diff


def colorize_by_ring(
    ring_ids,
    num_rings=64,
    cmap_name="turbo",
    outside_color=(0.35, 0.35, 0.35),
):
    cmap = plt.get_cmap(cmap_name, num_rings)

    colors = np.zeros((len(ring_ids), 3), dtype=np.float64)

    outside = ring_ids < 0
    colors[outside] = np.array(outside_color, dtype=np.float64)

    valid = ring_ids >= 0
    colors[valid] = cmap(ring_ids[valid])[:, :3]

    return colors


def make_colored_pcd(xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def save_pcd(pcd, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved: {out_path}")


def visualize_pcd(pcd, point_size=2.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="HDL-64E Hardcoded Ring Color")

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def print_ring_stats(ring_ids, elevation_deg, nearest_diff):
    print("=" * 90)
    print("Hard-coded HDL-64E S3 ring assignment")
    print("-" * 90)
    print(f"Raw elevation min: {np.nanmin(elevation_deg):.4f} deg")
    print(f"Raw elevation max: {np.nanmax(elevation_deg):.4f} deg")
    print(f"Assigned points   : {np.sum(ring_ids >= 0)}")
    print(f"Unassigned points : {np.sum(ring_ids < 0)}")
    print("-" * 90)
    print("ring | laser_id | calib_angle_deg | num_points")
    print("-" * 90)

    for r in range(64):
        count = int(np.sum(ring_ids == r))
        laser_id = int(HDL64E_S2_LASER_IDS_BY_RING[r])
        angle = HDL64E_S2_VERT_ANGLES_DEG[r]
        print(f"{r:02d}   | {laser_id:02d}       | {angle: .6f}       | {count}")

    valid_diff = nearest_diff[np.isfinite(nearest_diff)]
    if len(valid_diff) > 0:
        print("-" * 90)
        print(f"nearest angle diff mean: {np.mean(valid_diff):.6f} deg")
        print(f"nearest angle diff p90 : {np.percentile(valid_diff, 90):.6f} deg")
        print(f"nearest angle diff max : {np.max(valid_diff):.6f} deg")

    print("=" * 90)


def process_single_pcd(
    pcd_path,
    out_path=None,
    point_size=2.0,
    no_vis=False,
    cmap_name="turbo",
    max_angle_diff_deg=0.4,
    drop_unassigned=False,
):
    xyz = load_pcd(pcd_path)

    ring_ids, elevation_deg, nearest_diff = assign_ring_by_hardcoded_angles(
        xyz=xyz,
        ring_angles_deg=HDL64E_S2_VERT_ANGLES_DEG,
        vmin_deg=HDL64E_S2_VERT_ANGLES_DEG[0],
        vmax_deg=HDL64E_S2_VERT_ANGLES_DEG[-1],
        max_angle_diff_deg=max_angle_diff_deg,
    )

    print(f"Loaded: {pcd_path}")
    print(f"points: {len(xyz)}")

    print_ring_stats(ring_ids, elevation_deg, nearest_diff)

    colors = colorize_by_ring(
        ring_ids=ring_ids,
        num_rings=64,
        cmap_name=cmap_name,
    )

    if drop_unassigned:
        keep = ring_ids >= 0
        xyz = xyz[keep]
        colors = colors[keep]
        print(f"After dropping unassigned points: {len(xyz)}")

    colored_pcd = make_colored_pcd(xyz, colors)

    if out_path is not None:
        save_pcd(colored_pcd, out_path)

    if not no_vis:
        visualize_pcd(colored_pcd, point_size=point_size)


def process_pcd_folder(
    pcd_dir,
    out_dir,
    cmap_name="turbo",
    max_angle_diff_deg=0.4,
    drop_unassigned=False,
):
    pcd_dir = Path(pcd_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd_files = sorted(pcd_dir.glob("*.pcd"))

    if len(pcd_files) == 0:
        raise FileNotFoundError(f"No .pcd files found in {pcd_dir}")

    for pcd_path in pcd_files:
        print(f"\nProcessing: {pcd_path.name}")

        xyz = load_pcd(pcd_path)

        ring_ids, elevation_deg, nearest_diff = assign_ring_by_hardcoded_angles(
            xyz=xyz,
            ring_angles_deg=HDL64E_S2_VERT_ANGLES_DEG,
            vmin_deg=HDL64E_S2_VERT_ANGLES_DEG[0],
            vmax_deg=HDL64E_S2_VERT_ANGLES_DEG[-1],
            max_angle_diff_deg=max_angle_diff_deg,
        )

        colors = colorize_by_ring(
            ring_ids=ring_ids,
            num_rings=64,
            cmap_name=cmap_name,
        )

        if drop_unassigned:
            keep = ring_ids >= 0
            xyz = xyz[keep]
            colors = colors[keep]

        colored_pcd = make_colored_pcd(xyz, colors)

        out_path = out_dir / f"{pcd_path.stem}_ring_colored.pcd"
        save_pcd(colored_pcd, out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pcd", default=None, help="Single PCD file path")
    parser.add_argument("--pcd-dir", default=None, help="Directory containing PCD files")
    parser.add_argument("--out", default=None, help="Output PCD path for single file")
    parser.add_argument("--out-dir", default=None, help="Output directory for folder mode")

    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--cmap", default="turbo")

    parser.add_argument(
        "--max-angle-diff-deg",
        type=float,
        default=0.4,
        help="Max allowed elevation difference from nearest ring angle. Use negative value to disable.",
    )

    parser.add_argument(
        "--drop-unassigned",
        action="store_true",
        help="Drop points outside angle table or farther than max-angle-diff-deg",
    )

    parser.add_argument("--no-vis", action="store_true")

    args = parser.parse_args()

    max_angle_diff_deg = args.max_angle_diff_deg
    if max_angle_diff_deg < 0:
        max_angle_diff_deg = None

    if args.pcd is None and args.pcd_dir is None:
        raise ValueError("Use either --pcd or --pcd-dir")

    if args.pcd is not None:
        process_single_pcd(
            pcd_path=args.pcd,
            out_path=args.out,
            point_size=args.point_size,
            no_vis=args.no_vis,
            cmap_name=args.cmap,
            max_angle_diff_deg=max_angle_diff_deg,
            drop_unassigned=args.drop_unassigned,
        )

    if args.pcd_dir is not None:
        if args.out_dir is None:
            raise ValueError("Folder mode requires --out-dir")

        process_pcd_folder(
            pcd_dir=args.pcd_dir,
            out_dir=args.out_dir,
            cmap_name=args.cmap,
            max_angle_diff_deg=max_angle_diff_deg,
            drop_unassigned=args.drop_unassigned,
        )


if __name__ == "__main__":
    main()
