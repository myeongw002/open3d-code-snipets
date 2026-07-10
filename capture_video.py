import open3d as o3d
from pathlib import Path
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import cv2
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import math


def natural_key(path):
    path = Path(path)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

def set_color_option(opt):
    """
    pcd.colors에 저장된 RGB 색상을 그대로 사용하도록 설정.
    Open3D 버전별 차이를 최대한 흡수.
    """
    try:
        opt.point_color_option = o3d.visualization.PointColorOption.Color
    except Exception:
        pass

def render_worker(args):
    (
        worker_id,
        pcd_paths,
        output_dir,
        camera_json,
        render_json,
        point_size,
        visible,
        use_fixed_colormap,
        z_min,
        z_max,
        cmap_name,
    ) = args

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_param = o3d.io.read_pinhole_camera_parameters(str(camera_json))
    width = camera_param.intrinsic.width
    height = camera_param.intrinsic.height

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Batch Render Worker {worker_id}",
        width=width,
        height=height,
        visible=visible,
    )

    opt = vis.get_render_option()

    if render_json is not None and Path(render_json).exists():
        opt.load_from_json(str(render_json))

    opt.point_size = float(point_size)
    opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    opt.show_coordinate_frame = False
    set_color_option(opt)

    rendered_count = 0

    for idx, pcd_path in enumerate(pcd_paths):
        pcd_path = Path(pcd_path)

        print(f"[Worker {worker_id}] Rendering {idx + 1}/{len(pcd_paths)}: {pcd_path.name}")

        pcd = o3d.io.read_point_cloud(str(pcd_path))

        if pcd.is_empty():
            print(f"[Worker {worker_id}] Skip empty PCD: {pcd_path}")
            continue

        if use_fixed_colormap:
            pcd = colorize_by_z_fixed(
                pcd,
                z_min=z_min,
                z_max=z_max,
                cmap_name=cmap_name,
            )
        else:
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.0, 0.0, 0.0])

        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=True)

        vis.poll_events()
        vis.update_renderer()

        apply_camera(vis, camera_param)

        # 렌더 옵션 재적용
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        opt.show_coordinate_frame = False
        set_color_option(opt)

        for _ in range(3):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.005)

        out_path = output_dir / f"{pcd_path.stem}.png"
        vis.capture_screen_image(str(out_path), do_render=True)

        rendered_count += 1

    vis.destroy_window()

    return worker_id, rendered_count

def parallel_batch_capture_pcds(
    input_dir,
    output_dir,
    camera_json="camera.json",
    render_json="render_option.json",
    ext=".pcd",
    point_size=2.0,
    visible=True,
    use_fixed_colormap=True,
    z_min=-3.0,
    z_max=3.0,
    cmap_name="turbo",
    num_workers=2,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_json = Path(camera_json)
    render_json = Path(render_json)

    if not camera_json.exists():
        raise FileNotFoundError(f"camera_json not found: {camera_json}")

    pcd_paths = sorted(input_dir.glob(f"*{ext}"), key=natural_key)

    if len(pcd_paths) == 0:
        raise RuntimeError(f"No {ext} files found in {input_dir}")

    num_workers = int(num_workers)
    num_workers = max(1, min(num_workers, len(pcd_paths)))

    print(f"Total PCD count : {len(pcd_paths)}")
    print(f"Worker count    : {num_workers}")

    if num_workers == 1:
        batch_capture_pcds(
            input_dir=input_dir,
            output_dir=output_dir,
            camera_json=camera_json,
            render_json=render_json,
            ext=ext,
        )
        return

    chunk_size = math.ceil(len(pcd_paths) / num_workers)
    chunks = [
        pcd_paths[i:i + chunk_size]
        for i in range(0, len(pcd_paths), chunk_size)
    ]

    worker_args = []

    for worker_id, chunk in enumerate(chunks):
        worker_args.append(
            (
                worker_id,
                [str(p) for p in chunk],
                str(output_dir),
                str(camera_json),
                str(render_json),
                float(point_size),
                bool(visible),
                bool(use_fixed_colormap),
                float(z_min),
                float(z_max),
                str(cmap_name),
            )
        )

    total_rendered = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(render_worker, args)
            for args in worker_args
        ]

        for future in as_completed(futures):
            worker_id, rendered_count = future.result()
            print(f"[Worker {worker_id}] Done. Rendered: {rendered_count}")
            total_rendered += rendered_count

    print(f"Parallel rendering done. Total rendered images: {total_rendered}")
    print(f"Saved images to {output_dir}")

def parallel_capture_and_make_video(
    input_dir,
    image_dir,
    output_video,
    camera_json="camera.json",
    render_json="render_option.json",
    ext=".pcd",
    fps=10,
    video_length_sec=None,
    delete_images_after_video=True,
    num_workers=2,
):
    parallel_batch_capture_pcds(
        input_dir=input_dir,
        output_dir=image_dir,
        camera_json=camera_json,
        render_json=render_json,
        ext=ext,
        point_size=3.0,
        visible=False,
        use_fixed_colormap=True,
        z_min=-3.0,
        z_max=3.0,
        cmap_name="turbo",
        num_workers=num_workers,
    )

    images_to_video(
        image_dir=image_dir,
        output_video=output_video,
        fps=fps,
        video_length_sec=video_length_sec,
        image_ext=".png",
        delete_images_after_video=delete_images_after_video,
    )

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

    try:
        ctr.convert_from_pinhole_camera_parameters(
            camera_param,
            allow_arbitrary=True,
        )
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(camera_param)

    # clipping 때문에 안 보이는 경우 방지
    try:
        ctr.set_constant_z_near(0.01)
        ctr.set_constant_z_far(100000.0)
    except Exception:
        pass


def force_render_options(
    vis,
    render_json=None,
    point_size=2.0,
    background_color=(1.0, 1.0, 1.0),
):
    opt = vis.get_render_option()

    if render_json is not None and Path(render_json).exists():
        opt.load_from_json(str(render_json))

    # 포인트가 너무 작아서 안 보이는 문제 방지
    opt.point_size = float(point_size)

    # 배경색 강제
    opt.background_color = np.asarray(background_color, dtype=np.float64)

    # 좌표축 표시 끄기
    opt.show_coordinate_frame = False

    # 포인트 컬러를 pcd.colors 기준으로 사용
    set_color_option(opt)

    return opt


def batch_capture_pcds(
    input_dir,
    output_dir,
    camera_json="camera.json",
    render_json="render_option.json",
    ext=".pcd",
    point_size=2.0,
    visible=True,
    z_min=-3.0,
    z_max=3.0,
    use_fixed_colormap=True,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_json = Path(camera_json)
    render_json = Path(render_json)

    if not camera_json.exists():
        raise FileNotFoundError(f"camera_json not found: {camera_json}")

    camera_param = o3d.io.read_pinhole_camera_parameters(str(camera_json))

    width = camera_param.intrinsic.width
    height = camera_param.intrinsic.height

    pcd_paths = sorted(input_dir.glob(f"*{ext}"), key=natural_key)

    if len(pcd_paths) == 0:
        raise RuntimeError(f"No {ext} files found in {input_dir}")

    print(f"Input PCD count: {len(pcd_paths)}")
    print(f"Capture size: {width} x {height}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Batch Render",
        width=width,
        height=height,
        visible=visible,
    )

    force_render_options(
        vis,
        render_json=render_json,
        point_size=point_size,
        background_color=(1.0, 1.0, 1.0),
    )

    for idx, pcd_path in enumerate(pcd_paths):
        print(f"[{idx + 1}/{len(pcd_paths)}] Rendering {pcd_path.name}")

        pcd = o3d.io.read_point_cloud(str(pcd_path))

        if pcd.is_empty():
            print(f"Skip empty point cloud: {pcd_path}")
            continue

        points = np.asarray(pcd.points)
        print(
            f"  points: {len(points)}, "
            f"x[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
            f"y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
            f"z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]"
        )

        if use_fixed_colormap:
            pcd = colorize_by_z_fixed(
                pcd,
                z_min=z_min,
                z_max=z_max,
                cmap_name="turbo",
            )
        else:
            # 색이 없는 PCD일 경우 검정색으로 강제
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.0, 0.0, 0.0])

        vis.clear_geometries()

        # 처음 geometry bounding box를 인식시킨 뒤 카메라를 다시 적용
        vis.add_geometry(pcd, reset_bounding_box=True)

        vis.poll_events()
        vis.update_renderer()

        # 저장된 카메라 적용
        apply_camera(vis, camera_param)

        # 렌더 옵션 다시 강제
        force_render_options(
            vis,
            render_json=render_json,
            point_size=point_size,
            background_color=(1.0, 1.0, 1.0),
        )

        # 렌더링 안정화용
        for _ in range(5):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)

        out_path = output_dir / f"{pcd_path.stem}.png"
        vis.capture_screen_image(str(out_path), do_render=True)

    vis.destroy_window()
    print(f"Saved images to {output_dir}")


def images_to_video(
    image_dir,
    output_video,
    fps=10,
    video_length_sec=None,
    image_ext=".png",
    delete_images_after_video=False,
):
    image_dir = Path(image_dir)
    output_video = Path(output_video)

    image_paths = sorted(image_dir.glob(f"*{image_ext}"), key=natural_key)

    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in {image_dir}")

    # 영상 길이를 직접 지정하면 fps를 자동 계산
    if video_length_sec is not None:
        if video_length_sec <= 0:
            raise ValueError("video_length_sec must be positive.")

        fps = len(image_paths) / float(video_length_sec)

        print(f"Video length option enabled:")
        print(f"  image count      : {len(image_paths)}")
        print(f"  target length    : {video_length_sec:.2f} sec")
        print(f"  calculated fps   : {fps:.4f}")
    else:
        print(f"Video fps option enabled:")
        print(f"  image count      : {len(image_paths)}")
        print(f"  fps              : {fps}")

    first_img = cv2.imread(str(image_paths[0]))

    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {image_paths[0]}")

    height, width = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    for idx, image_path in enumerate(image_paths):
        print(f"[Video {idx + 1}/{len(image_paths)}] {image_path.name}")

        img = cv2.imread(str(image_path))

        if img is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        writer.write(img)

    writer.release()

    print(f"Video saved to {output_video}")

    # 영상 생성 후 렌더 이미지 삭제
    if delete_images_after_video:
        deleted_count = 0

        for image_path in image_paths:
            try:
                image_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {image_path}: {e}")

        print(f"Deleted rendered images: {deleted_count}")


if __name__ == "__main__":
    parallel_capture_and_make_video(
        input_dir="./tulip_pcd",
        image_dir="./renders",
        output_video="./videos/tulip_video.mp4",
        camera_json="camera.json",
        render_json="render_option.json",
        ext=".pcd",
        video_length_sec=120.0,
        delete_images_after_video=True,
        num_workers=16,
    )