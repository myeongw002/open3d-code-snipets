import open3d as o3d
import numpy as np
import time, os, datetime
from natsort import natsorted
import matplotlib.pyplot as plt

pcd_dir = "/home/antlab/ROS/scripts/sequences/sequence028/pcd"
pcd_files = natsorted([f for f in os.listdir(pcd_dir) if f.endswith(".pcd")])
camera_path = "/home/antlab/ROS/scripts/ScreenCamera_2025-09-09-16-39-19.json"
idx = 0

import matplotlib.colors as mcolors

def norm_percentile(x, lo_p=1, hi_p=99):
    lo, hi = np.percentile(x, lo_p), np.percentile(x, hi_p)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)

def norm_log(x, eps=1e-6):
    x = np.log1p(np.maximum(x, 0))  # 음수 방지
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def norm_gamma(x, gamma=0.5):  # gamma<1 → 어두운 쪽 확장
    x0 = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return np.power(x0, gamma)


def load_and_colorize_tensor_pcd(pcd_path, cmap_name='jet'):
    pcd_t = o3d.t.io.read_point_cloud(pcd_path)
    legacy = pcd_t.to_legacy()
    if "intensity" in pcd_t.point:
        inten = pcd_t.point["intensity"].numpy().flatten()
        inten = (inten - inten.min()) / (inten.max() - inten.min() + 1e-8)
        colors = plt.get_cmap(cmap_name)(inten)[:, :3].astype(np.float64)
        legacy.colors = o3d.utility.Vector3dVector(colors)
    return legacy

def load_and_colorize_by_distance(pcd_path, cmap_name='viridis',
                                  mode='percentile',  # 'percentile' | 'log' | 'gamma'
                                  lo_p=1, hi_p=99, gamma=0.5,
                                  min_dist=0.5):      # 근접 노이즈 컷(예: 30cm)
    pcd_t = o3d.t.io.read_point_cloud(pcd_path)
    legacy = pcd_t.to_legacy()
    if "positions" not in pcd_t.point: return legacy

    xyz  = pcd_t.point["positions"].numpy()
    mask = np.isfinite(xyz).all(axis=1)

    xyz = xyz[mask]
    if xyz.size == 0:
        print("No finite points"); return legacy
        
    dist = np.linalg.norm(xyz, axis=1)

    # 근거리 노이즈 제거(선택)
    keep = dist > min_dist
    dist = dist[keep]; xyz = xyz[keep]
    legacy.points = o3d.utility.Vector3dVector(xyz)
    dist = dist / 100
    # 정규화 선택
    if mode == 'percentile':
        val = norm_percentile(dist, lo_p, hi_p)
    elif mode == 'log':
        val = norm_log(dist)
    elif mode == 'gamma':
        val = norm_gamma(dist, gamma)
    else:
        raise ValueError

    colors = plt.get_cmap(cmap_name)(val)[:, :3].astype(np.float64)
    legacy.colors = o3d.utility.Vector3dVector(colors)
    return legacy

# ---------------------- 분포 저장 유틸 ----------------------
def save_distributions_from_pcd(pcd_path, out_prefix="pcd_dist"):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pcd_t = o3d.t.io.read_point_cloud(pcd_path)

    # 거리 분포
    if "positions" in pcd_t.point:
        xyz = pcd_t.point["positions"].numpy()
        dist = np.linalg.norm(xyz, axis=1)
        plt.figure(figsize=(8,4))
        plt.hist(dist, bins=100, edgecolor='k')
        plt.title("Distance Distribution from Origin")
        plt.xlabel("Distance"); plt.ylabel("Count"); plt.grid(True)
        dist_png = f"{out_prefix}_dist_{ts}.png"
        plt.savefig(dist_png, dpi=150, bbox_inches="tight"); plt.close()
        print("📌 거리 분포:",
              f"min={dist.min():.3f}, max={dist.max():.3f}, mean={dist.mean():.3f}, std={dist.std():.3f},",
              f"p1~p99={np.percentile(dist,1):.3f}~{np.percentile(dist,99):.3f}")
        print(f"🖼 저장: {dist_png}")

    # 인텐시티 분포
    if "intensity" in pcd_t.point:
        inten = pcd_t.point["intensity"].numpy().flatten()
        plt.figure(figsize=(8,4))
        plt.hist(inten, bins=100, edgecolor='k')
        plt.title("Intensity Distribution")
        plt.xlabel("Intensity"); plt.ylabel("Count"); plt.grid(True)
        inten_png = f"{out_prefix}_inten_{ts}.png"
        plt.savefig(inten_png, dpi=150, bbox_inches="tight"); plt.close()
        print("📌 인텐시티 분포:",
              f"min={inten.min():.3f}, max={inten.max():.3f}, mean={inten.mean():.3f}, std={inten.std():.3f},",
              f"p1~p99={np.percentile(inten,1):.3f}~{np.percentile(inten,99):.3f}")
        print(f"🖼 저장: {inten_png}")

# ---------------------- 재생 제어 ----------------------
is_paused = True
def toggle_play_pause(vis):
    global is_paused
    is_paused = not is_paused
    print("▶️ 재생" if not is_paused else "⏸️ 일시정지")
    return False

# === Visualizer ===
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='PCD Player', width=960, height=720)
vis.register_key_callback(ord(" "), toggle_play_pause)

# 첫 프레임 로딩 및 등록
first_pcd_path = os.path.join(pcd_dir, pcd_files[0])
pcd = load_and_colorize_by_distance(first_pcd_path)
vis.add_geometry(pcd)

# ✅ 카메라 로딩
vis.poll_events(); vis.update_renderer()
if os.path.exists(camera_path):
    try:
        cam_param = o3d.io.read_pinhole_camera_parameters(camera_path)
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
        print(f"[INFO] 카메라 파라미터 로딩 완료: {camera_path}")
    except Exception as e:
        print(f"[WARN] 카메라 파라미터 로딩 실패: {e}")

# ✅ 첫 프레임 분포 PNG 저장 + 통계 출력
#save_distributions_from_pcd(first_pcd_path, out_prefix="pcd_frame0")

# 재생 루프
while vis.poll_events():
    if not is_paused:
        idx += 1
        if idx >= len(pcd_files):
            break
        pcd_new = load_and_colorize_by_distance(os.path.join(pcd_dir, pcd_files[idx]))
        pcd.points = pcd_new.points
        pcd.colors = pcd_new.colors
        vis.update_geometry(pcd)
    vis.update_renderer()
    time.sleep(0.1)

vis.destroy_window()

