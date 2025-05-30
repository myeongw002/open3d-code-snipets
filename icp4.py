import open3d as o3d
import numpy as np

def draw_registration_result(source, target, transformation):
    source_temp = source.transform(transformation.copy())
    source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    source_frame.paint_uniform_color([1, 0, 0])  # Red
    source_frame.transform(transformation)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    target_frame.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([source_temp.paint_uniform_color([1, 0, 0]),  # Red
                                       target.paint_uniform_color([0, 1, 0]),
                                       source_frame, target_frame])     # Green

# 1. Load Point Clouds
source = o3d.io.read_point_cloud("/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/ouster1_points/2025-05-16_103546536693.pcd")
target = o3d.io.read_point_cloud("0424-0515/2025-05-17/os1/2025-05-17_13-03-50.pcd")

# 2. Initial Transformation (Identity)
trans_init = np.eye(4)

# 3. Point-to-Point ICP
print("Point-to-Point ICP")
icp_p2p = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=0.1,
    init=trans_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("Transformation:\n", icp_p2p.transformation)
print(f"Fitness: {icp_p2p.fitness:.4f}, RMSE: {icp_p2p.inlier_rmse:.4f}")
draw_registration_result(source, target, icp_p2p.transformation)

# 4. Point-to-Plane ICP
#   Note: Normals are required for point-to-plane ICP
print("Estimating normals for source/target...")
source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

print("Point-to-Plane ICP")
icp_p2l = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=0.1,
    init=trans_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print("Transformation:\n", icp_p2l.transformation)
print(f"Fitness: {icp_p2l.fitness:.4f}, RMSE: {icp_p2l.inlier_rmse:.4f}")
draw_registration_result(source, target, icp_p2l.transformation)
