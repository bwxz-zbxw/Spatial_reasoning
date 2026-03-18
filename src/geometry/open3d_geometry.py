from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class Open3DObstacleCluster:
    cluster_id: int
    point_count: int
    center_robot_frame: tuple[float, float, float]
    extent_robot_frame: tuple[float, float, float]


@dataclass
class PlaneFitResult:
    normal_robot_frame: tuple[float, float, float]
    offset_d: float
    inlier_count: int
    mean_error_m: float


def depth_to_robot_point_cloud(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    color_rgb: np.ndarray | None = None,
    stride: int = 2,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    sampled_depth = depth_m[::stride, ::stride]
    valid = np.isfinite(sampled_depth)
    if int(valid.sum()) == 0:
        return o3d.geometry.PointCloud(), np.zeros((0, 3), dtype=np.float32)

    rows, cols = np.where(valid)
    z = sampled_depth[valid].astype(np.float32)
    u = cols.astype(np.float32) * stride
    v = rows.astype(np.float32) * stride

    x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
    y_camera = ((v - intrinsics[1, 2]) / intrinsics[1, 1]) * z

    forward = z
    lateral = -x_camera
    vertical = -y_camera

    points_robot = np.stack([forward, lateral, vertical], axis=1)
    mask = (
        (points_robot[:, 0] > 0.15)
        & (points_robot[:, 0] < 4.5)
        & (np.abs(points_robot[:, 1]) < 3.0)
        & (points_robot[:, 2] > -0.4)
        & (points_robot[:, 2] < 2.2)
    )
    points_robot = points_robot[mask]

    pcd = o3d.geometry.PointCloud()
    if len(points_robot) == 0:
        return pcd, points_robot

    # Convert robot frame (forward, lateral, vertical) to Open3D xyz.
    pcd.points = o3d.utility.Vector3dVector(points_robot)

    if color_rgb is not None:
        sampled_color = color_rgb[::stride, ::stride][valid][mask].astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(sampled_color)

    return pcd, points_robot


def remove_floor_and_far_background(
    pcd: o3d.geometry.PointCloud,
    min_height_m: float = 0.08,
    max_forward_m: float = 4.0,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd

    points = np.asarray(pcd.points)
    mask = (points[:, 2] > min_height_m) & (points[:, 0] < max_forward_m)
    return pcd.select_by_index(np.where(mask)[0].tolist())


def remove_large_planar_surfaces(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 300,
    min_plane_points: int = 1500,
    max_planes: int = 3,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd

    remaining = pcd
    for _ in range(max_planes):
        if len(remaining.points) < min_plane_points:
            break

        _, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_plane_points:
            break
        remaining = remaining.select_by_index(inliers, invert=True)
    return remaining


def cluster_obstacles(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.12,
    min_points: int = 60,
) -> list[Open3DObstacleCluster]:
    if len(pcd.points) == 0:
        return []

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return []

    points = np.asarray(pcd.points)
    clusters: list[Open3DObstacleCluster] = []
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id < 0:
            continue
        mask = labels == cluster_id
        cluster_points = points[mask]
        if len(cluster_points) < min_points:
            continue

        min_corner = cluster_points.min(axis=0)
        max_corner = cluster_points.max(axis=0)
        center = cluster_points.mean(axis=0)
        extent = max_corner - min_corner

        clusters.append(
            Open3DObstacleCluster(
                cluster_id=int(cluster_id),
                point_count=int(len(cluster_points)),
                center_robot_frame=(float(center[0]), float(center[1]), float(center[2])),
                extent_robot_frame=(float(extent[0]), float(extent[1]), float(extent[2])),
            )
        )
    return clusters


def fit_plane_ransac(
    points_robot: np.ndarray,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 250,
    min_points: int = 80,
) -> PlaneFitResult | None:
    if points_robot is None or len(points_robot) < min_points:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_robot, dtype=np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    if len(inliers) < min_points:
        return None

    a, b, c, d = [float(value) for value in plane_model]
    normal = np.asarray([a, b, c], dtype=np.float32)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-6:
        return None
    normal = normal / normal_norm

    inlier_points = np.asarray(pcd.points)[inliers]
    errors = np.abs((inlier_points @ normal.astype(np.float64)) + d)
    return PlaneFitResult(
        normal_robot_frame=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset_d=d,
        inlier_count=int(len(inliers)),
        mean_error_m=float(errors.mean()) if errors.size else 0.0,
    )
