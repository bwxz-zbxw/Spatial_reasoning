from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

from src.geometry.open3d_geometry import (
    cluster_obstacles,
    depth_to_robot_point_cloud,
    fit_plane_ransac,
    remove_floor_and_far_background,
    remove_large_planar_surfaces,
)
from src.geometry.spatial_language import euclidean_distance_3d
from src.perception.observation_protocol import ObservedObject


@dataclass
class WallEstimate:
    side: str
    lateral_distance_m: float
    forward_distance_m: float
    distance_m: float
    valid_point_count: int
    normal_robot_frame: tuple[float, float, float] | None = None
    plane_inlier_count: int | None = None
    plane_fit_error_m: float | None = None


@dataclass
class ObstacleEstimate:
    object_id: str
    category: str
    forward_distance_m: float
    lateral_offset_m: float
    distance_m: float
    width_m: float
    source: str


@dataclass
class CorridorGap:
    start_m: float
    end_m: float
    width_m: float


@dataclass
class SceneGeometryState:
    reference_frame: str
    robot_width_m: float
    safety_margin_m: float
    left_wall: WallEstimate | None
    right_wall: WallEstimate | None
    left_wall_visible: bool | None
    right_wall_visible: bool | None
    left_wall_confidence: float | None
    right_wall_confidence: float | None
    corridor_width_m: float | None
    traversable_width_m: float | None
    passable: bool | None
    nearest_obstacle_distance_m: float | None
    nearest_obstacle: ObstacleEstimate | None
    obstacles: List[ObstacleEstimate] = field(default_factory=list)
    gaps: List[CorridorGap] = field(default_factory=list)
    blocking_obstacle_ids: List[str] = field(default_factory=list)


@dataclass
class WallTrack:
    track_id: str
    side: str
    estimate: WallEstimate
    confidence: float
    last_seen_frame: int
    visible: bool
    miss_count: int = 0


class RGBDGeometryBranch:
    """Compute robot-centric corridor geometry from depth and detected objects."""

    def __init__(
        self,
        robot_width_m: float = 0.55,
        safety_margin_m: float = 0.15,
        max_obstacle_forward_m: float = 3.0,
    ) -> None:
        self.robot_width_m = robot_width_m
        self.safety_margin_m = safety_margin_m
        self.max_obstacle_forward_m = max_obstacle_forward_m
        self.depth_obstacle_forward_m = min(2.5, max_obstacle_forward_m)
        self.obstacle_bin_width_m = 0.05
        self.wall_edge_margin_m = 0.12
        self.wall_alpha = 0.35
        self.max_wall_lateral_jump_m = 0.45
        self.max_wall_forward_jump_m = 1.0
        self.max_wall_misses = 8
        self.open3d_stride = 2
        self._wall_tracks: dict[str, WallTrack] = {}
        self._frame_counter = -1

    def estimate(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        detected_objects: Iterable[ObservedObject] | None = None,
        frame_index: int | None = None,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> SceneGeometryState:
        if frame_index is None:
            self._frame_counter += 1
            frame_index = self._frame_counter
        else:
            self._frame_counter = frame_index

        depth_clean = self._sanitize_depth(depth_m)
        walls = self._extract_corridor_walls(depth_clean, intrinsics, guidance_maps)
        left_wall_raw = next((wall for wall in walls if wall.side == "left"), None)
        right_wall_raw = next((wall for wall in walls if wall.side == "right"), None)
        left_wall = self._update_wall_track("left", left_wall_raw, frame_index)
        right_wall = self._update_wall_track("right", right_wall_raw, frame_index)

        corridor_width = None
        if left_wall is not None and right_wall is not None:
            corridor_width = round(left_wall.lateral_distance_m + right_wall.lateral_distance_m, 3)

        obstacles = list(detected_objects or [])
        open3d_obstacles = self._extract_open3d_obstacles(
            depth_clean,
            intrinsics,
            left_wall,
            right_wall,
            guidance_maps,
        )
        if open3d_obstacles:
            obstacles.extend(open3d_obstacles)
        else:
            obstacles.extend(self._extract_depth_obstacles(depth_clean, intrinsics, left_wall, right_wall, guidance_maps))
        obstacle_estimates = self._serialize_obstacles(obstacles)
        nearest_obstacle = self._nearest_obstacle(obstacles)
        gaps, blocking_ids, traversable_width = self._estimate_free_space(left_wall, right_wall, obstacles)

        passable = None
        if traversable_width is not None:
            passable = traversable_width >= (self.robot_width_m + self.safety_margin_m)
        elif corridor_width is not None:
            passable = corridor_width >= (self.robot_width_m + self.safety_margin_m)

        return SceneGeometryState(
            reference_frame="robot_base",
            robot_width_m=self.robot_width_m,
            safety_margin_m=self.safety_margin_m,
            left_wall=left_wall,
            right_wall=right_wall,
            left_wall_visible=self._wall_tracks.get("left").visible if self._wall_tracks.get("left") else None,
            right_wall_visible=self._wall_tracks.get("right").visible if self._wall_tracks.get("right") else None,
            left_wall_confidence=self._wall_tracks.get("left").confidence if self._wall_tracks.get("left") else None,
            right_wall_confidence=self._wall_tracks.get("right").confidence if self._wall_tracks.get("right") else None,
            corridor_width_m=corridor_width,
            traversable_width_m=traversable_width,
            passable=passable,
            nearest_obstacle_distance_m=nearest_obstacle.distance_m if nearest_obstacle else None,
            nearest_obstacle=nearest_obstacle,
            obstacles=obstacle_estimates,
            gaps=gaps,
            blocking_obstacle_ids=blocking_ids,
        )

    def reset_tracks(self) -> None:
        self._wall_tracks.clear()
        self._frame_counter = -1

    def _update_wall_track(
        self,
        side: str,
        observed: WallEstimate | None,
        frame_index: int,
    ) -> WallEstimate | None:
        track = self._wall_tracks.get(side)

        if observed is None:
            if track is None:
                return None
            track.miss_count += 1
            track.visible = False
            track.confidence *= 0.82
            if track.miss_count > self.max_wall_misses or track.confidence < 0.2:
                self._wall_tracks.pop(side, None)
                return None
            return track.estimate

        if track is None:
            self._wall_tracks[side] = WallTrack(
                track_id=f"{side}_wall",
                side=side,
                estimate=observed,
                confidence=1.0,
                last_seen_frame=frame_index,
                visible=True,
                miss_count=0,
            )
            return observed

        if self._is_wall_jump_too_large(track.estimate, observed):
            track.miss_count += 1
            track.visible = False
            track.confidence *= 0.85
            if track.miss_count > self.max_wall_misses or track.confidence < 0.2:
                self._wall_tracks[side] = WallTrack(
                    track_id=f"{side}_wall",
                    side=side,
                    estimate=observed,
                    confidence=0.6,
                    last_seen_frame=frame_index,
                    visible=True,
                    miss_count=0,
                )
                return observed
            return track.estimate

        smoothed = WallEstimate(
            side=side,
            lateral_distance_m=round(
                ((1.0 - self.wall_alpha) * track.estimate.lateral_distance_m)
                + (self.wall_alpha * observed.lateral_distance_m),
                3,
            ),
            forward_distance_m=round(
                ((1.0 - self.wall_alpha) * track.estimate.forward_distance_m)
                + (self.wall_alpha * observed.forward_distance_m),
                3,
            ),
            distance_m=round(
                ((1.0 - self.wall_alpha) * track.estimate.distance_m)
                + (self.wall_alpha * observed.distance_m),
                3,
            ),
            valid_point_count=observed.valid_point_count,
            normal_robot_frame=self._blend_wall_normals(track.estimate.normal_robot_frame, observed.normal_robot_frame),
            plane_inlier_count=observed.plane_inlier_count,
            plane_fit_error_m=observed.plane_fit_error_m,
        )
        track.estimate = smoothed
        track.confidence = min(1.0, 0.7 * track.confidence + 0.3)
        track.last_seen_frame = frame_index
        track.visible = True
        track.miss_count = 0
        return smoothed

    def _is_wall_jump_too_large(self, previous: WallEstimate, observed: WallEstimate) -> bool:
        lateral_jump = abs(previous.lateral_distance_m - observed.lateral_distance_m)
        forward_jump = abs(previous.forward_distance_m - observed.forward_distance_m)
        return lateral_jump > self.max_wall_lateral_jump_m or forward_jump > self.max_wall_forward_jump_m

    def _extract_open3d_obstacles(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        left_wall: WallEstimate | None,
        right_wall: WallEstimate | None,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> List[ObservedObject]:
        if left_wall is None or right_wall is None:
            return []

        corridor_left = float(left_wall.lateral_distance_m) + 0.08
        corridor_right = -float(right_wall.lateral_distance_m) - 0.08
        if corridor_left <= corridor_right:
            return []

        pcd, points_robot = depth_to_robot_point_cloud(
            depth_m=depth_m,
            intrinsics=intrinsics,
            stride=self.open3d_stride,
        )
        if len(points_robot) == 0:
            return []

        filtered = remove_floor_and_far_background(
            pcd,
            min_height_m=0.08,
            max_forward_m=self.max_obstacle_forward_m,
        )
        filtered = remove_large_planar_surfaces(filtered)
        if len(filtered.points) == 0:
            return []

        filtered_points = np.asarray(filtered.points)
        corridor_mask = (
            (filtered_points[:, 0] > 0.25)
            & (filtered_points[:, 0] < self.max_obstacle_forward_m)
            & (filtered_points[:, 1] > corridor_right)
            & (filtered_points[:, 1] < corridor_left)
            & (filtered_points[:, 2] > 0.05)
            & (filtered_points[:, 2] < 1.6)
        )
        if int(corridor_mask.sum()) < 80:
            return []

        corridor_cloud = filtered.select_by_index(np.where(corridor_mask)[0].tolist())
        clusters = cluster_obstacles(corridor_cloud, eps=0.10, min_points=80)
        clusters = self._select_open3d_clusters(clusters, intrinsics, guidance_maps)

        obstacles: List[ObservedObject] = []
        for cluster in clusters:
            forward, lateral, vertical = cluster.center_robot_frame
            depth_extent, width_extent, height_extent = cluster.extent_robot_frame
            if width_extent < 0.08 or height_extent < 0.10:
                continue
            if depth_extent > 2.5 or width_extent > 1.6:
                continue

            obstacles.append(
                ObservedObject(
                    object_id=f"open3d_obstacle_{cluster.cluster_id}",
                    category="obstacle",
                    position_robot_frame=(round(forward, 3), round(lateral, 3), round(vertical, 3)),
                    size=(
                        round(max(0.2, depth_extent), 3),
                        round(max(0.1, width_extent), 3),
                        round(max(0.2, height_extent), 3),
                    ),
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    confidence=0.7,
                    attributes={
                        "source": "open3d_cluster",
                        "point_count": str(cluster.point_count),
                    },
                )
            )
        return obstacles

    def _select_open3d_clusters(
        self,
        clusters: List,
        intrinsics: np.ndarray,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> List:
        selected = []
        for cluster in clusters:
            depth_extent, width_extent, height_extent = cluster.extent_robot_frame
            if width_extent < 0.16 and cluster.point_count < 400:
                continue
            if width_extent < 0.24 and cluster.point_count < 700:
                continue
            if height_extent < 0.08:
                continue
            guidance_score = self._sample_guidance_for_cluster(cluster, intrinsics, guidance_maps)
            selected.append((cluster, guidance_score))

        if not selected:
            return []

        selected = sorted(
            selected,
            key=lambda item: (
                -item[1],
                -item[0].point_count,
                -item[0].extent_robot_frame[1],
                item[0].center_robot_frame[0],
            ),
        )
        return [item[0] for item in selected[:2]]

    def _sample_guidance_for_cluster(
        self,
        cluster,
        intrinsics: np.ndarray,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> float:
        if not guidance_maps or "obstacle_guidance_map" not in guidance_maps:
            return 0.0

        guidance_map = guidance_maps["obstacle_guidance_map"]
        forward, lateral, vertical = cluster.center_robot_frame
        u, v = self._project_robot_point_to_image(forward, lateral, vertical, intrinsics, guidance_map.shape)
        if u is None or v is None:
            return 0.0

        x1 = max(0, u - 3)
        x2 = min(guidance_map.shape[1], u + 4)
        y1 = max(0, v - 3)
        y2 = min(guidance_map.shape[0], v + 4)
        patch = guidance_map[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.nanmean(patch))

    def _extract_depth_obstacles(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        left_wall: WallEstimate | None,
        right_wall: WallEstimate | None,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> List[ObservedObject]:
        if left_wall is None or right_wall is None:
            return []

        height, width = depth_m.shape
        y_start = int(height * 0.30)
        y_end = int(height * 0.70)
        roi = depth_m[y_start:y_end, :]
        valid = np.isfinite(roi)
        if int(valid.sum()) < 100:
            return []

        v_coords, u_coords = np.where(valid)
        z = roi[valid]
        u = u_coords.astype(np.float32)
        v = v_coords.astype(np.float32) + y_start

        x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
        lateral = -x_camera
        vertical = -((v - intrinsics[1, 2]) / intrinsics[1, 1]) * z
        forward = z

        left_boundary = float(left_wall.lateral_distance_m) - self.wall_edge_margin_m
        right_boundary = -float(right_wall.lateral_distance_m) + self.wall_edge_margin_m

        point_mask = (
            (forward > 0.25)
            & (forward < self.depth_obstacle_forward_m)
            & (lateral > right_boundary)
            & (lateral < left_boundary)
            & (vertical > -0.2)
            & (vertical < 1.4)
        )
        if guidance_maps and "obstacle_guidance_map" in guidance_maps:
            guidance_map = guidance_maps["obstacle_guidance_map"][y_start:y_end, :]
            guidance_values = guidance_map[valid]
            point_mask = point_mask & (guidance_values > np.nanpercentile(guidance_values, 45))
        if int(point_mask.sum()) < 80:
            return []

        lateral_points = lateral[point_mask]
        forward_points = forward[point_mask]
        interval_bins = self._build_obstacle_bins(lateral_points, forward_points, right_boundary, left_boundary)
        if not interval_bins:
            return []

        obstacles: List[ObservedObject] = []
        for index, (start_bin, end_bin, min_forward) in enumerate(interval_bins):
            start = right_boundary + (start_bin * self.obstacle_bin_width_m)
            end = right_boundary + ((end_bin + 1) * self.obstacle_bin_width_m)
            interval_mask = (
                (lateral_points >= start)
                & (lateral_points < end)
                & (forward_points <= min_forward + 0.35)
            )
            if int(interval_mask.sum()) < 40:
                continue

            interval_lateral = lateral_points[interval_mask]
            interval_forward = forward_points[interval_mask]
            center_lateral = float(np.nanmedian(interval_lateral))
            forward_distance = float(np.nanpercentile(interval_forward, 20))
            width_m = float(max(self.obstacle_bin_width_m, end - start))

            obstacles.append(
                ObservedObject(
                    object_id=f"depth_obstacle_{index}",
                    category="obstacle",
                    position_robot_frame=(round(forward_distance, 3), round(center_lateral, 3), 0.0),
                    size=(0.4, round(width_m, 3), 0.8),
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    confidence=0.6,
                    attributes={"source": "depth_geometry"},
                )
            )
        return obstacles

    def _build_obstacle_bins(
        self,
        lateral_points: np.ndarray,
        forward_points: np.ndarray,
        right_boundary: float,
        left_boundary: float,
    ) -> List[tuple[int, int, float]]:
        corridor_width = left_boundary - right_boundary
        if corridor_width <= self.obstacle_bin_width_m:
            return []

        num_bins = max(1, int(np.ceil(corridor_width / self.obstacle_bin_width_m)))
        bin_indices = np.floor((lateral_points - right_boundary) / self.obstacle_bin_width_m).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        min_counts = 25
        occupied_bins: List[tuple[int, float]] = []
        for bin_idx in range(num_bins):
            mask = bin_indices == bin_idx
            count = int(mask.sum())
            if count < min_counts:
                continue
            forward_value = float(np.nanpercentile(forward_points[mask], 20))
            if forward_value >= self.depth_obstacle_forward_m:
                continue
            occupied_bins.append((bin_idx, forward_value))

        if not occupied_bins:
            return []

        merged: List[tuple[int, int, float]] = []
        current_start, current_end, current_min_forward = occupied_bins[0][0], occupied_bins[0][0], occupied_bins[0][1]
        for bin_idx, forward_value in occupied_bins[1:]:
            if bin_idx <= current_end + 1:
                current_end = bin_idx
                current_min_forward = min(current_min_forward, forward_value)
            else:
                merged.append((current_start, current_end, current_min_forward))
                current_start, current_end, current_min_forward = bin_idx, bin_idx, forward_value
        merged.append((current_start, current_end, current_min_forward))
        return merged

    def _sanitize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        depth_clean = np.asarray(depth_m, dtype=np.float32).copy()
        invalid = ~np.isfinite(depth_clean) | (depth_clean <= 0.05) | (depth_clean >= 10.0)
        depth_clean[invalid] = np.nan
        return depth_clean

    def _extract_corridor_walls(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> List[WallEstimate]:
        estimates: List[WallEstimate] = []
        for side in ("left", "right"):
            estimate = self._estimate_side_wall(depth_m, intrinsics, side, guidance_maps)
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def _estimate_side_wall(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        side: str,
        guidance_maps: dict[str, np.ndarray] | None = None,
    ) -> WallEstimate | None:
        height, width = depth_m.shape
        y_start = int(height * 0.28)
        y_end = int(height * 0.78)
        x_start, x_end = (0, int(width * 0.22)) if side == "left" else (int(width * 0.78), width)

        roi = depth_m[y_start:y_end, x_start:x_end]
        valid = np.isfinite(roi)
        if int(valid.sum()) < 40:
            return None

        v_coords, u_coords = np.where(valid)
        z = roi[valid]
        u = u_coords.astype(np.float32) + x_start
        v = v_coords.astype(np.float32) + y_start
        x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
        y_camera = ((v - intrinsics[1, 2]) / intrinsics[1, 1]) * z

        lateral_values = -x_camera
        vertical_values = -y_camera

        side_mask = x_camera < 0 if side == "left" else x_camera > 0
        if guidance_maps and "wall_guidance_map" in guidance_maps:
            guidance_roi = guidance_maps["wall_guidance_map"][y_start:y_end, x_start:x_end]
            guidance_values = guidance_roi[valid]
            threshold = np.nanpercentile(guidance_values, 35)
            side_mask = side_mask & (guidance_values >= threshold)
        if int(side_mask.sum()) < 25:
            return None

        wall_lateral_samples = lateral_values[side_mask]
        wall_forward_samples = z[side_mask]
        wall_vertical_samples = vertical_values[side_mask]
        wall_u_samples = u[side_mask]
        wall_v_samples = v[side_mask]

        lateral = float(np.nanmedian(np.abs(x_camera[side_mask])))
        forward = float(np.nanmedian(z[side_mask]))
        wall_points = np.stack(
            [wall_forward_samples, wall_lateral_samples, wall_vertical_samples],
            axis=1,
        )
        plane_fit = self._fit_wall_plane(
            wall_points=wall_points,
            u=wall_u_samples,
            v=wall_v_samples,
            side=side,
        )

        return WallEstimate(
            side=side,
            lateral_distance_m=round(lateral, 3),
            forward_distance_m=round(forward, 3),
            distance_m=round(float(np.sqrt(lateral**2 + forward**2)), 3),
            valid_point_count=int(side_mask.sum()),
            normal_robot_frame=plane_fit.normal_robot_frame if plane_fit else None,
            plane_inlier_count=plane_fit.inlier_count if plane_fit else None,
            plane_fit_error_m=round(plane_fit.mean_error_m, 4) if plane_fit else None,
        )

    def _fit_wall_plane(
        self,
        wall_points: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        side: str,
    ):
        if len(wall_points) < 80:
            return None

        lateral_values = wall_points[:, 1]
        lateral_center = float(np.nanmedian(lateral_values))
        lateral_band = np.abs(lateral_values - lateral_center) < 0.18
        lower_v = np.nanpercentile(v, 15)
        upper_v = np.nanpercentile(v, 85)
        vertical_band = (v >= lower_v) & (v <= upper_v)
        support_mask = lateral_band & vertical_band
        support_points = wall_points[support_mask]
        if len(support_points) < 80:
            support_points = wall_points

        plane_fit = fit_plane_ransac(support_points, distance_threshold=0.035, min_points=80)
        if plane_fit is None:
            return None

        raw_normal = np.asarray(plane_fit.normal_robot_frame, dtype=np.float32)
        horizontal_normal = self._estimate_wall_horizontal_normal(support_points, side=side)
        normal = np.asarray([horizontal_normal[0], horizontal_normal[1], raw_normal[2]], dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-6:
            normal = np.asarray(plane_fit.normal_robot_frame, dtype=np.float32)
        else:
            normal = normal / normal_norm

        return plane_fit.__class__(
            normal_robot_frame=(float(normal[0]), float(normal[1]), float(normal[2])),
            offset_d=plane_fit.offset_d,
            inlier_count=plane_fit.inlier_count,
            mean_error_m=plane_fit.mean_error_m,
        )

    def _estimate_wall_horizontal_normal(
        self,
        wall_points: np.ndarray,
        side: str,
    ) -> np.ndarray:
        xy_points = np.asarray(wall_points[:, :2], dtype=np.float32)
        if len(xy_points) < 8:
            return np.asarray([0.0, -1.0], dtype=np.float32) if side == "left" else np.asarray([0.0, 1.0], dtype=np.float32)

        centered = xy_points - xy_points.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(covariance)
        tangent = eigvecs[:, int(np.argmax(eigvals))]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-6:
            return np.asarray([0.0, -1.0], dtype=np.float32) if side == "left" else np.asarray([0.0, 1.0], dtype=np.float32)
        tangent = tangent / tangent_norm

        normal_xy = np.asarray([tangent[1], -tangent[0]], dtype=np.float32)
        reference = np.asarray([0.0, -1.0], dtype=np.float32) if side == "left" else np.asarray([0.0, 1.0], dtype=np.float32)
        if float(np.dot(normal_xy, reference)) < 0.0:
            normal_xy = -normal_xy

        alignment = float(np.dot(normal_xy, reference))
        if alignment < 0.85:
            normal_xy = (0.25 * normal_xy) + (0.75 * reference)

        normal_norm = float(np.linalg.norm(normal_xy))
        if normal_norm < 1e-6:
            return reference
        return normal_xy / normal_norm

    def _blend_wall_normals(
        self,
        previous: tuple[float, float, float] | None,
        current: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        if previous is None:
            return current
        if current is None:
            return previous

        prev = np.asarray(previous, dtype=np.float32)
        curr = np.asarray(current, dtype=np.float32)
        blended = ((1.0 - self.wall_alpha) * prev) + (self.wall_alpha * curr)
        norm = float(np.linalg.norm(blended))
        if norm < 1e-6:
            return current
        blended = blended / norm
        return (float(blended[0]), float(blended[1]), float(blended[2]))

    def _project_robot_point_to_image(
        self,
        forward: float,
        lateral: float,
        vertical: float,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[int | None, int | None]:
        if forward <= 0.05:
            return None, None

        x_camera = -lateral
        y_camera = -vertical
        u = int(round((intrinsics[0, 0] * x_camera / forward) + intrinsics[0, 2]))
        v = int(round((intrinsics[1, 1] * y_camera / forward) + intrinsics[1, 2]))
        height, width = image_shape
        if u < 0 or u >= width or v < 0 or v >= height:
            return None, None
        return u, v

    def _nearest_obstacle(self, objects: List[ObservedObject]) -> ObstacleEstimate | None:
        if not objects:
            return None

        nearest = min(objects, key=lambda item: euclidean_distance_3d(item.position_robot_frame))
        return self._serialize_observed_object(nearest)

    def _serialize_obstacles(self, objects: List[ObservedObject]) -> List[ObstacleEstimate]:
        return [self._serialize_observed_object(obj) for obj in objects]

    def _serialize_observed_object(self, obj: ObservedObject) -> ObstacleEstimate:
        return ObstacleEstimate(
            object_id=obj.object_id,
            category=obj.category,
            forward_distance_m=round(float(obj.position_robot_frame[0]), 3),
            lateral_offset_m=round(float(obj.position_robot_frame[1]), 3),
            distance_m=round(euclidean_distance_3d(obj.position_robot_frame), 3),
            width_m=round(float(obj.size[1]), 3),
            source=obj.attributes.get("source", "detector"),
        )

    def _estimate_free_space(
        self,
        left_wall: WallEstimate | None,
        right_wall: WallEstimate | None,
        objects: List[ObservedObject],
    ) -> tuple[List[CorridorGap], List[str], float | None]:
        if left_wall is None or right_wall is None:
            return [], [], None

        left_boundary = float(left_wall.lateral_distance_m)
        right_boundary = -float(right_wall.lateral_distance_m)

        occupied: List[tuple[float, float, str]] = []
        blocking_ids: List[str] = []
        for obj in objects:
            forward, lateral, _ = obj.position_robot_frame
            if forward <= 0.0 or forward > self.max_obstacle_forward_m:
                continue

            half_width = max(0.05, float(obj.size[1]) / 2.0)
            start = max(right_boundary, float(lateral) - half_width)
            end = min(left_boundary, float(lateral) + half_width)
            if end <= start:
                continue

            occupied.append((start, end, obj.object_id))
            blocking_ids.append(obj.object_id)

        merged = self._merge_intervals(occupied)
        gaps = self._compute_gaps(right_boundary, left_boundary, merged)
        if not gaps:
            return [], blocking_ids, 0.0

        traversable_width = round(max(gap.width_m for gap in gaps), 3)
        return gaps, blocking_ids, traversable_width

    def _merge_intervals(self, intervals: List[tuple[float, float, str]]) -> List[tuple[float, float]]:
        if not intervals:
            return []

        sorted_intervals = sorted((start, end) for start, end, _ in intervals)
        merged: List[tuple[float, float]] = [sorted_intervals[0]]

        for start, end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _compute_gaps(
        self,
        right_boundary: float,
        left_boundary: float,
        occupied_intervals: List[tuple[float, float]],
    ) -> List[CorridorGap]:
        gaps: List[CorridorGap] = []
        cursor = right_boundary

        for start, end in occupied_intervals:
            if start > cursor:
                gaps.append(
                    CorridorGap(
                        start_m=round(cursor, 3),
                        end_m=round(start, 3),
                        width_m=round(start - cursor, 3),
                    )
                )
            cursor = max(cursor, end)

        if cursor < left_boundary:
            gaps.append(
                CorridorGap(
                    start_m=round(cursor, 3),
                    end_m=round(left_boundary, 3),
                    width_m=round(left_boundary - cursor, 3),
                )
            )
        return gaps
