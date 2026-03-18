from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

import numpy as np
from PIL import Image


@dataclass
class OpenLORISFrame:
    frame_index: int
    color_timestamp: float
    depth_timestamp: float
    color_path: Path
    aligned_depth_path: Path
    groundtruth: np.ndarray | None


class OpenLORISLoader:
    """Lightweight loader for OpenLORIS packaged RGB-D sequences."""

    def load_sequence(self, sequence_dir: Path) -> List[OpenLORISFrame]:
        sequence_dir = Path(sequence_dir)
        color_entries = self._load_index(sequence_dir / "color.txt")
        depth_entries = self._load_index(sequence_dir / "aligned_depth.txt")
        groundtruth_entries = self._load_groundtruth(sequence_dir / "groundtruth.txt")

        frames: List[OpenLORISFrame] = []
        for frame_index, (color_ts, color_relpath) in enumerate(color_entries):
            depth_ts, depth_relpath = self._nearest_entry(depth_entries, color_ts)
            gt = self._nearest_groundtruth(groundtruth_entries, color_ts)
            frames.append(
                OpenLORISFrame(
                    frame_index=frame_index,
                    color_timestamp=color_ts,
                    depth_timestamp=depth_ts,
                    color_path=sequence_dir / color_relpath,
                    aligned_depth_path=sequence_dir / depth_relpath,
                    groundtruth=gt,
                )
            )
        return frames

    def load_color_rgb(self, frame: OpenLORISFrame) -> np.ndarray:
        return np.array(Image.open(frame.color_path).convert("RGB"), dtype=np.uint8)

    def load_aligned_depth_meters(self, frame: OpenLORISFrame) -> np.ndarray:
        depth_raw = np.array(Image.open(frame.aligned_depth_path), dtype=np.uint16)
        depth_m = depth_raw.astype(np.float32) / 1000.0
        invalid = (depth_m <= 0.05) | (depth_m >= 10.0)
        depth_m[invalid] = np.nan
        return depth_m

    def load_color_intrinsics(self, sequence_dir: Path) -> np.ndarray:
        return self._load_intrinsics(sequence_dir, sensor_name="d400_color_optical_frame")

    def _load_intrinsics(self, sequence_dir: Path, sensor_name: str) -> np.ndarray:
        text = (Path(sequence_dir) / "sensors.yaml").read_text(encoding="utf-8")
        section_pattern = rf"{re.escape(sensor_name)}:\s*(.*?)(?:\n\S|\Z)"
        section_match = re.search(section_pattern, text, re.DOTALL)
        if not section_match:
            raise ValueError(f"Sensor block not found: {sensor_name}")

        section = section_match.group(1)
        data_match = re.search(r"data:\s*\[\s*([^\]]+)\]", section)
        if not data_match:
            raise ValueError(f"Intrinsics not found for sensor: {sensor_name}")

        values = [float(item.strip()) for item in data_match.group(1).split(",")]
        if len(values) != 4:
            raise ValueError(f"Unexpected intrinsics format for sensor: {sensor_name}")

        fx, cx, fy, cy = values
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _load_index(self, path: Path) -> List[tuple[float, str]]:
        entries: List[tuple[float, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            timestamp_str, relpath = line.split(maxsplit=1)
            entries.append((float(timestamp_str), relpath.strip()))
        return entries

    def _load_groundtruth(self, path: Path) -> List[tuple[float, np.ndarray]]:
        entries: List[tuple[float, np.ndarray]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            timestamp = float(parts[0])
            values = np.array([float(item) for item in parts[1:]], dtype=np.float32)
            entries.append((timestamp, values))
        return entries

    def _nearest_entry(self, entries: List[tuple[float, str]], target: float) -> tuple[float, str]:
        if not entries:
            raise ValueError("No indexed entries found.")
        return min(entries, key=lambda item: abs(item[0] - target))

    def _nearest_groundtruth(
        self,
        entries: List[tuple[float, np.ndarray]],
        target: float,
    ) -> np.ndarray | None:
        if not entries:
            return None
        return min(entries, key=lambda item: abs(item[0] - target))[1]
