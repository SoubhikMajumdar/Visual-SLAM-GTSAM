"""KITTI odometry dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class StereoFrame:
    """Container for a KITTI stereo frame."""

    sequence: str
    index: int
    timestamp: float
    left: np.ndarray
    right: np.ndarray
    intrinsics: np.ndarray
    baseline: float
    gt_pose: np.ndarray | None = None


class KITTIOdometryDataset:
    """Reader for KITTI Odometry stereo sequences and ground truth poses."""

    def __init__(self, root: str | Path, sequence: str, use_color: bool = False) -> None:
        """Initialize paths and load calibration/ground truth.

        Args:
            root: KITTI odometry root folder.
            sequence: Sequence id in range 00-10.
            use_color: Whether to load color images instead of grayscale.
        """
        self.root = Path(root)
        self.sequence = f"{int(sequence):02d}"
        self.use_color = use_color

        self.seq_dir = self.root / "sequences" / self.sequence
        self.left_dir = self.seq_dir / "image_0"
        self.right_dir = self.seq_dir / "image_1"
        self.calib_path = self.seq_dir / "calib.txt"
        self.pose_path = self.root / "poses" / f"{self.sequence}.txt"

        if not self.seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {self.seq_dir}")
        if not self.calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")

        self.left_images = sorted(self.left_dir.glob("*.png"))
        self.right_images = sorted(self.right_dir.glob("*.png"))
        if len(self.left_images) == 0 or len(self.left_images) != len(self.right_images):
            raise ValueError("Stereo image folders are empty or inconsistent")

        self.p0, self.p1, self.intrinsics, self.baseline = self._load_calibration(self.calib_path)
        self.gt_poses = self._load_gt_poses(self.pose_path) if self.pose_path.exists() else []

    @staticmethod
    def _load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Parse KITTI camera projection matrices and derive intrinsics/baseline."""
        data: dict[str, np.ndarray] = {}
        with calib_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, values = line.strip().split(":", maxsplit=1)
                floats = np.fromstring(values.strip(), sep=" ", dtype=np.float64)
                if floats.size == 12:
                    data[key] = floats.reshape(3, 4)

        if "P0" not in data or "P1" not in data:
            raise ValueError("Calibration must include P0 and P1")

        p0 = data["P0"]
        p1 = data["P1"]
        intrinsics = p0[:, :3].copy()
        tx0 = p0[0, 3] / p0[0, 0]
        tx1 = p1[0, 3] / p1[0, 0]
        baseline = abs(tx1 - tx0)
        return p0, p1, intrinsics, float(baseline)

    @staticmethod
    def _load_gt_poses(pose_path: Path) -> list[np.ndarray]:
        """Load KITTI ground-truth poses (3x4) into homogeneous transforms."""
        poses: list[np.ndarray] = []
        with pose_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                vals = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
                if vals.size != 12:
                    continue
                t = np.eye(4, dtype=np.float64)
                t[:3, :4] = vals.reshape(3, 4)
                poses.append(t)
        return poses

    def __len__(self) -> int:
        """Return number of stereo frames in this sequence."""
        return len(self.left_images)

    def get_frame(self, index: int) -> StereoFrame:
        """Load and return a stereo frame by index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self) - 1}]")

        flag = cv2.IMREAD_COLOR if self.use_color else cv2.IMREAD_GRAYSCALE
        left = cv2.imread(str(self.left_images[index]), flag)
        right = cv2.imread(str(self.right_images[index]), flag)
        if left is None or right is None:
            raise IOError(f"Failed reading stereo image at index {index}")

        gt_pose = self.gt_poses[index] if index < len(self.gt_poses) else None
        return StereoFrame(
            sequence=self.sequence,
            index=index,
            timestamp=float(index),
            left=left,
            right=right,
            intrinsics=self.intrinsics.copy(),
            baseline=self.baseline,
            gt_pose=gt_pose,
        )

    def iter_frames(self, max_frames: int = -1) -> Iterator[StereoFrame]:
        """Yield stereo frames in sequence order."""
        limit = len(self) if max_frames <= 0 else min(max_frames, len(self))
        for index in range(limit):
            yield self.get_frame(index)
