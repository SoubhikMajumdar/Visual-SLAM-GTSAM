"""Unit tests for KITTI dataset loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from slam_kitti.dataset import KITTIOdometryDataset


class TestKITTIOdometryDataset(unittest.TestCase):
    """Tests for parsing calibration, images, and poses."""

    def setUp(self) -> None:
        """Create a tiny synthetic KITTI-like folder structure."""
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        seq_dir = root / "sequences" / "00"
        left_dir = seq_dir / "image_0"
        right_dir = seq_dir / "image_1"
        poses_dir = root / "poses"

        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)

        calib = (
            "P0: 718.856 0 607.1928 0 0 718.856 185.2157 0 0 0 1 0\n"
            "P1: 718.856 0 607.1928 -387.5744 0 718.856 185.2157 0 0 0 1 0\n"
        )
        (seq_dir / "calib.txt").write_text(calib, encoding="utf-8")

        pose_line_0 = "1 0 0 0 0 1 0 0 0 0 1 0\n"
        pose_line_1 = "1 0 0 1 0 1 0 0 0 0 1 0\n"
        (poses_dir / "00.txt").write_text(pose_line_0 + pose_line_1, encoding="utf-8")

        image = np.full((20, 30), 120, dtype=np.uint8)
        cv2.imwrite(str(left_dir / "000000.png"), image)
        cv2.imwrite(str(right_dir / "000000.png"), image)
        cv2.imwrite(str(left_dir / "000001.png"), image)
        cv2.imwrite(str(right_dir / "000001.png"), image)

        self.root = root

    def tearDown(self) -> None:
        """Release temporary directory."""
        self.tmp.cleanup()

    def test_loads_frames_and_calibration(self) -> None:
        """Dataset should parse calib, baseline, images and poses."""
        dataset = KITTIOdometryDataset(self.root, "00")

        self.assertEqual(len(dataset), 2)
        self.assertGreater(dataset.baseline, 0.0)
        self.assertEqual(dataset.intrinsics.shape, (3, 3))

        frame0 = dataset.get_frame(0)
        self.assertEqual(frame0.index, 0)
        self.assertEqual(frame0.left.shape, (20, 30))
        self.assertEqual(frame0.right.shape, (20, 30))
        self.assertIsNotNone(frame0.gt_pose)

        frame1 = dataset.get_frame(1)
        self.assertIsNotNone(frame1.gt_pose)
        self.assertAlmostEqual(float(frame1.gt_pose[0, 3]), 1.0)

    def test_iter_frames_respects_max_frames(self) -> None:
        """Frame iterator should stop at max_frames when provided."""
        dataset = KITTIOdometryDataset(self.root, "00")
        frames = list(dataset.iter_frames(max_frames=1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].index, 0)


if __name__ == "__main__":
    unittest.main()
