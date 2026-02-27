"""Unit tests for trajectory evaluation metrics."""

from __future__ import annotations

import unittest

import numpy as np

from slam_kitti.evaluation import TrajectoryEvaluator


def _pose(tx: float) -> np.ndarray:
    """Create identity-rotation SE(3) pose at x=tx."""
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = tx
    return transform


class TestTrajectoryEvaluator(unittest.TestCase):
    """Tests for ATE/RPE computations and formatting."""

    def test_ate_zero_for_identical_trajectories(self) -> None:
        """ATE should be exactly zero for matching trajectories."""
        gt = [_pose(0.0), _pose(1.0), _pose(2.0)]
        est = [_pose(0.0), _pose(1.0), _pose(2.0)]
        ate = TrajectoryEvaluator.compute_ate(gt, est)
        self.assertAlmostEqual(ate.mean, 0.0)
        self.assertAlmostEqual(ate.median, 0.0)
        self.assertAlmostEqual(ate.rmse, 0.0)

    def test_rpe_nonzero_for_relative_scale_error(self) -> None:
        """RPE translation should increase when relative motion is wrong."""
        gt = [_pose(0.0), _pose(1.0), _pose(2.0)]
        est = [_pose(0.0), _pose(2.0), _pose(4.0)]
        rpe_t, rpe_r = TrajectoryEvaluator.compute_rpe(gt, est, delta=1)
        self.assertGreater(rpe_t.mean, 0.0)
        self.assertAlmostEqual(rpe_r.mean, 0.0)

    def test_metrics_table_contains_sequence(self) -> None:
        """Formatted metrics table should include sequence and headers."""
        gt = [_pose(0.0), _pose(1.0)]
        est = [_pose(0.0), _pose(1.0)]
        metrics = TrajectoryEvaluator.evaluate(gt, est)
        table = TrajectoryEvaluator.format_metrics_table({"00": metrics})
        self.assertIn("Sequence", table)
        self.assertIn("00", table)


if __name__ == "__main__":
    unittest.main()
