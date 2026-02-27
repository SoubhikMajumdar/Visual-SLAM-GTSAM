"""Trajectory evaluation utilities for KITTI SLAM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tabulate import tabulate


def _translation(pose: np.ndarray) -> np.ndarray:
    """Extract translation vector from SE(3) pose."""
    return pose[:3, 3]


def _relative_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute relative transform from a to b."""
    return np.linalg.inv(a) @ b


def _rotation_angle(r: np.ndarray) -> float:
    """Compute SO(3) angle in radians from rotation matrix."""
    value = (np.trace(r) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.arccos(value))


@dataclass(slots=True)
class ErrorStats:
    """Summary stats for one scalar trajectory error metric."""

    mean: float
    median: float
    rmse: float


class TrajectoryEvaluator:
    """Compute ATE and RPE metrics for estimated trajectories."""

    @staticmethod
    def _to_positions(poses: list[np.ndarray]) -> np.ndarray:
        """Convert pose list to Nx3 positions."""
        if len(poses) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([_translation(p) for p in poses])

    @staticmethod
    def _stats(values: np.ndarray) -> ErrorStats:
        """Compute mean, median and RMSE from values."""
        if values.size == 0:
            return ErrorStats(mean=0.0, median=0.0, rmse=0.0)
        return ErrorStats(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            rmse=float(np.sqrt(np.mean(values**2))),
        )

    @classmethod
    def compute_ate(cls, ground_truth: list[np.ndarray], estimated: list[np.ndarray]) -> ErrorStats:
        """Compute absolute trajectory error over aligned timestamps."""
        n = min(len(ground_truth), len(estimated))
        gt = cls._to_positions(ground_truth[:n])
        est = cls._to_positions(estimated[:n])
        errors = np.linalg.norm(gt - est, axis=1)
        return cls._stats(errors)

    @classmethod
    def compute_rpe(
        cls,
        ground_truth: list[np.ndarray],
        estimated: list[np.ndarray],
        delta: int = 1,
    ) -> tuple[ErrorStats, ErrorStats]:
        """Compute relative pose error (translation and rotation)."""
        n = min(len(ground_truth), len(estimated))
        trans_errors: list[float] = []
        rot_errors: list[float] = []

        for i in range(0, n - delta):
            gt_rel = _relative_pose(ground_truth[i], ground_truth[i + delta])
            est_rel = _relative_pose(estimated[i], estimated[i + delta])
            err = np.linalg.inv(gt_rel) @ est_rel
            trans_errors.append(float(np.linalg.norm(err[:3, 3])))
            rot_errors.append(_rotation_angle(err[:3, :3]))

        return cls._stats(np.array(trans_errors)), cls._stats(np.array(rot_errors))

    @classmethod
    def evaluate(
        cls,
        ground_truth: list[np.ndarray],
        estimated: list[np.ndarray],
        delta: int = 1,
    ) -> dict[str, ErrorStats]:
        """Evaluate trajectory and return all metrics."""
        ate = cls.compute_ate(ground_truth, estimated)
        rpe_t, rpe_r = cls.compute_rpe(ground_truth, estimated, delta=delta)
        return {
            "ate": ate,
            "rpe_trans": rpe_t,
            "rpe_rot": rpe_r,
        }

    @staticmethod
    def format_metrics_table(results_by_sequence: dict[str, dict[str, ErrorStats]]) -> str:
        """Render metrics table for multiple sequences."""
        rows: list[list[float | str]] = []
        for seq, metrics in sorted(results_by_sequence.items()):
            ate = metrics["ate"]
            rpe_t = metrics["rpe_trans"]
            rpe_r = metrics["rpe_rot"]
            rows.append(
                [
                    seq,
                    ate.mean,
                    ate.median,
                    ate.rmse,
                    rpe_t.mean,
                    rpe_t.median,
                    rpe_t.rmse,
                    rpe_r.mean,
                    rpe_r.median,
                    rpe_r.rmse,
                ]
            )

        return tabulate(
            rows,
            headers=[
                "Sequence",
                "ATE Mean",
                "ATE Median",
                "ATE RMSE",
                "RPE_t Mean",
                "RPE_t Median",
                "RPE_t RMSE",
                "RPE_r Mean(rad)",
                "RPE_r Median(rad)",
                "RPE_r RMSE(rad)",
            ],
            floatfmt=".4f",
            tablefmt="github",
        )
