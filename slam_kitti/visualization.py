"""Visualization and trajectory export helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
except Exception:
    o3d = None


class Open3DMapViewer:
    """Real-time 3D map and trajectory viewer backed by Open3D."""

    def __init__(self, enabled: bool = True) -> None:
        """Initialize Open3D visualizer when enabled."""
        self.enabled = bool(enabled and (o3d is not None))
        self.vis = None
        self.pcd = None
        self.traj = None

        if self.enabled:
            assert o3d is not None
            self.vis = o3d.visualization.Visualizer()
            self.pcd = o3d.geometry.PointCloud()
            self.traj = o3d.geometry.LineSet()
            self.vis.create_window(window_name="VI-SLAM-KITTI", width=1280, height=720)
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.traj)

    def update(self, points: np.ndarray, poses: list[np.ndarray]) -> None:
        """Update map points and camera trajectory rendering."""
        if not self.enabled or self.vis is None or self.pcd is None or self.traj is None:
            return

        if points.size > 0:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            colors = np.tile(np.array([[0.2, 0.8, 0.3]], dtype=np.float64), (len(points), 1))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        if len(poses) > 1:
            trajectory = np.vstack([pose[:3, 3] for pose in poses])
            lines = np.array([[i, i + 1] for i in range(len(trajectory) - 1)], dtype=np.int32)
            self.traj.points = o3d.utility.Vector3dVector(trajectory)
            self.traj.lines = o3d.utility.Vector2iVector(lines)
            self.traj.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float64), (len(lines), 1))
            )

        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.traj)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        """Close Open3D window."""
        if self.vis is not None:
            self.vis.destroy_window()


class TrajectoryPlotter:
    """2D plotting helper for estimated vs ground-truth trajectories."""

    @staticmethod
    def plot_topdown(
        estimated: list[np.ndarray],
        ground_truth: list[np.ndarray] | None,
        out_path: str | Path,
    ) -> None:
        """Save top-down XZ trajectory comparison plot."""
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        est_xyz = np.vstack([pose[:3, 3] for pose in estimated]) if estimated else np.empty((0, 3))
        gt_xyz = (
            np.vstack([pose[:3, 3] for pose in ground_truth])
            if ground_truth is not None and len(ground_truth) > 0
            else np.empty((0, 3))
        )

        plt.figure(figsize=(8, 6))
        if len(est_xyz) > 0:
            plt.plot(est_xyz[:, 0], est_xyz[:, 2], label="Estimated", linewidth=2)
        if len(gt_xyz) > 0:
            plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground Truth", linewidth=2)
        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")
        plt.title("Top-down Trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close()


def save_point_cloud_ply(points: np.ndarray, out_path: str | Path) -> None:
    """Save point cloud to PLY file."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(str(out_file), pcd)
        return

    pts = points.astype(np.float64) if points.size > 0 else np.empty((0, 3), dtype=np.float64)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with out_file.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for x, y, z in pts:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_tum_trajectory(poses: list[np.ndarray], out_path: str | Path) -> None:
    """Save trajectory in TUM format (timestamp tx ty tz qx qy qz qw)."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as handle:
        for idx, pose in enumerate(poses):
            t = pose[:3, 3]
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            handle.write(
                f"{idx:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )
