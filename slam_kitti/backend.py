"""Backend pose-graph optimization with GTSAM iSAM2."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np


def _np_to_pose3(transform: np.ndarray) -> gtsam.Pose3:
    """Convert 4x4 homogeneous matrix to GTSAM Pose3."""
    rotation = gtsam.Rot3(transform[:3, :3])
    translation = gtsam.Point3(transform[0, 3], transform[1, 3], transform[2, 3])
    return gtsam.Pose3(rotation, translation)


def _pose3_to_np(pose: gtsam.Pose3) -> np.ndarray:
    """Convert GTSAM Pose3 to 4x4 homogeneous matrix."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose.rotation().matrix()
    transform[:3, 3] = np.array([pose.x(), pose.y(), pose.z()], dtype=np.float64)
    return transform


@dataclass(slots=True)
class BackendConfig:
    """Configuration for backend optimization and loop closure."""

    prior_noise_sigmas: tuple[float, float, float, float, float, float]
    odom_noise_sigmas: tuple[float, float, float, float, float, float]
    loop_noise_sigmas: tuple[float, float, float, float, float, float]
    loop_similarity_threshold: float
    loop_min_separation: int


class LoopClosureDetector:
    """Simple descriptor-vector similarity loop closure detector."""

    def __init__(self, similarity_threshold: float, min_separation: int) -> None:
        """Initialize detection thresholds."""
        self.similarity_threshold = similarity_threshold
        self.min_separation = min_separation

    @staticmethod
    def descriptor_embedding(descriptors: np.ndarray | None) -> np.ndarray | None:
        """Create a normalized embedding from ORB descriptors."""
        if descriptors is None or len(descriptors) == 0:
            return None
        vector = descriptors.astype(np.float32).mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return None
        return vector / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    def detect(
        self,
        current_id: int,
        current_descriptors: np.ndarray | None,
        history: dict[int, np.ndarray | None],
    ) -> int | None:
        """Find loop closure candidate keyframe id, if any."""
        current_emb = self.descriptor_embedding(current_descriptors)
        if current_emb is None:
            return None

        best_id: int | None = None
        best_score = self.similarity_threshold
        for keyframe_id, descriptors in history.items():
            if current_id - keyframe_id < self.min_separation:
                continue
            emb = self.descriptor_embedding(descriptors)
            if emb is None:
                continue
            score = self.cosine_similarity(current_emb, emb)
            if score > best_score:
                best_score = score
                best_id = keyframe_id
        return best_id


class SLAMBackend:
    """Incremental pose-graph optimizer using iSAM2."""

    def __init__(self, config: BackendConfig) -> None:
        """Initialize GTSAM factor graph and noise models."""
        self.config = config
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        isam_params = gtsam.ISAM2Params()
        isam_params.setFactorization("CHOLESKY")
        self.isam = gtsam.ISAM2(isam_params)

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.prior_noise_sigmas))
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.odom_noise_sigmas))
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(config.loop_noise_sigmas))

        self.optimized: dict[int, np.ndarray] = {}
        self.pose_ids: set[int] = set()
        self.initialized_ids: set[int] = set()

    @staticmethod
    def _x(keyframe_id: int) -> int:
        """Generate integer key for pose variable."""
        return gtsam.symbol("x", keyframe_id)

    def add_prior(self, keyframe_id: int, pose: np.ndarray) -> None:
        """Add prior constraint for first keyframe."""
        self.graph.add(gtsam.PriorFactorPose3(self._x(keyframe_id), _np_to_pose3(pose), self.prior_noise))
        self.pose_ids.add(keyframe_id)
        if keyframe_id not in self.initialized_ids:
            self.initial.insert(self._x(keyframe_id), _np_to_pose3(pose))
            self.initialized_ids.add(keyframe_id)

    def add_odometry_factor(self, from_id: int, to_id: int, relative_pose: np.ndarray) -> None:
        """Add between factor for consecutive keyframes."""
        self.graph.add(
            gtsam.BetweenFactorPose3(
                self._x(from_id),
                self._x(to_id),
                _np_to_pose3(relative_pose),
                self.odom_noise,
            )
        )

    def add_loop_factor(self, from_id: int, to_id: int, relative_pose: np.ndarray) -> None:
        """Add loop closure factor between distant keyframes."""
        self.graph.add(
            gtsam.BetweenFactorPose3(
                self._x(from_id),
                self._x(to_id),
                _np_to_pose3(relative_pose),
                self.loop_noise,
            )
        )

    def add_initial_estimate(self, keyframe_id: int, pose: np.ndarray) -> None:
        """Provide initial estimate for a new keyframe variable."""
        key = self._x(keyframe_id)
        self.pose_ids.add(keyframe_id)
        if keyframe_id not in self.initialized_ids:
            self.initial.insert(key, _np_to_pose3(pose))
            self.initialized_ids.add(keyframe_id)

    def update(self) -> None:
        """Run one incremental optimization step and cache estimates."""
        self.isam.update(self.graph, self.initial)
        result = self.isam.calculateEstimate()

        self.optimized.clear()
        for keyframe_id in sorted(self.pose_ids):
            key = self._x(keyframe_id)
            if not result.exists(key):
                continue
            self.optimized[keyframe_id] = _pose3_to_np(result.atPose3(key))

        self.graph.resize(0)
        self.initial.clear()

    def get_pose(self, keyframe_id: int) -> np.ndarray | None:
        """Get optimized keyframe pose if available."""
        return self.optimized.get(keyframe_id)
