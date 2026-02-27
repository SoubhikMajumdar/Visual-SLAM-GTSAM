"""Visual frontend for stereo visual-inertial style SLAM (vision-only core)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class FrontendConfig:
    """Configuration for the visual frontend."""

    num_features: int = 2000
    scale_factor: float = 1.2
    n_levels: int = 8
    fast_threshold: int = 20
    stereo_match_ratio: float = 0.75
    temporal_match_ratio: float = 0.8
    parallax_threshold_px: float = 15.0
    min_keyframe_features: int = 120
    pnp_reproj_error: float = 3.0


@dataclass(slots=True)
class TrackingResult:
    """Output of frame-to-frame tracking."""

    success: bool
    relative_pose: np.ndarray
    inlier_count: int
    curr_keypoints: list[cv2.KeyPoint]
    curr_descriptors: np.ndarray | None
    tracked_curr_points: np.ndarray
    parallax_px: float


class VisualFrontend:
    """Stereo frontend using ORB + FLANN + PnP RANSAC."""

    def __init__(self, intrinsics: np.ndarray, baseline: float, config: FrontendConfig) -> None:
        """Create detector/matcher and store camera model."""
        self.k = intrinsics.astype(np.float64)
        self.baseline = float(baseline)
        self.config = config

        self.orb = cv2.ORB_create(
            nfeatures=config.num_features,
            scaleFactor=config.scale_factor,
            nlevels=config.n_levels,
            fastThreshold=config.fast_threshold,
        )

        index_params = {
            "algorithm": 6,
            "table_number": 6,
            "key_size": 12,
            "multi_probe_level": 1,
        }
        self.flann = cv2.FlannBasedMatcher(index_params, {"checks": 50})

    def detect_and_compute(self, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Detect ORB features and compute descriptors."""
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_descriptors(
        self,
        descriptors_a: np.ndarray | None,
        descriptors_b: np.ndarray | None,
        ratio: float,
    ) -> list[cv2.DMatch]:
        """Perform FLANN kNN matching with ratio test."""
        if descriptors_a is None or descriptors_b is None:
            return []
        if len(descriptors_a) < 2 or len(descriptors_b) < 2:
            return []

        raw_matches = self.flann.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def bootstrap_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None, dict[int, np.ndarray]]:
        """Initialize 3D landmarks from a stereo pair by disparity triangulation."""
        kp_left, desc_left = self.detect_and_compute(left)
        kp_right, desc_right = self.detect_and_compute(right)
        matches_lr = self.match_descriptors(desc_left, desc_right, self.config.stereo_match_ratio)

        fx = self.k[0, 0]
        fy = self.k[1, 1]
        cx = self.k[0, 2]
        cy = self.k[1, 2]

        landmarks: dict[int, np.ndarray] = {}
        for match in matches_lr:
            left_idx = match.queryIdx
            right_idx = match.trainIdx
            ul, vl = kp_left[left_idx].pt
            ur, _ = kp_right[right_idx].pt
            disparity = ul - ur
            if disparity <= 1.0:
                continue
            z = fx * self.baseline / disparity
            x = (ul - cx) * z / fx
            y = (vl - cy) * z / fy
            landmarks[left_idx] = np.array([x, y, z], dtype=np.float64)

        return kp_left, desc_left, landmarks

    def _compute_essential_inliers(
        self,
        prev_kp: list[cv2.KeyPoint],
        curr_kp: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ) -> set[int]:
        """Estimate essential matrix with RANSAC and return inlier match indices."""
        if len(matches) < 8:
            return set()

        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
        _, mask = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            self.k,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if mask is None:
            return set()
        return {idx for idx, valid in enumerate(mask.ravel().tolist()) if valid}

    @staticmethod
    def _to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert Rodrigues vector and translation into homogeneous transform."""
        rmat, _ = cv2.Rodrigues(rvec)
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = rmat
        t[:3, 3] = tvec.ravel()
        return t

    def track_frame(
        self,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
        prev_keypoints: list[cv2.KeyPoint],
        prev_descriptors: np.ndarray | None,
        prev_landmarks: dict[int, np.ndarray],
    ) -> TrackingResult:
        """Track from previous keyframe to current frame using PnP+RANSAC."""
        curr_keypoints, curr_descriptors = self.detect_and_compute(curr_image)
        matches = self.match_descriptors(
            prev_descriptors, curr_descriptors, self.config.temporal_match_ratio
        )

        if len(matches) < 8:
            return TrackingResult(
                success=False,
                relative_pose=np.eye(4, dtype=np.float64),
                inlier_count=0,
                curr_keypoints=curr_keypoints,
                curr_descriptors=curr_descriptors,
                tracked_curr_points=np.empty((0, 2), dtype=np.float32),
                parallax_px=0.0,
            )

        essential_inliers = self._compute_essential_inliers(prev_keypoints, curr_keypoints, matches)

        object_points: list[np.ndarray] = []
        image_points: list[tuple[float, float]] = []
        prev_points_px: list[tuple[float, float]] = []

        for idx, match in enumerate(matches):
            if idx not in essential_inliers:
                continue
            if match.queryIdx not in prev_landmarks:
                continue
            object_points.append(prev_landmarks[match.queryIdx])
            image_points.append(curr_keypoints[match.trainIdx].pt)
            prev_points_px.append(prev_keypoints[match.queryIdx].pt)

        if len(object_points) < 6:
            return TrackingResult(
                success=False,
                relative_pose=np.eye(4, dtype=np.float64),
                inlier_count=0,
                curr_keypoints=curr_keypoints,
                curr_descriptors=curr_descriptors,
                tracked_curr_points=np.empty((0, 2), dtype=np.float32),
                parallax_px=0.0,
            )

        obj = np.asarray(object_points, dtype=np.float64)
        img = np.asarray(image_points, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=self.k,
            distCoeffs=None,
            iterationsCount=200,
            reprojectionError=self.config.pnp_reproj_error,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok or inliers is None or len(inliers) < 6:
            return TrackingResult(
                success=False,
                relative_pose=np.eye(4, dtype=np.float64),
                inlier_count=0,
                curr_keypoints=curr_keypoints,
                curr_descriptors=curr_descriptors,
                tracked_curr_points=np.empty((0, 2), dtype=np.float32),
                parallax_px=0.0,
            )

        inlier_idx = inliers.ravel()
        prev_px = np.asarray(prev_points_px, dtype=np.float32)[inlier_idx]
        curr_px = np.asarray(image_points, dtype=np.float32)[inlier_idx]
        parallax = float(np.median(np.linalg.norm(curr_px - prev_px, axis=1)))

        return TrackingResult(
            success=True,
            relative_pose=self._to_transform(rvec, tvec),
            inlier_count=int(len(inlier_idx)),
            curr_keypoints=curr_keypoints,
            curr_descriptors=curr_descriptors,
            tracked_curr_points=curr_px,
            parallax_px=parallax,
        )

    def should_add_keyframe(self, tracking: TrackingResult) -> bool:
        """Decide keyframe insertion from tracking quality/parallax."""
        return (
            tracking.parallax_px >= self.config.parallax_threshold_px
            or tracking.inlier_count < self.config.min_keyframe_features
        )

    def estimate_relative_pose_between_keyframes(
        self,
        keypoints_a: list[cv2.KeyPoint],
        descriptors_a: np.ndarray | None,
        keypoints_b: list[cv2.KeyPoint],
        descriptors_b: np.ndarray | None,
    ) -> np.ndarray | None:
        """Estimate relative transform from 2D-2D correspondences for loop constraints."""
        matches = self.match_descriptors(descriptors_a, descriptors_b, self.config.temporal_match_ratio)
        if len(matches) < 16:
            return None

        pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
        pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
        e, mask = cv2.findEssentialMat(pts_a, pts_b, self.k, method=cv2.RANSAC, threshold=1.0)
        if e is None or mask is None:
            return None

        _, r, t, _ = cv2.recoverPose(e, pts_a, pts_b, self.k, mask=mask)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = r
        pose[:3, 3] = t.ravel()
        return pose
