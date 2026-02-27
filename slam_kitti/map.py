"""Map and keyframe management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Landmark:
    """3D landmark tracked in the local map."""

    landmark_id: int
    position: np.ndarray
    descriptor: np.ndarray | None
    observations: set[tuple[int, int]] = field(default_factory=set)
    last_seen_keyframe: int = 0


@dataclass(slots=True)
class Keyframe:
    """Keyframe state used for backend optimization and mapping."""

    keyframe_id: int
    frame_index: int
    timestamp: float
    pose: np.ndarray
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    landmark_by_feature: dict[int, int] = field(default_factory=dict)


class LocalMap:
    """Local map with sliding-window keyframes and landmark bookkeeping."""

    def __init__(self, window_size: int = 10) -> None:
        """Create local map containers."""
        self.window_size = window_size
        self.keyframes: deque[Keyframe] = deque(maxlen=window_size)
        self.landmarks: dict[int, Landmark] = {}
        self._next_landmark_id = 0

    def add_keyframe(self, keyframe: Keyframe) -> None:
        """Insert keyframe into sliding window."""
        self.keyframes.append(keyframe)

    def add_landmark(
        self,
        position: np.ndarray,
        descriptor: np.ndarray | None,
        keyframe_id: int,
        feature_idx: int,
    ) -> int:
        """Create a landmark and register first observation."""
        landmark_id = self._next_landmark_id
        self._next_landmark_id += 1

        landmark = Landmark(
            landmark_id=landmark_id,
            position=position.astype(np.float64),
            descriptor=descriptor.copy() if descriptor is not None else None,
            observations={(keyframe_id, feature_idx)},
            last_seen_keyframe=keyframe_id,
        )
        self.landmarks[landmark_id] = landmark
        return landmark_id

    def add_observation(self, landmark_id: int, keyframe_id: int, feature_idx: int) -> None:
        """Attach another keyframe-feature observation to a landmark."""
        if landmark_id not in self.landmarks:
            return
        landmark = self.landmarks[landmark_id]
        landmark.observations.add((keyframe_id, feature_idx))
        landmark.last_seen_keyframe = max(landmark.last_seen_keyframe, keyframe_id)

    def create_landmarks_from_stereo(
        self,
        keyframe: Keyframe,
        triangulated: dict[int, np.ndarray],
    ) -> None:
        """Add newly triangulated keyframe landmarks into map."""
        if keyframe.descriptors is None:
            return

        for feat_idx, point in triangulated.items():
            if feat_idx >= len(keyframe.descriptors):
                continue
            descriptor = keyframe.descriptors[feat_idx]
            landmark_id = self.add_landmark(point, descriptor, keyframe.keyframe_id, feat_idx)
            keyframe.landmark_by_feature[feat_idx] = landmark_id

    def cull_landmarks(self, min_observations: int, current_keyframe: int, max_age: int) -> int:
        """Remove weak/stale landmarks and return number removed."""
        to_remove: list[int] = []
        for landmark_id, landmark in self.landmarks.items():
            is_weak = len(landmark.observations) < min_observations
            is_stale = (current_keyframe - landmark.last_seen_keyframe) > max_age
            if is_weak or is_stale:
                to_remove.append(landmark_id)

        for landmark_id in to_remove:
            self.landmarks.pop(landmark_id, None)

        for keyframe in self.keyframes:
            keyframe.landmark_by_feature = {
                feature_idx: lm_id
                for feature_idx, lm_id in keyframe.landmark_by_feature.items()
                if lm_id in self.landmarks
            }

        return len(to_remove)

    def get_active_keyframes(self) -> list[Keyframe]:
        """Return active keyframes from oldest to newest."""
        return list(self.keyframes)

    def get_landmark_points(self) -> np.ndarray:
        """Return all current landmark coordinates."""
        if not self.landmarks:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([lm.position for lm in self.landmarks.values()])
