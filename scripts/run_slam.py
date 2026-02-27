"""Main entry point for VI-SLAM-KITTI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from slam_kitti.backend import BackendConfig, LoopClosureDetector, SLAMBackend
from slam_kitti.dataset import KITTIOdometryDataset
from slam_kitti.evaluation import ErrorStats, TrajectoryEvaluator
from slam_kitti.frontend import FrontendConfig, TrackingResult, VisualFrontend
from slam_kitti.map import Keyframe, LocalMap
from slam_kitti.visualization import Open3DMapViewer, TrajectoryPlotter, save_point_cloud_ply, save_tum_trajectory


def _load_config(config_path: str | Path) -> dict:
    """Load YAML config into a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _invert_se3(transform: np.ndarray) -> np.ndarray:
    """Invert 4x4 rigid transform."""
    r = transform[:3, :3]
    t = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    return inv


def _make_keyframe(
    keyframe_id: int,
    frame_index: int,
    timestamp: float,
    pose: np.ndarray,
    image: np.ndarray,
    keypoints: list,
    descriptors: np.ndarray | None,
) -> Keyframe:
    """Factory for keyframe object."""
    return Keyframe(
        keyframe_id=keyframe_id,
        frame_index=frame_index,
        timestamp=timestamp,
        pose=pose.copy(),
        image=image.copy(),
        keypoints=keypoints,
        descriptors=descriptors.copy() if descriptors is not None else None,
    )


def run_pipeline(
    config_path: str | Path,
    sequence_override: str | None = None,
    max_frames_override: int | None = None,
    visualize_override: bool | None = None,
    output_dir_override: str | None = None,
) -> tuple[str, dict[str, ErrorStats] | None, Path]:
    """Run the complete VI-SLAM-KITTI pipeline."""
    config = _load_config(config_path)

    dataset_cfg = config["dataset"]
    frontend_cfg = FrontendConfig(**config["frontend"])
    map_cfg = config["map"]
    backend_cfg = BackendConfig(
        prior_noise_sigmas=tuple(config["backend"]["prior_noise_sigmas"]),
        odom_noise_sigmas=tuple(config["backend"]["odom_noise_sigmas"]),
        loop_noise_sigmas=tuple(config["backend"]["loop_noise_sigmas"]),
        loop_similarity_threshold=float(config["backend"]["loop_similarity_threshold"]),
        loop_min_separation=int(config["backend"]["loop_min_separation"]),
    )
    runtime_cfg = config["runtime"]

    if sequence_override is not None:
        dataset_cfg["sequence"] = sequence_override
    if max_frames_override is not None:
        runtime_cfg["max_frames"] = int(max_frames_override)
    if visualize_override is not None:
        runtime_cfg["visualize_3d"] = bool(visualize_override)
    if output_dir_override is not None:
        runtime_cfg["output_dir"] = output_dir_override

    dataset = KITTIOdometryDataset(dataset_cfg["root"], dataset_cfg["sequence"])
    first_frame = dataset.get_frame(0)

    frontend = VisualFrontend(first_frame.intrinsics, first_frame.baseline, frontend_cfg)
    local_map = LocalMap(window_size=int(map_cfg["sliding_window_size"]))
    backend = SLAMBackend(backend_cfg)
    loop_detector = LoopClosureDetector(
        similarity_threshold=backend_cfg.loop_similarity_threshold,
        min_separation=backend_cfg.loop_min_separation,
    )

    output_dir = Path(runtime_cfg["output_dir"]) / f"seq_{dataset.sequence}"
    output_dir.mkdir(parents=True, exist_ok=True)

    viewer = Open3DMapViewer(enabled=bool(runtime_cfg["visualize_3d"]))

    keypoints0, descriptors0, landmarks0 = frontend.bootstrap_stereo(first_frame.left, first_frame.right)
    global_pose = np.eye(4, dtype=np.float64)

    keyframe0 = _make_keyframe(
        keyframe_id=0,
        frame_index=0,
        timestamp=first_frame.timestamp,
        pose=global_pose,
        image=first_frame.left,
        keypoints=keypoints0,
        descriptors=descriptors0,
    )
    local_map.add_keyframe(keyframe0)
    local_map.create_landmarks_from_stereo(keyframe0, landmarks0)

    backend.add_prior(0, global_pose)
    backend.add_initial_estimate(0, global_pose)
    backend.update()

    estimated_poses: list[np.ndarray] = [global_pose.copy()]
    gt_poses: list[np.ndarray] = [first_frame.gt_pose.copy()] if first_frame.gt_pose is not None else []

    descriptor_history: dict[int, np.ndarray | None] = {0: descriptors0}
    keyframes_by_id: dict[int, Keyframe] = {0: keyframe0}

    last_keyframe = keyframe0
    last_keyframe_landmarks = landmarks0
    keyframe_counter = 1

    max_frames = int(runtime_cfg["max_frames"])
    frame_limit = len(dataset) if max_frames <= 0 else min(max_frames, len(dataset))

    for frame_idx in tqdm(range(1, frame_limit), desc=f"Sequence {dataset.sequence}"):
        frame = dataset.get_frame(frame_idx)
        tracking: TrackingResult = frontend.track_frame(
            prev_image=last_keyframe.image,
            curr_image=frame.left,
            prev_keypoints=last_keyframe.keypoints,
            prev_descriptors=last_keyframe.descriptors,
            prev_landmarks=last_keyframe_landmarks,
        )

        if tracking.success:
            global_pose = last_keyframe.pose @ tracking.relative_pose
        estimated_poses.append(global_pose.copy())
        if frame.gt_pose is not None:
            gt_poses.append(frame.gt_pose.copy())

        insert_keyframe = tracking.success and frontend.should_add_keyframe(tracking)
        if not insert_keyframe:
            viewer.update(local_map.get_landmark_points(), estimated_poses)
            continue

        curr_kp, curr_desc, curr_landmarks = frontend.bootstrap_stereo(frame.left, frame.right)
        current_kf = _make_keyframe(
            keyframe_id=keyframe_counter,
            frame_index=frame.index,
            timestamp=frame.timestamp,
            pose=global_pose,
            image=frame.left,
            keypoints=curr_kp,
            descriptors=curr_desc,
        )
        local_map.add_keyframe(current_kf)
        local_map.create_landmarks_from_stereo(current_kf, curr_landmarks)

        rel = _invert_se3(last_keyframe.pose) @ current_kf.pose
        backend.add_odometry_factor(last_keyframe.keyframe_id, current_kf.keyframe_id, rel)
        backend.add_initial_estimate(current_kf.keyframe_id, current_kf.pose)

        candidate_id = loop_detector.detect(current_kf.keyframe_id, curr_desc, descriptor_history)
        if candidate_id is not None and candidate_id in keyframes_by_id:
            candidate_kf = keyframes_by_id[candidate_id]
            loop_rel = frontend.estimate_relative_pose_between_keyframes(
                candidate_kf.keypoints,
                candidate_kf.descriptors,
                current_kf.keypoints,
                current_kf.descriptors,
            )
            if loop_rel is not None:
                backend.add_loop_factor(candidate_id, current_kf.keyframe_id, loop_rel)

        if current_kf.keyframe_id == 1:
            backend.add_initial_estimate(last_keyframe.keyframe_id, last_keyframe.pose)

        backend.update()

        for keyframe in local_map.get_active_keyframes():
            optimized = backend.get_pose(keyframe.keyframe_id)
            if optimized is not None:
                keyframe.pose = optimized

        local_map.cull_landmarks(
            min_observations=int(map_cfg["min_landmark_observations"]),
            current_keyframe=current_kf.keyframe_id,
            max_age=int(map_cfg["max_landmark_age"]),
        )

        keyframes_by_id[current_kf.keyframe_id] = current_kf
        descriptor_history[current_kf.keyframe_id] = current_kf.descriptors
        last_keyframe = current_kf
        last_keyframe_landmarks = curr_landmarks
        keyframe_counter += 1

        viewer.update(local_map.get_landmark_points(), estimated_poses)

    viewer.close()

    results = {}
    if gt_poses:
        metrics = TrajectoryEvaluator.evaluate(gt_poses, estimated_poses[: len(gt_poses)], delta=1)
        results[dataset.sequence] = metrics
        print(TrajectoryEvaluator.format_metrics_table(results))

    if bool(runtime_cfg["save_outputs"]):
        save_tum_trajectory(estimated_poses, output_dir / "trajectory_est_tum.txt")
        if gt_poses:
            save_tum_trajectory(gt_poses, output_dir / "trajectory_gt_tum.txt")
        save_point_cloud_ply(local_map.get_landmark_points(), output_dir / "map_points.ply")
        TrajectoryPlotter.plot_topdown(estimated_poses, gt_poses, output_dir / "trajectory_topdown.png")

    print(f"Outputs saved to: {output_dir}")
    return dataset.sequence, results.get(dataset.sequence), output_dir


def main() -> None:
    """Parse args and run pipeline."""
    parser = argparse.ArgumentParser(description="Run VI-SLAM pipeline on KITTI odometry")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
