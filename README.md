# VI-SLAM-KITTI

Complete, modular Visual(-Inertial style) SLAM pipeline for KITTI Odometry sequences **00-10**, implemented from scratch in Python using stereo vision, local mapping, and GTSAM pose-graph optimization.

> Dataset: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

## Features

- Stereo frontend with ORB + FLANN matching
- RANSAC essential matrix outlier rejection
- Frame-to-frame pose estimation via PnP + RANSAC
- Keyframe insertion using parallax + inlier count criteria
- Local 3D landmark map and sliding-window keyframes
- Landmark culling by observation count and age
- Pose graph backend with GTSAM factor graph + iSAM2
- Loop closure using descriptor similarity (cosine)
- ATE/RPE evaluation and markdown table reporting
- Real-time Open3D map + trajectory visualization
- Trajectory export as `.ply` and TUM `.txt`

## Project Structure

```text
SLAM/
├── configs/
│   ├── kitti.yaml
│   └── kitti_real_subset.yaml
├── slam_kitti/
│   ├── __init__.py
│   ├── frontend.py
│   ├── backend.py
│   ├── map.py
│   ├── dataset.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   └── run_slam.py
├── tests/
│   ├── test_dataset.py
│   └── test_evaluation.py
├── requirements.txt
└── README.md
```

## Architecture Diagram (ASCII)

```text
				 +----------------------------+
				 |   KITTI Stereo + GT Data   |
				 +--------------+-------------+
								|
								v
					  +---------+---------+
					  |   Dataset Loader  |
					  |  (images/calib)   |
					  +---------+---------+
								|
								v
					  +---------+---------+
					  |      Frontend     |
					  | ORB+FLANN Stereo  |
					  | E-Mat RANSAC      |
					  | PnP RANSAC        |
					  +----+---------+----+
						   |         |
						   |         v
						   |   Keyframe Decision
						   v
				  +--------+---------+
				  |     Local Map     |
				  | landmarks + SW KF |
				  +--------+---------+
						   |
						   v
				  +--------+---------+
				  |      Backend      |
				  | GTSAM FactorGraph |
				  | Between + Loop    |
				  | iSAM2 Incremental |
				  +---+-----------+---+
					  |           |
					  v           v
			  +-------+---+   +---+----------------+
			  | Evaluation|   | Visualization/Save |
			  | ATE / RPE |   | Open3D + Matplotlib|
			  +-----------+   +--------------------+
```

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## KITTI Data Layout

Expected layout (KITTI odometry):

```text
<kitti_root>/
├── sequences/
│   ├── 00/
│   │   ├── image_0/*.png
│   │   ├── image_1/*.png
│   │   └── calib.txt
│   └── ...
└── poses/
	├── 00.txt
	└── ...
```

Set `dataset.root` and `dataset.sequence` in `configs/kitti.yaml`.

## Run

```bash
PYTHONPATH=. python scripts/run_slam.py --config configs/kitti.yaml
```

Run on the prepared real subset config:

```bash
PYTHONPATH=. python scripts/run_slam.py --config configs/kitti_real_subset.yaml
```

Outputs are written to:

```text
outputs/seq_XX/
├── map_points.ply
├── trajectory_est_tum.txt
├── trajectory_gt_tum.txt   # if GT available
└── trajectory_topdown.png
```

## Evaluation Metrics

Computed metrics:

- **ATE**: Absolute Trajectory Error (position)
- **RPE-trans**: Relative pose translational error
- **RPE-rot**: Relative pose rotational error (radians)

The pipeline prints a markdown table with mean/median/RMSE.

### Real Run Results (sequence 00, first 50 frames)

| Sequence | ATE Mean (m) | ATE Median (m) | ATE RMSE (m) | RPE-t Mean (m) | RPE-t Median (m) | RPE-t RMSE (m) | RPE-r Mean (rad) | RPE-r Median (rad) | RPE-r RMSE (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 42.4685 | 41.2691 | 49.9650 | 1.8198 | 1.8457 | 1.8274 | 0.0070 | 0.0067 | 0.0075 |

Generated artifacts from this run:

- `outputs_real/seq_00/map_points.ply`
- `outputs_real/seq_00/trajectory_est_tum.txt`
- `outputs_real/seq_00/trajectory_gt_tum.txt`
- `outputs_real/seq_00/trajectory_topdown.png`

## Notes

- The frontend is stereo VO-based and does not depend on any external SLAM wrapper.
- Loop closure is intentionally lightweight (descriptor cosine similarity baseline).
- You can disable Open3D in headless environments by setting `runtime.visualize_3d: false`.
- In headless/dev-container environments, Open3D may be unavailable at runtime; the code falls back to a plain ASCII `.ply` writer.

## Result GIFs

Add generated GIFs (trajectory + map) under a folder such as `assets/` and embed them here:

```markdown
![Seq00 Map](assets/seq00_map.gif)
![Seq00 Trajectory](assets/seq00_traj.gif)
```