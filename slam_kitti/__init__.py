"""VI-SLAM-KITTI package."""

from .dataset import KITTIOdometryDataset, StereoFrame
from .evaluation import TrajectoryEvaluator
from .frontend import FrontendConfig, TrackingResult, VisualFrontend
from .map import Keyframe, Landmark, LocalMap

try:
    from .backend import LoopClosureDetector, SLAMBackend
except Exception:
    LoopClosureDetector = None
    SLAMBackend = None

__all__ = [
    "KITTIOdometryDataset",
    "StereoFrame",
    "FrontendConfig",
    "TrackingResult",
    "VisualFrontend",
    "Landmark",
    "Keyframe",
    "LocalMap",
    "TrajectoryEvaluator",
]

if LoopClosureDetector is not None and SLAMBackend is not None:
    __all__.extend(["LoopClosureDetector", "SLAMBackend"])
