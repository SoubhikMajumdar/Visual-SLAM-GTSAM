"""Batch evaluation runner for KITTI sequences 00-10."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from slam_kitti.evaluation import ErrorStats, TrajectoryEvaluator
from scripts.run_slam import run_pipeline


def _parse_sequences(value: str) -> list[str]:
    """Parse comma-separated sequence ids/ranges into zero-padded ids."""
    items = [part.strip() for part in value.split(",") if part.strip()]
    sequences: list[str] = []
    for item in items:
        if "-" in item:
            start, end = item.split("-", maxsplit=1)
            for seq in range(int(start), int(end) + 1):
                sequences.append(f"{seq:02d}")
        else:
            sequences.append(f"{int(item):02d}")
    return sorted(set(sequences))


def _write_csv(path: Path, results: dict[str, dict[str, ErrorStats]]) -> None:
    """Write consolidated metrics to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence",
                "ate_mean",
                "ate_median",
                "ate_rmse",
                "rpe_t_mean",
                "rpe_t_median",
                "rpe_t_rmse",
                "rpe_r_mean",
                "rpe_r_median",
                "rpe_r_rmse",
            ]
        )
        for seq, metric in sorted(results.items()):
            writer.writerow(
                [
                    seq,
                    metric["ate"].mean,
                    metric["ate"].median,
                    metric["ate"].rmse,
                    metric["rpe_trans"].mean,
                    metric["rpe_trans"].median,
                    metric["rpe_trans"].rmse,
                    metric["rpe_rot"].mean,
                    metric["rpe_rot"].median,
                    metric["rpe_rot"].rmse,
                ]
            )


def main() -> None:
    """Run batch KITTI sequence evaluation and save summary artifacts."""
    parser = argparse.ArgumentParser(description="Run batch VI-SLAM evaluation for KITTI sequences")
    parser.add_argument("--config", type=str, default="configs/kitti.yaml", help="Base config path")
    parser.add_argument(
        "--sequences",
        type=str,
        default="00-10",
        help="Comma-separated list/ranges, e.g. 00-02,05,07",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional per-sequence frame limit (-1 for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_batch",
        help="Output root folder for per-sequence results and summaries",
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Enable Open3D visualization during batch run",
    )
    args = parser.parse_args()

    sequences = _parse_sequences(args.sequences)
    all_results: dict[str, dict[str, ErrorStats]] = {}
    failures: list[tuple[str, str]] = []

    max_frames_override = None if args.max_frames < 0 else args.max_frames
    for sequence in sequences:
        print(f"\n=== Running sequence {sequence} ===")
        try:
            _, metrics, _ = run_pipeline(
                config_path=args.config,
                sequence_override=sequence,
                max_frames_override=max_frames_override,
                visualize_override=args.visualize_3d,
                output_dir_override=args.output_dir,
            )
            if metrics is not None:
                all_results[sequence] = metrics
            else:
                failures.append((sequence, "No ground truth metrics available"))
        except Exception as exc:
            failures.append((sequence, str(exc)))
            print(f"[WARN] Sequence {sequence} failed: {exc}")

    summary_root = Path(args.output_dir)
    summary_root.mkdir(parents=True, exist_ok=True)

    if all_results:
        table = TrajectoryEvaluator.format_metrics_table(all_results)
        print("\nBatch metrics table:\n")
        print(table)
        (summary_root / "batch_metrics.md").write_text(table + "\n", encoding="utf-8")
        _write_csv(summary_root / "batch_metrics.csv", all_results)
        print(f"\nSaved: {summary_root / 'batch_metrics.md'}")
        print(f"Saved: {summary_root / 'batch_metrics.csv'}")

    if failures:
        lines = [f"{sequence}: {reason}" for sequence, reason in failures]
        failure_log = "\n".join(lines) + "\n"
        (summary_root / "batch_failures.log").write_text(failure_log, encoding="utf-8")
        print("\nFailed sequences:")
        for line in lines:
            print(f"- {line}")


if __name__ == "__main__":
    main()
