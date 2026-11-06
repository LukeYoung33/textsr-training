"""Run the fixed V5 visual and stroke-metrics evaluation after training.

This is an explicit post-training stage.  It does not alter the shared training
loop, so existing V4 launches keep their current behavior.  A training launch
can append this command with ``&&`` after the process that writes
``net_g_250000.pth``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_lenovo_v5_best_checkpoint_visuals_20260912 import evaluate


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="BasicSR per-arm experiment directory containing models/.",
    )
    parser.add_argument("--stroke-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--iteration", type=int, default=250000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_post_training_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment Resolve the final checkpoint and run the immutable V5 evaluator."""
    experiment_root = args.experiment_root.expanduser().resolve()
    checkpoint = experiment_root / "models" / f"net_g_{int(args.iteration)}.pth"
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else experiment_root / f"post_evaluation_{int(args.iteration)}"
    )
    report = evaluate(
        args.arm,
        args.config,
        checkpoint,
        args.stroke_metrics,
        output_root,
        args.device,
    )
    marker = {
        "status": "COMPLETE",
        "arm": args.arm,
        "iteration": int(args.iteration),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "stroke_metrics_summary": str(output_root / "stroke_metrics_summary.json"),
        "visual_root": str(output_root / "visuals"),
    }
    (output_root / "post_training_evaluation.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """@luna-comment Execute one final V5 evaluation and print its output locations."""
    args = _args()
    report = run_post_training_evaluation(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": report["arm"],
                "iteration": int(args.iteration),
                "samples": report["samples"],
                "gt_groups": report["gt_groups"],
                "output_root": str(
                    args.output_root.expanduser().resolve()
                    if args.output_root is not None
                    else args.experiment_root.expanduser().resolve()
                    / f"post_evaluation_{int(args.iteration)}"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
