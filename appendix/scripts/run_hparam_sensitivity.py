from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiments.common import build_cfg, ensure_checkpoint
from run_appendix_ablation import evaluate_variant


SWEEPS = {
    "rank": [1, 2, 4, 8],
    "template_decay": [0.80, 0.90, 0.92, 0.96],
    "global_mix": [0.25, 0.50, 0.75, 1.00],
    "local_alpha": [0.10, 0.50, 1.00, 2.00],
    "clip_value": [0.50, 1.00, 2.50, 5.00],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter sensitivity for GTTD.")
    parser.add_argument("--backbone", choices=["DLinear", "PatchTST", "OLS", "MICN"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--parameters", nargs="*", choices=list(SWEEPS.keys()), default=list(SWEEPS.keys()))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = build_cfg(args.dataset, args.horizon, args.backbone)
    model = ensure_checkpoint(cfg)
    rows = []
    for parameter in args.parameters:
        for value in SWEEPS[parameter]:
            result = evaluate_variant(cfg, model, "full", overrides={parameter: value})
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "backbone": args.backbone,
                    "dataset": args.dataset,
                    "horizon": args.horizon,
                    "mse": result["mse"],
                    "mae": result["mae"],
                    "latency_ms": result["latency_ms"],
                }
            )
            print(f"[{parameter}={value}] mse={result['mse']:.6f}")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

