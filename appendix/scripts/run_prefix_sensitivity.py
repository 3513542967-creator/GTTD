from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiments.common import build_cfg, ensure_checkpoint, evaluate_zero
from run_appendix_ablation import evaluate_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse revealed-prefix sensitivity for GTTD.")
    parser.add_argument("--backbone", choices=["DLinear", "PatchTST", "OLS", "MICN"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 3, 6, 12, 24])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = build_cfg(args.dataset, args.horizon, args.backbone)
    model = ensure_checkpoint(cfg)
    zero = evaluate_zero(cfg, model)
    rows = []
    for k in args.k_values:
        result = evaluate_variant(cfg, model, "full", overrides={"max_prefix_k": k})
        improvement = (zero["test_mse"] - result["mse"]) / zero["test_mse"] * 100.0
        rows.append(
            {
                "backbone": args.backbone,
                "dataset": args.dataset,
                "horizon": args.horizon,
                "k": k,
                "mse": result["mse"],
                "mae": result["mae"],
                "improvement_vs_zero_shot_pct": improvement,
                "latency_ms": result["latency_ms"],
            }
        )
        print(f"[k={k}] mse={result['mse']:.6f} improvement={improvement:.2f}%")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

