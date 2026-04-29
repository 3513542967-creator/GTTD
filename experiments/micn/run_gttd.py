from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from experiments.common import RESULTS_DIR, build_cfg, ensure_checkpoint, evaluate_zero
from experiments.dlinear.run_gttd import evaluate_template


def run_experiment(datasets: list[str], horizons: list[int], out_name: str) -> Path:
    csv_path = RESULTS_DIR / out_name
    if csv_path.exists():
        csv_path.unlink()
    for dataset in datasets:
        for horizon in horizons:
            cfg = build_cfg(dataset, horizon, backbone="MICN")
            print(f"[template-gttd] MICN {dataset} H={horizon}")
            model = ensure_checkpoint(cfg)
            zero = evaluate_zero(cfg, model)
            start = time.perf_counter()
            ours = evaluate_template(cfg, model)
            adapt_ms = (time.perf_counter() - start) * 1000.0
            rows = [
                {"dataset": dataset, "backbone": "MICN", "horizon": horizon, "method": "Zero-Shot", "mse": zero["test_mse"], "mae": zero["test_mae"], "adapt_latency_ms_total": 0.0},
                {"dataset": dataset, "backbone": "MICN", "horizon": horizon, "method": "Template-GTTD", "mse": ours["mse"], "mae": ours["mae"], "adapt_latency_ms_total": adapt_ms},
            ]
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            print(f"[template-gttd] MICN {dataset} H={horizon} Zero={zero['test_mse']:.4f} Ours={ours['mse']:.4f}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"])
    parser.add_argument("--horizons", nargs="*", type=int, default=[96, 192, 336])
    parser.add_argument("--out", type=str, default="gttd_micn.csv")
    args = parser.parse_args()
    print(run_experiment(args.datasets, args.horizons, args.out))
