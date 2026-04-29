import argparse
import time
from pathlib import Path

import pandas as pd

from experiments.common import (
    build_cfg,
    ensure_checkpoint,
    evaluate_zero,
    RESULTS_DIR,
)
from datasets.loader import get_test_dataloader
from models.forecast import forecast
from utils.misc import prepare_inputs
from gttd import evaluate_gttd


def evaluate_template(cfg, model) -> dict:
    return evaluate_gttd(cfg, model, get_test_dataloader, forecast, prepare_inputs)


def run_experiment(datasets: list[str], horizons: list[int], out_name: str) -> Path:
    csv_path = RESULTS_DIR / out_name
    if csv_path.exists():
        csv_path.unlink()
    rows = []
    for dataset in datasets:
        for horizon in horizons:
            cfg = build_cfg(dataset, horizon)
            print(f"[template-gttd] DLinear {dataset} H={horizon}")
            model = ensure_checkpoint(cfg)
            zero = evaluate_zero(cfg, model)
            start = time.perf_counter()
            ours = evaluate_template(cfg, model)
            adapt_ms = (time.perf_counter() - start) * 1000.0
            rows.extend(
                [
                    {"dataset": dataset, "backbone": "DLinear", "horizon": horizon, "method": "Zero-Shot", "mse": zero["test_mse"], "mae": zero["test_mae"], "adapt_latency_ms_total": 0.0},
                    {"dataset": dataset, "backbone": "DLinear", "horizon": horizon, "method": "Template-GTTD", "mse": ours["mse"], "mae": ours["mae"], "adapt_latency_ms_total": adapt_ms},
                ]
            )
            pd.DataFrame(rows[-2:]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            print(f"[template-gttd] {dataset} H={horizon} Zero={zero['test_mse']:.4f} Ours={ours['mse']:.4f}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["ETTh1", "ETTh2", "ETTm2", "exchange_rate", "weather"])
    parser.add_argument("--horizons", nargs="*", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--out", type=str, default="gttd_dlinear.csv")
    args = parser.parse_args()
    csv = run_experiment(args.datasets, args.horizons, args.out)
    print(csv)
