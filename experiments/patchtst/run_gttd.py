import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

from experiments.common import (
    BENCHMARK_ROOT,
    CHECKPOINT_ROOT,
    RESULTS_DIR,
    ensure_checkpoint,
    evaluate_zero,
    get_cfg_defaults,
    set_devices,
    set_seeds,
    update_cfg_from_dataset,
)
from experiments.dlinear.run_gttd import evaluate_template


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"]
HORIZONS = [96, 192, 336]
SMOKE_ROOT = RESULTS_DIR / "smoke_data"


def prepare_smoke_dataset(dataset: str, rows: int) -> Path:
    src = BENCHMARK_ROOT / "data" / dataset / f"{dataset}.csv"
    dst_dir = SMOKE_ROOT / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{dataset}.csv"
    df = pd.read_csv(src).iloc[:rows].copy()
    df.to_csv(dst, index=False)
    return SMOKE_ROOT


def build_cfg(
    dataset: str,
    horizon: int,
    *,
    base_dir: Path | None = None,
    checkpoint_tag: str | None = None,
    max_epoch: int | None = None,
):
    cfg = get_cfg_defaults()
    update_cfg_from_dataset(cfg, dataset)
    cfg.SEED = 0
    cfg.VISIBLE_DEVICES = 0
    cfg.DATA.BASE_DIR = str(base_dir or (BENCHMARK_ROOT / "data"))
    cfg.DATA.SEQ_LEN = 96
    cfg.DATA.LABEL_LEN = 48
    cfg.DATA.PRED_LEN = horizon
    cfg.MODEL.NAME = "PatchTST"
    cfg.MODEL.seq_len = 96
    cfg.MODEL.label_len = 48
    cfg.MODEL.pred_len = horizon
    cfg.TRAIN.ENABLE = True
    cfg.TEST.ENABLE = True
    cfg.TTA.ENABLE = False
    cfg.DATA_LOADER.NUM_WORKERS = 0
    ckpt_name = f"{dataset}_{horizon}" if checkpoint_tag is None else f"{dataset}_{horizon}_{checkpoint_tag}"
    cfg.TRAIN.CHECKPOINT_DIR = str(CHECKPOINT_ROOT / "PatchTST" / ckpt_name)
    cfg.RESULT_DIR = str(RESULTS_DIR / "official_petsa_patchtst" / ckpt_name)
    if max_epoch is not None:
        cfg.SOLVER.MAX_EPOCH = max_epoch
    return cfg


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def run_experiment(
    datasets: list[str],
    horizons: list[int],
    out_name: str,
    *,
    base_dir: Path | None = None,
    checkpoint_tag: str | None = None,
    max_epoch: int | None = None,
) -> Path:
    csv_path = RESULTS_DIR / out_name
    if csv_path.exists():
        csv_path.unlink()

    rows = []
    for dataset in datasets:
        for horizon in horizons:
            cfg = build_cfg(
                dataset,
                horizon,
                base_dir=base_dir,
                checkpoint_tag=checkpoint_tag,
                max_epoch=max_epoch,
            )
            print(f"[template-gttd] PatchTST {dataset} H={horizon}")
            set_devices(cfg.VISIBLE_DEVICES)
            set_seeds(cfg.SEED)
            model = ensure_checkpoint(cfg)
            zero = evaluate_zero(cfg, model)
            start = time.perf_counter()
            ours = evaluate_template(cfg, model)
            adapt_ms = (time.perf_counter() - start) * 1000.0
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "backbone": "PatchTST",
                        "horizon": horizon,
                        "method": "Zero-Shot",
                        "mse": zero["test_mse"],
                        "mae": zero["test_mae"],
                        "adapt_latency_ms_total": 0.0,
                    },
                    {
                        "dataset": dataset,
                        "backbone": "PatchTST",
                        "horizon": horizon,
                        "method": "Template-GTTD",
                        "mse": ours["mse"],
                        "mae": ours["mae"],
                        "adapt_latency_ms_total": adapt_ms,
                    },
                ]
            )
            append_rows(csv_path, rows[-2:])
            print(
                f"[template-gttd] PatchTST {dataset} H={horizon} "
                f"Zero={zero['test_mse']:.4f} Ours={ours['mse']:.4f}"
            )
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    parser.add_argument("--out", type=str, default="gttd_patchtst.csv")
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-tag", type=str, default=None)
    args = parser.parse_args()
    base_dir = None
    tag = args.checkpoint_tag
    if args.smoke_rows > 0:
        if len(args.datasets) != 1:
            raise ValueError("Smoke mode currently supports one dataset at a time.")
        base_dir = prepare_smoke_dataset(args.datasets[0], args.smoke_rows)
        tag = tag or f"smoke{args.smoke_rows}"
    csv = run_experiment(
        args.datasets,
        args.horizons,
        args.out,
        base_dir=base_dir,
        checkpoint_tag=tag,
        max_epoch=args.max_epoch,
    )
    print(csv)
