from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(backbone: str, datasets: list[str], horizons: list[int], out: str) -> Path:
    if backbone == "DLinear":
        from experiments.dlinear.run_gttd import run_experiment

        return run_experiment(datasets, horizons, out)
    if backbone == "PatchTST":
        from experiments.patchtst.run_gttd import run_experiment

        return run_experiment(datasets, horizons, out)
    if backbone == "OLS":
        from experiments.ols.run_gttd import run_experiment

        return run_experiment(datasets, horizons, out)
    if backbone == "MICN":
        from experiments.micn.run_gttd import run_experiment

        return run_experiment(datasets, horizons, out)
    raise ValueError(
        f"{backbone} is configured but not yet backed by retained checkpoints. "
        "Add checkpoints and a runner module before executing it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GTTD experiments under the retained benchmark protocol.")
    parser.add_argument("--backbone", choices=["DLinear", "PatchTST", "OLS", "MICN"], required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output CSV filename under results/tta/.")
    args = parser.parse_args()
    out_path = run(args.backbone, args.datasets, args.horizons, args.out)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
