from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_expect(value: str) -> tuple[str, str, int, str, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 5:
        raise ValueError("--expect must be backbone,dataset,horizon,method,mse")
    backbone, dataset, horizon, method, mse = parts
    return backbone, dataset, int(horizon), method, float(mse)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CSV values against expected MSEs.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--expect", action="append", required=True, help="backbone,dataset,horizon,method,mse")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    df = pd.read_csv(csv_path)
    failures = []
    for item in args.expect:
        backbone, dataset, horizon, method, expected = parse_expect(item)
        row = df[
            (df["backbone"] == backbone)
            & (df["dataset"] == dataset)
            & (df["horizon"].astype(int) == horizon)
            & (df["method"] == method)
        ]
        if row.empty:
            failures.append(f"missing row: {item}")
            continue
        actual = float(row.iloc[0]["mse"])
        diff = abs(actual - expected)
        print(
            f"[check] {backbone} {dataset} H={horizon} {method}: "
            f"actual={actual:.6f} expected={expected:.6f} diff={diff:.6f}"
        )
        if diff > args.tolerance:
            failures.append(f"{item}: actual={actual:.6f}, diff={diff:.6f}")
    if failures:
        raise SystemExit("Verification failed:\n" + "\n".join(failures))
    print("[ok] all checks passed")


if __name__ == "__main__":
    main()
