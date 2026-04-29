from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "tta"
OUT_DIR = ROOT / "appendix" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def action_for(row: pd.Series) -> str:
    if bool(row["verified_or_official"]):
        return "keep as verified retained result"
    return "rerun before using as evidence; otherwise mark as placeholder"


def main() -> None:
    frames = []
    for csv_path in sorted(RESULT_DIR.glob("gttd_*.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise SystemExit(f"No retained result CSV files found in {RESULT_DIR}")

    df = pd.concat(frames, ignore_index=True)
    keep_cols = ["backbone", "dataset", "horizon", "method", "mse", "source", "verified_or_official"]
    out_df = df[keep_cols].copy()
    out_df["action"] = out_df.apply(action_for, axis=1)
    out_df = out_df.sort_values(["verified_or_official", "backbone", "dataset", "horizon"], ascending=[True, True, True, True])

    out = OUT_DIR / "result_provenance.csv"
    out_df.to_csv(out, index=False)
    unverified = out_df[out_df["verified_or_official"] == False]
    print(f"[saved] {out}")
    print(f"[summary] total_rows={len(out_df)} unverified_rows={len(unverified)}")
    if not unverified.empty:
        print("[unverified]")
        for _, row in unverified.iterrows():
            print(f"- {row['backbone']} {row['dataset']} H={row['horizon']} source={row['source']}")


if __name__ == "__main__":
    main()

