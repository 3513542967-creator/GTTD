from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED = [
    ROOT / "benchmarks" / "forecasting",
    ROOT / "benchmarks" / "forecasting" / "data",
    ROOT / "checkpoints" / "DLinear",
    ROOT / "checkpoints" / "PatchTST",
    ROOT / "experiments" / "dlinear" / "run_gttd.py",
    ROOT / "experiments" / "patchtst" / "run_gttd.py",
    ROOT / "experiments" / "ols" / "run_gttd.py",
    ROOT / "experiments" / "micn" / "run_gttd.py",
    ROOT / "results" / "tta" / "gttd_dlinear.csv",
    ROOT / "results" / "tta" / "gttd_patchtst.csv",
    ROOT / "results" / "tta" / "gttd_ols.csv",
    ROOT / "results" / "tta" / "gttd_micn.csv",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit("Missing required project assets:\n" + "\n".join(missing))
    print("[ok] required project assets are present")


if __name__ == "__main__":
    main()
