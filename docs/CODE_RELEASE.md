# Code Release Statement

This repository is prepared as the public code artifact for the GTTD paper.

## Scope

The release contains:

- the standalone GTTD adapter implementation in `gttd/`;
- backbone-specific evaluation runners in `experiments/`;
- unified experiment scripts in `scripts/`;
- retained configuration files in `configs/`;
- retained table CSV files in `results/tta/`;
- a forecasting benchmark dependency in `benchmarks/forecasting/`.

## Task

The code evaluates test-time adaptation for long-term time-series forecasting.
Given an input window of length 96, frozen forecasting backbones predict future
horizons of 96, 192, or 336 steps. GTTD uses residuals that become observable
during test-time rollout to adjust subsequent predictions without updating the
frozen backbone weights.

## Backbones

The retained runners support:

```text
DLinear
PatchTST
OLS
MICN
```

## Third-party and derived code

`benchmarks/forecasting/` is derived from the PETSA/TAFAS forecasting benchmark
implementation and retains its original license. GTTD-specific method code is
kept separately under `gttd/`.

## Large artifacts

Checkpoint files are intentionally excluded from git by `.gitignore`. Store or
release checkpoint binaries separately, then place them at:

```text
checkpoints/<Backbone>/<Dataset>_<Horizon>/checkpoint_best.pth
```

## Reproducibility checks

Run:

```bash
python scripts/check_project.py
python scripts/build_gttd_table_csv.py
```

Run one retained experiment:

```bash
python scripts/run_gttd.py --backbone DLinear --datasets ETTh1 --horizons 96 --out scratch_dlinear.csv
```

Outputs are written under:

```text
results/tta/
```
