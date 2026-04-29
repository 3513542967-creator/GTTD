# GTTD: Persistent-Field Test-Time Adaptation

This is the cleaned code release for the GTTD paper experiments. The project is
GTTD; `benchmarks/forecasting` is only the benchmark/backbone dependency.

GTTD targets long-term time-series forecasting under test-time adaptation. Frozen
forecasting backbones first predict a future horizon; GTTD then uses residuals
that become observable during the test-time rollout to adjust subsequent
predictions without updating the backbone weights.

## Layout

```text
configs/                  Experiment configs
docs/                     Protocol and result notes
gttd/                     Standalone GTTD method implementation
experiments/dlinear/      DLinear training and GTTD experiment
experiments/patchtst/     PatchTST training and GTTD experiment
experiments/ols/          OLS training and GTTD experiment
experiments/micn/         MICN training and GTTD experiment
scripts/                  Run and verify commands
benchmarks/forecasting/   Benchmark/backbone code
checkpoints/              Best model checkpoints
results/tta/              CSV results
```

## Quick Checks

```bash
pip install -r requirements.txt
python scripts/check_project.py
python scripts/build_gttd_table_csv.py
```

## Run One Experiment

```bash
python scripts/run_gttd.py --backbone DLinear --datasets ETTh1 --horizons 96 --out scratch_dlinear.csv
```

The output is written under `results/tta/`. Use a scratch filename for ad-hoc
runs so the retained table CSV files stay unchanged.

## Where To Look

```text
gttd/adapter.py                  GTTD method implementation
experiments/<backbone>/run_gttd.py  Backbone-specific GTTD runner
scripts/run_gttd.py              Unified experiment launcher
scripts/build_gttd_table_csv.py  Rebuild retained table CSV files
results/tta/                     Paper-table GTTD CSV files
```

## Code Release Notes

- GTTD-specific code lives in `gttd/`, `experiments/`, `scripts/`, and
  `configs/`.
- `benchmarks/forecasting/` is a retained forecasting benchmark dependency
  derived from the PETSA/TAFAS benchmark code and keeps its original license.
- Large checkpoint binaries are excluded from git. Put them under
  `checkpoints/<Backbone>/<Dataset>_<Horizon>/checkpoint_best.pth` or publish
  them separately through GitHub Releases, Git LFS, or an archival service.
- See `NOTICE.md` and `docs/CODE_RELEASE.md` for attribution and release-scope
  details.

## Table CSV Files

```text
results/tta/gttd_dlinear.csv
results/tta/gttd_patchtst.csv
results/tta/gttd_ols.csv
results/tta/gttd_micn.csv
```

Each CSV contains only the GTTD rows for one backbone. CSV files are the source
of truth. LaTeX tables are not retained.

## Citation

If you use this code, please cite the associated GTTD paper. Before public
release, replace the placeholder metadata in `CITATION.cff` with the final
paper title, author list, repository URL, and DOI/arXiv identifier if available.
