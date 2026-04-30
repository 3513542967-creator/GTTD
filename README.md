# GTTD: Persistent Residual-Template Test-Time Adaptation

GTTD is a lightweight test-time adaptation method for long-term time-series
forecasting. It keeps the forecasting backbone frozen and corrects predictions
with online residual prefixes and a persistent residual template.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/check_project.py
```

Run one retained experiment:

```bash
python scripts/run_gttd.py --backbone DLinear --datasets ETTh1 --horizons 96 --out scratch_dlinear.csv
```

Outputs are written to `results/tta/`.

## Repository Map

```text
gttd/                     GTTD adapter and evaluation code
experiments/              Backbone-specific training/evaluation runners
scripts/                  Unified run and verification scripts
configs/                  Experiment configs
results/tta/              Retained CSV results
appendix/                 Appendix experiment scripts and LaTeX tables
benchmarks/forecasting/   Retained forecasting benchmark dependency
```

## Data and Backbones

Datasets used in the retained protocol:

```text
ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather
```

The ETT datasets come from [ETDataset](https://github.com/zhouhaoyi/ETDataset).
`exchange_rate` and `weather` follow the common long-term forecasting dataset
collection used by [Autoformer](https://github.com/thuml/Autoformer).

Supported forecasting backbones:

```text
DLinear, PatchTST, OLS, MICN
```

The retained backbone implementations live in `benchmarks/forecasting/models/`.
GTTD-specific code is separate under `gttd/`.

## Appendix Experiments

Appendix experiment plans, real CSV outputs, and generated LaTeX tables are in:

```text
appendix/
```

Useful commands:

```bash
python appendix/scripts/build_provenance_table.py
python appendix/scripts/build_latex_tables.py
```

## Notes

- Large checkpoint binaries are excluded from git. Place them under
  `checkpoints/<Backbone>/<Dataset>_<Horizon>/checkpoint_best.pth` when needed.
- `benchmarks/forecasting/` is derived from the PETSA/TAFAS forecasting
  benchmark code and keeps its original license.
- See `NOTICE.md`, `LICENSE`, and `docs/CODE_RELEASE.md` for attribution and
  release details.

