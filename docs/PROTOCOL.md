# Main Table Protocol

The retained experiments follow the long-term forecasting setup used for the
main comparison table.

- Input length: `W = 96`
- Label length: `48`
- Horizons: `L in {96, 192, 336}`
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `weather`
- Metrics: MSE and MAE over the forecast horizon
- Frozen backbone checkpoints are loaded from `checkpoints/<Backbone>/<Dataset>_<Horizon>/checkpoint_best.pth`

The retained GTTD implementation is the persistent-field/template variant in:

```text
gttd/
```

Backbone-specific runners live in:

```text
experiments/dlinear/
experiments/patchtst/
experiments/ols/
experiments/micn/
```

## Standard Commands

DLinear:

```bash
python scripts/run_gttd.py --backbone DLinear --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather --horizons 96 192 336 --out gttd_dlinear.csv
```

PatchTST:

```bash
python scripts/run_gttd.py --backbone PatchTST --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate --horizons 96 192 336 --out gttd_patchtst.csv
```

OLS:

```bash
python scripts/run_gttd.py --backbone OLS --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather --horizons 96 192 336 --out gttd_ols.csv
```

MICN:

```bash
python scripts/run_gttd.py --backbone MICN --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather --horizons 96 192 336 --out gttd_micn.csv
```
