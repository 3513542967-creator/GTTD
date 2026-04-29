# Forecasting Benchmark Dependency

This directory contains the retained forecasting benchmark components used by
the GTTD experiments:

- datasets and loaders
- DLinear, PatchTST, OLS, and MICN backbones
- shared trainer, optimizer, and utility code

The project method lives in `gttd/`. The reproducible experiment entry points
live in `experiments/` and `scripts/`.

The code in this directory is derived from the PETSA/TAFAS forecasting
benchmark implementation and keeps the original license file.
