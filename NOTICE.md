# Notice and Attribution

This repository contains the code release for GTTD, a test-time adaptation
method for long-term time-series forecasting.

## Original project code

The GTTD method implementation and experiment glue live in:

```text
gttd/
experiments/
scripts/
configs/
docs/
```

These files implement the persistent residual-template adapter, evaluation
protocol, backbone-specific runners, retained configuration files, and result
table helpers.

## Retained benchmark/backbone dependency

The directory below is a retained benchmark dependency:

```text
benchmarks/forecasting/
```

It includes dataset loaders, trainer utilities, and backbone implementations
for DLinear, PatchTST, OLS, and MICN. The files in this directory are derived
from the PETSA/TAFAS forecasting benchmark implementation. The original license
text is preserved at:

```text
benchmarks/forecasting/LICENSE
```

Many files in that directory also keep their file-level attribution comments.

## Datasets

The retained protocol uses standard long-term forecasting datasets:

```text
ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather
```

Dataset files are included only to support reproducibility of the retained
benchmark protocol. Users should also follow the dataset providers' original
terms and citation requirements.

## Checkpoints

Model checkpoint files are large binary artifacts and are not intended to be
committed directly to git. Place them under:

```text
checkpoints/<Backbone>/<Dataset>_<Horizon>/checkpoint_best.pth
```

For public release, publish checkpoints through GitHub Releases, Git LFS, or an
external archival service, and document the download URL in docs/CHECKPOINTS.md.

## Result provenance

The CSV files in results/tta/ are the retained table artifacts. Their schema
includes source/provenance fields where available. Some table-construction
helpers may contain values imported from prior comparison tables or marked as
estimated placeholders; do not treat those as newly verified measurements unless
the corresponding source flag says they were locally verified.
