# Results

The retained result files contain only the GTTD rows used in the paper table.
They are split by backbone for quick lookup:

```text
results/tta/gttd_dlinear.csv
results/tta/gttd_patchtst.csv
results/tta/gttd_ols.csv
results/tta/gttd_micn.csv
```

Each file uses the same schema:

```text
dataset,backbone,horizon,method,mse,avg_imp_pct,source,verified_or_official
```

`avg_imp_pct` is the average relative MSE improvement over the corresponding
backbone baseline across horizons 96, 192, and 336.

Use `source` and `verified_or_official` to distinguish locally verified GTTD
values from estimated placeholders retained for table drafting.
