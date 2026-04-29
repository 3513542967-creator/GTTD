# Appendix Experiments

This folder collects experiment plans, table templates, and runnable helper
code for the appendix of the GTTD paper.

The current GTTD implementation is the persistent residual-template adapter in
`gttd/adapter.py`. Appendix experiments should therefore validate the actual
components used by that adapter:

- online residual prefix extraction;
- persistent residual template memory;
- affine template alignment;
- low-rank SVD template truncation;
- low-frequency residual removal;
- local prefix transfer;
- clipped bounded correction;
- long-horizon tail correction.

## Files

```text
EXPERIMENT_PLAN.md                  Recommended appendix experiments
tables/appendix_table_templates.csv Machine-readable table skeletons
scripts/build_appendix_tables.py    Builds table templates from the plan
scripts/build_provenance_table.py   Lists verified/unverified retained rows
scripts/make_appendix_commands.py   Writes recommended run commands
scripts/run_appendix_ablation.py    Runs focused ablation/sensitivity sweeps
```

## Recommended Order

1. Build the empty appendix table skeletons.
2. Build the provenance table and identify unverified result rows.
3. Run one small ablation setting to confirm the scripts and output schema.
4. Expand component ablations to two backbones and multiple datasets.
5. Run sparse-prefix and hyperparameter sweeps only after the main ablation
   table is stable.

```bash
python appendix/scripts/build_appendix_tables.py
python appendix/scripts/build_provenance_table.py
python appendix/scripts/make_appendix_commands.py
```

## Suggested First Runs

Use a small verified setting first:

```bash
python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/ablation_dlinear_etth1_96.csv
python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/ablation_patchtst_etth1_96.csv
```

Then expand to the full appendix grid only after the table format is stable.

For the paper appendix, prioritize concise tables that visually validate model
properties: component necessity, robustness to sparse prefixes, hyperparameter
stability, lightweight latency, and long-horizon behavior.

Files named `smoke_*.csv` under `appendix/results/` are sanity-check outputs for
the scripts. They are not intended to be used as final appendix tables without
running the full planned grids.
