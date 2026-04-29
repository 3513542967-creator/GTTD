# Recommended Appendix Commands

Run these commands from the repository root. Start with the smoke runs,
then expand only after output schemas are confirmed.

## Smoke Runs

```bash
python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/ablation_dlinear_etth1_96.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/ablation_patchtst_etth1_96.csv
```

## Component Ablation Runs

```bash
python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/ablation_dlinear_etth1_96.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTm2 --horizon 96 --out appendix/results/ablation_dlinear_ettm2_96.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/ablation_patchtst_etth1_96.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone OLS --dataset ETTh1 --horizon 96 --out appendix/results/ablation_ols_etth1_96.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone MICN --dataset ETTh1 --horizon 96 --out appendix/results/ablation_micn_etth1_96.csv
```

## Sparse Prefix Sensitivity Runs

```bash
python appendix/scripts/run_prefix_sensitivity.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/prefix_dlinear_etth1_96.csv
```

```bash
python appendix/scripts/run_prefix_sensitivity.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/prefix_patchtst_etth1_96.csv
```

## Hyperparameter Sensitivity Runs

```bash
python appendix/scripts/run_hparam_sensitivity.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/hparam_dlinear_etth1_96.csv
```

```bash
python appendix/scripts/run_hparam_sensitivity.py --backbone PatchTST --dataset ETTh1 --horizon 96 --parameters rank template_decay global_mix --out appendix/results/hparam_patchtst_etth1_96.csv
```

## Long-Horizon Tail Runs

```bash
python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 192 --variants full no_tail --out appendix/results/tail_dlinear_etth1_192.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 336 --variants full no_tail --out appendix/results/tail_dlinear_etth1_336.csv
```

```bash
python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 336 --variants full no_tail --out appendix/results/tail_patchtst_etth1_336.csv
```

## Verification Gap Runs

```bash
python scripts/run_gttd.py --backbone PatchTST --datasets weather --horizons 96 192 336 --out verify_patchtst_weather.csv
```

```bash
python scripts/run_gttd.py --backbone MICN --datasets ETTh1 --horizons 96 --out verify_micn_etth1_96.csv
```

```bash
python scripts/run_gttd.py --backbone OLS --datasets ETTh1 --horizons 96 --out verify_ols_etth1_96.csv
```
