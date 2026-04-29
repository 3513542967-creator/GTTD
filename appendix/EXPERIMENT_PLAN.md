# Appendix Experiment Plan

The main paper should keep Table 1 and Table 2 focused. The appendix can carry
the diagnostic evidence that the current GTTD design works for the reasons
claimed in the revised methodology.

## A1. Component Ablation

Goal: show which parts of the persistent residual-template adapter matter.

Recommended rows:

| Variant | What it removes or changes | Expected interpretation |
| --- | --- | --- |
| Full GTTD | Current adapter | Reference |
| w/o persistent template | disables `template_state` global correction | Tests whether cross-batch residual memory helps |
| w/o affine alignment | uses raw template without scale/bias fitting | Tests adaptation to current residual magnitude |
| w/o low-rank truncation | keeps full residual template | Tests whether SVD suppresses noisy residual fields |
| w/o low-frequency removal | propagates raw local residual prefix | Tests separation of fast residual from trend |
| w/o local propagation | disables prefix-transfer correction | Tests short-term residual propagation |
| w/o clipping | removes bounded correction | Tests safety of correction magnitude |
| fixed period | replaces FFT period estimate with a constant | Tests the period-aware batching design |

Recommended table columns:

```text
backbone, dataset, horizon, variant, mse, mae, delta_mse_vs_full_pct, latency_ms
```

## A2. Sparse Prefix Sensitivity

Goal: support the test-time adaptation setting under very limited revealed
future observations.

Sweep the maximum usable prefix length:

```text
k in {1, 2, 3, 6, 12, 24, FFT-period}
```

Recommended table columns:

```text
backbone, dataset, horizon, k, mse, mae, improvement_vs_zero_shot_pct
```

This is especially important because the old manuscript discusses few-shot TTA,
while the current implementation uses period-derived online residual prefixes.

## A3. Hyperparameter Sensitivity

Goal: show the method is not a fragile tuned trick.

Sweep:

```text
rank:           {1, 2, 4, 8}
template_decay: {0.80, 0.90, 0.92, 0.96}
global_mix:     {0.25, 0.50, 0.75, 1.00}
local_alpha:    {0.10, 0.50, 1.00, 2.00}
clip_value:     {0.50, 1.00, 2.50, 5.00}
```

Recommended table columns:

```text
parameter, value, backbone, dataset, horizon, mse, mae
```

## A4. Provenance and Verification Completion

Goal: separate locally verified values from placeholders.

Current retained CSVs include provenance flags. The appendix should explicitly
list any rows that are not locally verified, then either remove them from claims
or run them before camera-ready.

Known priority:

- verify PatchTST on Weather for horizons 96/192/336;
- complete any OLS/MICN dataset-horizon rows that are currently marked as
  previous-run or estimated placeholder;
- avoid using placeholder rows as evidence for a main claim.

Recommended table columns:

```text
backbone, dataset, horizon, source, verified_or_official, action
```

## A5. Long-Horizon Tail Behavior

Goal: justify the `far_tail_mix` branch for horizons beyond 192.

Run:

```text
horizon in {96, 192, 336, 720}
```

Compare:

```text
Full GTTD vs w/o tail correction
```

This should be placed in the appendix unless 720-step forecasting becomes a
central paper claim.

## A6. Latency and Memory

Goal: support the "lightweight / no backbone update" claim without using the
old "exact O(1) harmonic solver" language.

Measure:

- zero-shot inference time;
- GTTD adapter time;
- total evaluation time;
- peak GPU memory if CUDA is available;
- CPU adapter time for the NumPy/SciPy correction path.

Recommended table columns:

```text
backbone, dataset, horizon, zero_shot_ms, adapter_ms, total_ms, peak_gpu_mb
```

## A7. Stress Tests

Goal: show bounded correction remains stable under perturbations.

Recommended stressors:

- additive Gaussian noise on the revealed prefix;
- impulse shock in the revealed prefix;
- synthetic level shift after the input window;
- missing prefix values.

Recommended variants:

```text
Full GTTD, w/o clipping, w/o low-rank truncation, zero-shot
```

This directly supports the revised claim: bounded, non-intrusive correction
instead of parameter contamination.

