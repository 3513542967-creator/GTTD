# Appendix Experiment Selection

This note records which appendix experiments are worth keeping for the current
GTTD implementation and which claims they support.

## Keep: Component Ablation

Necessary. The revised GTTD is not a graph Laplacian solver; it is a persistent
residual-template adapter. Component ablation directly validates the actual
mechanisms in code:

- persistent template memory;
- affine template alignment;
- SVD low-rank truncation;
- low-frequency residual removal;
- local prefix propagation;
- clipping;
- FFT-derived period scheduling.

Current real-data finding on DLinear/ETTh1/H=96: removing local propagation
causes the largest degradation, while removing low-frequency removal slightly
improves this setting. This is useful and honest: it identifies which modules
are core and which are heuristics.

## Keep: Sparse Prefix Sensitivity

Necessary. The paper discusses test-time adaptation from a small revealed
future prefix. This table shows when the adapter has enough online evidence to
help.

Current real-data finding on DLinear/ETTh1/H=96: k=1 gives zero-shot behavior,
k=2/3/6 can hurt, and k=12/24 becomes beneficial. This should be framed as a
limitation/behavioral property rather than hidden.

## Keep: Hyperparameter Sensitivity

Necessary. This validates that the method is not a single lucky setting.
Sensitivity over rank, template decay, global mix, local alpha, and clipping
also explains which knobs matter most.

Current real-data finding: local_alpha is important; too small a value can
substantially hurt. global_mix also has a clear optimum range. rank and clipping
are relatively stable in this setting.

## Keep with Caution: Long-Horizon Tail Correction

Useful, but not a main claim. It tests the `far_tail_mix` heuristic for long
horizons.

Current real-data finding: the tail branch is inactive at H=192 and slightly
hurts DLinear/ETTh1/H=336. Therefore, the paper should not present tail
correction as a core mechanism unless broader evidence supports it.

## Keep: Result Provenance

Necessary for paper hygiene. It prevents estimated or previous-run values from
being silently mixed with locally verified rows.

Current status: MICN rows and PatchTST/Weather rows require reruns before they
can support a main experimental claim.

## Defer: Stress Tests

Potentially useful for a later appendix revision, but not yet necessary before
the core method rewrite. Add stress tests only after the main text correctly
describes the residual-template adapter.

## Defer: Full Multi-Backbone Appendix Sweeps

Useful for camera-ready, but expensive. First stabilize the paper narrative with
the representative real-data tables. Then expand the same scripts to PatchTST,
OLS, and MICN if time allows.

