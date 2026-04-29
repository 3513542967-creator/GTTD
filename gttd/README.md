# GTTD Implementation

This folder contains the method implementation used by the retained benchmark
experiments.

Files:

```text
adapter.py   Persistent residual-template field adapter
evaluate.py  Benchmark evaluator wrapper
```

The experiment runners in `experiments/{dlinear,patchtst,ols,micn}/` import
this package rather than defining the method inline. This keeps the paper
method easy to find and separate from benchmark-specific wiring.
