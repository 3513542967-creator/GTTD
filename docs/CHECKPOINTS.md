# Checkpoints

Best checkpoints live under:

```text
checkpoints/<Backbone>/<Dataset>_<Horizon>/checkpoint_best.pth
```

Checkpoint binaries are large and are excluded from git by the repository
`.gitignore`. For a public release, upload them through GitHub Releases, Git LFS,
or another archival host, then document the download link here.

Retained backbone folders:

```text
checkpoints/DLinear/
checkpoints/PatchTST/
checkpoints/OLS/
checkpoints/MICN/
```

If an `OLS` or `MICN` checkpoint is missing, the corresponding runner trains it
on demand through the retained forecasting benchmark code.
