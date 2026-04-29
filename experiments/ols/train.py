from __future__ import annotations

import argparse

from experiments.common import build_cfg, ensure_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    ensure_checkpoint(build_cfg(args.dataset, args.horizon, backbone="OLS"))
