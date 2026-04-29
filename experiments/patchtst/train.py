from __future__ import annotations

import argparse

from experiments.common import ensure_checkpoint
from experiments.patchtst.run_gttd import build_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--checkpoint-tag", default=None)
    args = parser.parse_args()
    ensure_checkpoint(build_cfg(args.dataset, args.horizon, checkpoint_tag=args.checkpoint_tag))
