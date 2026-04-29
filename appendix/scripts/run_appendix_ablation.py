from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "benchmarks" / "forecasting"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from datasets.loader import get_test_dataloader
from experiments.common import build_cfg, ensure_checkpoint, evaluate_zero
from gttd.adapter import build_prefix_transfer, low_freq_projection
from models.forecast import forecast
from utils.misc import prepare_inputs


@dataclass
class VariantParams:
    horizon: int
    local_alpha: float = 0.5
    support_floor: int = 2
    template_decay: float = 0.92
    global_mix: float = 0.75
    scale_prior: float = 0.8
    bias_prior: float = 0.1
    clip_value: float = 2.5
    rank: int = 4
    far_tail_mix: float = 0.18
    use_template: bool = True
    use_alignment: bool = True
    use_low_rank: bool = True
    use_low_freq_removal: bool = True
    use_local_propagation: bool = True
    use_clipping: bool = True
    use_tail: bool = True
    fixed_period: int | None = None
    max_prefix_k: int | None = None


VARIANTS = {
    "full": {},
    "no_template": {"use_template": False},
    "no_alignment": {"use_alignment": False},
    "no_low_rank": {"use_low_rank": False},
    "no_low_freq_removal": {"use_low_freq_removal": False},
    "no_local_propagation": {"use_local_propagation": False},
    "no_clipping": {"use_clipping": False},
    "no_tail": {"use_tail": False},
    "fixed_period_24": {"fixed_period": 24},
}


class AppendixResidualTemplateAdapter:
    def __init__(self, horizon: int, seq_len: int, variant: str, overrides: dict[str, float | int | None] | None = None):
        params = VariantParams(horizon=horizon)
        for key, value in VARIANTS[variant].items():
            setattr(params, key, value)
        for key, value in (overrides or {}).items():
            if value is not None:
                setattr(params, key, value)
        self.params = params
        self.transfer = build_prefix_transfer(horizon, alpha=self.params.local_alpha)
        self.cur_step = seq_len - 2
        self.pred_step_end_dict: dict[int, int] = {}
        self.full_template_dict: dict[int, np.ndarray] = {}
        self.template_state: np.ndarray | None = None

    def calculate_period_and_batch_size(self, enc_window_first: torch.Tensor) -> tuple[int, int]:
        if self.params.fixed_period is not None:
            period = self.params.fixed_period
            return period, period + 1
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude**2, dim=0)
        try:
            denom = torch.argmax(amplitude[:, power.argmax()]).item()
            period = enc_window_first.shape[0] // max(denom, 1)
        except Exception:
            period = 24
        period = max(1, int(period))
        return period, period + 1

    def truncate_template(self, template: np.ndarray) -> np.ndarray:
        if not self.params.use_low_rank:
            return template.astype(np.float32)
        u, s, vt = np.linalg.svd(template, full_matrices=False)
        rank = min(self.params.rank, len(s))
        return ((u[:, :rank] * s[:rank]) @ vt[:rank, :]).astype(np.float32)

    def update_state_if_available(self) -> None:
        if not self.params.use_template:
            self.pred_step_end_dict.clear()
            self.full_template_dict.clear()
            self.template_state = None
            return
        while self.pred_step_end_dict and self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx = min(self.pred_step_end_dict.keys())
            template = self.truncate_template(self.full_template_dict.pop(batch_idx))
            if self.template_state is None:
                self.template_state = template.astype(np.float32)
            else:
                self.template_state = (
                    self.params.template_decay * self.template_state
                    + (1.0 - self.params.template_decay) * template.astype(np.float32)
                )
            self.pred_step_end_dict.pop(batch_idx)

    def align_template(self, residual_prefix: np.ndarray, avail: int) -> np.ndarray:
        if not self.params.use_template or self.template_state is None:
            return np.zeros((self.params.horizon, residual_prefix.shape[1]), dtype=np.float32)
        if not self.params.use_alignment:
            return self.template_state.copy()
        template_prefix = self.template_state[:avail]
        n_var = residual_prefix.shape[1]
        corr = np.zeros_like(self.template_state)
        for j in range(n_var):
            x = template_prefix[:, j]
            y = residual_prefix[:, j]
            design = np.stack([x, np.ones_like(x)], axis=1)
            reg = np.diag([self.params.scale_prior, self.params.bias_prior]).astype(np.float32)
            rhs = design.T @ y + np.array([self.params.scale_prior, 0.0], dtype=np.float32)
            mat = design.T @ design + reg
            scale, bias = np.linalg.solve(mat, rhs).astype(np.float32)
            corr[:, j] = scale * self.template_state[:, j] + bias
        return corr

    def adapt_batch(self, pred: torch.Tensor, ground_truth: torch.Tensor, period: int, batch_idx: int) -> torch.Tensor:
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        gt_np = ground_truth.detach().cpu().numpy().astype(np.float32)
        batch_size, horizon, _ = pred_np.shape
        out = pred_np.copy()
        full_templates = []

        for i in range(batch_size):
            residual_full = gt_np[i] - pred_np[i]
            full_templates.append(residual_full)
            avail = max(period - i, 0)
            if self.params.max_prefix_k is not None:
                avail = min(avail, int(self.params.max_prefix_k))
            if avail < self.params.support_floor:
                continue

            residual_prefix = gt_np[i, :avail, :] - pred_np[i, :avail, :]
            global_corr = self.align_template(residual_prefix, avail)
            local_prefix = residual_prefix - global_corr[:avail]
            if self.params.use_low_freq_removal:
                fast_residual = local_prefix - low_freq_projection(local_prefix)
            else:
                fast_residual = local_prefix

            if self.params.use_local_propagation:
                local_corr = self.transfer[:, :avail] @ fast_residual
            else:
                local_corr = np.zeros_like(global_corr)

            raw_err = float(np.mean(residual_prefix**2))
            template_err = float(np.mean((residual_prefix - global_corr[:avail]) ** 2))
            trust = max(0.0, 1.0 - template_err / (raw_err + 1e-8))
            horizon_factor = min(1.0, np.sqrt(192.0 / float(self.params.horizon)))
            effective_mix = self.params.global_mix * trust * horizon_factor

            tail_corr = 0.0
            if (
                self.params.use_template
                and self.params.use_tail
                and self.template_state is not None
                and self.params.horizon > 192
            ):
                tail_weight = self.params.far_tail_mix * max(0.0, (float(self.params.horizon) - 192.0) / (720.0 - 192.0))
                ramp = np.linspace(0.0, 1.0, self.params.horizon, dtype=np.float32)[:, None] ** 1.5
                tail_corr = tail_weight * ramp * self.template_state

            total_corr = effective_mix * global_corr + local_corr + tail_corr
            if self.params.use_clipping:
                total_corr = np.clip(total_corr, -self.params.clip_value, self.params.clip_value)
            out[i] = pred_np[i] + total_corr

        self.pred_step_end_dict[batch_idx] = self.cur_step + horizon
        self.full_template_dict[batch_idx] = np.mean(np.stack(full_templates, axis=0), axis=0).astype(np.float32)
        return torch.from_numpy(out).to(pred.device)


def evaluate_variant(cfg, model, variant: str, overrides: dict[str, float | int | None] | None = None) -> dict[str, float]:
    cfg_test = cfg.clone()
    cfg_test.TEST.BATCH_SIZE = len(get_test_dataloader(cfg).dataset)
    test_loader = get_test_dataloader(cfg_test)
    adapter = AppendixResidualTemplateAdapter(
        horizon=cfg.DATA.PRED_LEN,
        seq_len=cfg.DATA.SEQ_LEN,
        variant=variant,
        overrides=overrides,
    )
    mse_all, mae_all = [], []
    start = time.perf_counter()

    with torch.no_grad():
        for inputs in test_loader:
            enc_all, enc_stamp_all, dec_all, dec_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0
            batch_idx = 0
            while batch_end < len(enc_all):
                enc_first = enc_all[batch_start]
                period, batch_size = adapter.calculate_period_and_batch_size(enc_first)
                batch_end = min(batch_start + batch_size, len(enc_all))
                adapter.cur_step += batch_end - batch_start
                sliced = (
                    enc_all[batch_start:batch_end],
                    enc_stamp_all[batch_start:batch_end],
                    dec_all[batch_start:batch_end],
                    dec_stamp_all[batch_start:batch_end],
                )
                adapter.update_state_if_available()
                pred, gt = forecast(cfg, sliced, model, None)
                pred_adj = adapter.adapt_batch(pred, gt, period, batch_idx)
                mse = F.mse_loss(pred_adj, gt, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred_adj, gt, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mse_all.append(mse)
                mae_all.append(mae)
                batch_start = batch_end
                batch_idx += 1

    return {
        "mse": float(np.concatenate(mse_all).mean()),
        "mae": float(np.concatenate(mae_all).mean()),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run appendix ablations for the current residual-template GTTD.")
    parser.add_argument("--backbone", choices=["DLinear", "PatchTST", "OLS", "MICN"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--max-prefix-k", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--template-decay", type=float, default=None)
    parser.add_argument("--global-mix", type=float, default=None)
    parser.add_argument("--local-alpha", type=float, default=None)
    parser.add_argument("--clip-value", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    overrides = {
        "max_prefix_k": args.max_prefix_k,
        "rank": args.rank,
        "template_decay": args.template_decay,
        "global_mix": args.global_mix,
        "local_alpha": args.local_alpha,
        "clip_value": args.clip_value,
    }

    cfg = build_cfg(args.dataset, args.horizon, args.backbone)
    model = ensure_checkpoint(cfg)
    zero = evaluate_zero(cfg, model)
    rows = [
        {
            "backbone": args.backbone,
            "dataset": args.dataset,
            "horizon": args.horizon,
            "variant": "zero_shot",
            "mse": zero["test_mse"],
            "mae": zero["test_mae"],
            "delta_mse_vs_full_pct": np.nan,
            "latency_ms": 0.0,
            "max_prefix_k": args.max_prefix_k,
            "rank": args.rank,
            "template_decay": args.template_decay,
            "global_mix": args.global_mix,
            "local_alpha": args.local_alpha,
            "clip_value": args.clip_value,
        }
    ]

    full_mse = None
    for variant in args.variants:
        result = evaluate_variant(cfg, model, variant, overrides=overrides)
        if variant == "full":
            full_mse = result["mse"]
        delta = np.nan if full_mse is None else (result["mse"] - full_mse) / full_mse * 100.0
        rows.append(
            {
                "backbone": args.backbone,
                "dataset": args.dataset,
                "horizon": args.horizon,
                "variant": variant,
                "mse": result["mse"],
                "mae": result["mae"],
                "delta_mse_vs_full_pct": delta,
                "latency_ms": result["latency_ms"],
                "max_prefix_k": args.max_prefix_k,
                "rank": args.rank,
                "template_decay": args.template_decay,
                "global_mix": args.global_mix,
                "local_alpha": args.local_alpha,
                "clip_value": args.clip_value,
            }
        )
        print(f"[{variant}] mse={result['mse']:.6f} mae={result['mae']:.6f}")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
