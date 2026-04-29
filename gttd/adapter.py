from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import spsolve


def build_prefix_transfer(horizon: int, alpha: float = 0.5) -> np.ndarray:
    num_edges = horizon - 1
    row_indices, col_indices, data = [], [], []
    for i in range(num_edges):
        row_indices.extend([i, i])
        col_indices.extend([i, i + 1])
        data.extend([-1.0, 1.0])
    diff = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, horizon))
    lap = diff.T.dot(diff)
    reg = lap + alpha * sp.eye(horizon, format="csr")
    transfer = []
    for idx in range(horizon):
        source = np.zeros(horizon, dtype=np.float32)
        source[idx] = 1.0
        transfer.append(spsolve(reg, source).astype(np.float32))
    return np.stack(transfer, axis=1)


def low_freq_projection(residual: np.ndarray) -> np.ndarray:
    avail, n_var = residual.shape
    if avail <= 1:
        return residual.copy()
    x = np.linspace(-1.0, 1.0, avail, dtype=np.float32)
    design = np.stack([np.ones_like(x), x], axis=1)
    proj = np.zeros_like(residual)
    for j in range(n_var):
        coef, *_ = np.linalg.lstsq(design, residual[:, j], rcond=None)
        proj[:, j] = design @ coef
    return proj


@dataclass
class TemplateParams:
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


class ResidualTemplateAdapter:
    """Persistent-field GTTD adapter for batched forecasting outputs."""

    def __init__(self, horizon: int, seq_len: int):
        self.params = TemplateParams(horizon=horizon)
        self.transfer = build_prefix_transfer(horizon, alpha=self.params.local_alpha)
        self.cur_step = seq_len - 2
        self.pred_step_end_dict: dict[int, int] = {}
        self.full_template_dict: dict[int, np.ndarray] = {}
        self.template_state: np.ndarray | None = None

    def calculate_period_and_batch_size(self, enc_window_first: torch.Tensor) -> tuple[int, int]:
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
        u, s, vt = np.linalg.svd(template, full_matrices=False)
        rank = min(self.params.rank, len(s))
        return (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    def update_state_if_available(self) -> None:
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
        if self.template_state is None:
            return np.zeros((self.params.horizon, residual_prefix.shape[1]), dtype=np.float32)
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
            if avail < self.params.support_floor:
                continue

            residual_prefix = gt_np[i, :avail, :] - pred_np[i, :avail, :]
            global_corr = self.align_template(residual_prefix, avail)
            local_prefix = residual_prefix - global_corr[:avail]
            fast_residual = local_prefix - low_freq_projection(local_prefix)
            local_corr = self.transfer[:, :avail] @ fast_residual

            raw_err = float(np.mean(residual_prefix**2))
            template_err = float(np.mean((residual_prefix - global_corr[:avail]) ** 2))
            trust = max(0.0, 1.0 - template_err / (raw_err + 1e-8))
            horizon_factor = min(1.0, np.sqrt(192.0 / float(self.params.horizon)))
            effective_mix = self.params.global_mix * trust * horizon_factor

            tail_corr = 0.0
            if self.template_state is not None and self.params.horizon > 192:
                tail_weight = self.params.far_tail_mix * max(0.0, (float(self.params.horizon) - 192.0) / (720.0 - 192.0))
                ramp = np.linspace(0.0, 1.0, self.params.horizon, dtype=np.float32)[:, None] ** 1.5
                tail_corr = tail_weight * ramp * self.template_state

            total_corr = effective_mix * global_corr + local_corr + tail_corr
            total_corr = np.clip(total_corr, -self.params.clip_value, self.params.clip_value)
            out[i] = pred_np[i] + total_corr

        self.pred_step_end_dict[batch_idx] = self.cur_step + horizon
        self.full_template_dict[batch_idx] = np.mean(np.stack(full_templates, axis=0), axis=0).astype(np.float32)
        return torch.from_numpy(out).to(pred.device)
