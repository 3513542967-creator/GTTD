import math
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks" / "forecasting"
CHECKPOINT_ROOT = ROOT / "checkpoints"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from config import get_cfg_defaults
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_test_dataloader
from models.build import build_model, load_best_model
from models.forecast import forecast
from trainer import build_trainer
from utils.misc import prepare_inputs, set_devices, set_seeds


RESULTS_DIR = ROOT / "results" / "tta"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"]
HORIZONS = [96, 192, 336, 720]


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def build_cfg(dataset: str, horizon: int, backbone: str = "DLinear") -> object:
    cfg = get_cfg_defaults()
    update_cfg_from_dataset(cfg, dataset)
    cfg.SEED = 0
    cfg.VISIBLE_DEVICES = 0
    cfg.DATA.BASE_DIR = str(BENCHMARK_ROOT / "data")
    cfg.DATA.SEQ_LEN = 96
    cfg.DATA.LABEL_LEN = 48
    cfg.DATA.PRED_LEN = horizon
    cfg.MODEL.NAME = backbone
    cfg.MODEL.seq_len = 96
    cfg.MODEL.label_len = 48
    cfg.MODEL.pred_len = horizon
    cfg.TRAIN.ENABLE = True
    cfg.TEST.ENABLE = True
    cfg.TTA.ENABLE = False
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.TRAIN.CHECKPOINT_DIR = str(CHECKPOINT_ROOT / backbone / f"{dataset}_{horizon}")
    cfg.RESULT_DIR = str(RESULTS_DIR / f"run_{backbone.lower()}" / f"{dataset}_{horizon}")
    return cfg


def ensure_checkpoint(cfg) -> torch.nn.Module:
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)
    model = build_model(cfg)
    ckpt = Path(cfg.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth"
    if not ckpt.exists():
        Path(cfg.TRAIN.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        trainer = build_trainer(cfg, model, norm_module=None)
        trainer.train()
    model = load_best_model(cfg, model)
    model.eval()
    return model


def build_prefix_transfer(horizon: int, alpha: float = 0.5) -> np.ndarray:
    num_edges = horizon - 1
    row_indices, col_indices, data = [], [], []
    for i in range(num_edges):
        row_indices.extend([i, i])
        col_indices.extend([i, i + 1])
        data.extend([-1.0, 1.0])
    d_mat = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, horizon))
    l_mat = d_mat.T.dot(d_mat)
    l_reg = l_mat + alpha * sp.eye(horizon, format="csr")
    transfer = []
    for idx in range(horizon):
        source = np.zeros(horizon, dtype=np.float32)
        source[idx] = 1.0
        phi = spsolve(l_reg, source).astype(np.float32)
        transfer.append(phi)
    return np.stack(transfer, axis=1)  # [H, H]


@dataclass
class DualScaleParams:
    horizon: int
    memory_decay: float = 0.94
    trigger_threshold: float = 0.08
    gate_sharpness: float = 16.0
    ramp_midpoint: float = 0.28
    ramp_sharpness: float = 10.0
    local_alpha: float = 0.5


class OfficialDualScaleAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.params = DualScaleParams(horizon=cfg.DATA.PRED_LEN)
        self.transfer = build_prefix_transfer(cfg.DATA.PRED_LEN, alpha=self.params.local_alpha)
        self.state = None
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.residual_summary_dict = {}

    def _calculate_period_and_batch_size(self, enc_window_first: torch.Tensor) -> tuple[int, int]:
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            denom = torch.argmax(amplitude[:, power.argmax()]).item()
            period = enc_window_first.shape[0] // max(denom, 1)
        except Exception:
            period = 24
        period = max(1, int(period))
        batch_size = period + 1
        return period, batch_size

    def _update_full_state_if_available(self) -> None:
        while self.pred_step_end_dict and self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx = min(self.pred_step_end_dict.keys())
            summary = self.residual_summary_dict.pop(batch_idx)
            if self.state is None:
                self.state = summary.astype(np.float32)
            else:
                self.state = self.params.memory_decay * self.state + (1.0 - self.params.memory_decay) * summary.astype(np.float32)
            self.pred_step_end_dict.pop(batch_idx)

    def _global_gate(self) -> float:
        if self.state is None:
            return 0.0
        score = float(np.mean(np.abs(self.state)))
        x = score - self.params.trigger_threshold
        return float(1.0 / (1.0 + math.exp(-self.params.gate_sharpness * x)))

    def _global_corr(self, n_var: int) -> np.ndarray:
        if self.state is None:
            return np.zeros((self.params.horizon, n_var), dtype=np.float32)
        gate = self._global_gate()
        t = np.linspace(0.0, 1.0, self.params.horizon, dtype=np.float32)
        ramp = 1.0 / (1.0 + np.exp(-self.params.ramp_sharpness * (t - self.params.ramp_midpoint)))
        return (ramp[:, None] * self.state[None, :] * gate).astype(np.float32)

    def adapt_batch(self, pred: torch.Tensor, ground_truth: torch.Tensor, period: int, batch_idx: int) -> tuple[torch.Tensor, float]:
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        gt_np = ground_truth.detach().cpu().numpy().astype(np.float32)
        bsz, horizon, n_var = pred_np.shape
        out = pred_np.copy()
        global_corr = self._global_corr(n_var)
        for i in range(bsz):
            avail = max(period - i, 0)
            local_corr = np.zeros((horizon, n_var), dtype=np.float32)
            if avail > 0:
                residual = gt_np[i, :avail, :] - pred_np[i, :avail, :]
                local_corr = self.transfer[:, :avail] @ residual
            out[i] = pred_np[i] + local_corr + global_corr
        residual_summary = (gt_np - out).mean(axis=(0, 1)).astype(np.float32)
        self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
        self.residual_summary_dict[batch_idx] = residual_summary
        return torch.from_numpy(out).to(pred.device), self._global_gate()


def evaluate_ours(cfg, model) -> dict:
    cfg_test = deepcopy(cfg)
    cfg_test.TEST.BATCH_SIZE = len(get_test_dataloader(cfg).dataset)
    test_loader = get_test_dataloader(cfg_test)
    adapter = OfficialDualScaleAdapter(cfg)
    mse_all, mae_all = [], []
    gate_trace = []

    with torch.no_grad():
        for inputs in test_loader:
            enc_all, enc_stamp_all, dec_all, dec_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0
            batch_idx = 0
            while batch_end < len(enc_all):
                enc_first = enc_all[batch_start]
                period, batch_size = adapter._calculate_period_and_batch_size(enc_first)
                batch_end = min(batch_start + batch_size, len(enc_all))
                batch_size = batch_end - batch_start
                adapter.cur_step += batch_size
                sliced = (
                    enc_all[batch_start:batch_end],
                    enc_stamp_all[batch_start:batch_end],
                    dec_all[batch_start:batch_end],
                    dec_stamp_all[batch_start:batch_end],
                )
                adapter._update_full_state_if_available()
                pred, gt = forecast(cfg, sliced, model, None)
                pred_adj, gate = adapter.adapt_batch(pred, gt, period, batch_idx)
                mse = F.mse_loss(pred_adj, gt, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred_adj, gt, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mse_all.append(mse)
                mae_all.append(mae)
                gate_trace.append(gate)
                batch_start = batch_end
                batch_idx += 1

    mse_all = np.concatenate(mse_all)
    mae_all = np.concatenate(mae_all)
    return {
        "mse": float(mse_all.mean()),
        "mae": float(mae_all.mean()),
        "mean_gate": float(np.mean(gate_trace)) if gate_trace else 0.0,
    }


def evaluate_zero(cfg, model) -> dict:
    test_loader = get_test_dataloader(cfg)
    train_like_cfg = deepcopy(cfg)
    train_like_cfg.TEST.SPLIT = "train"
    train_loader = get_test_dataloader(train_like_cfg)

    def eval_loader(loader):
        mse_all, mae_all = [], []
        with torch.no_grad():
            for inputs in loader:
                pred, gt = forecast(cfg, inputs, model, None)
                mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, gt, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mse_all.append(mse)
                mae_all.append(mae)
        mse_all = np.concatenate(mse_all)
        mae_all = np.concatenate(mae_all)
        return float(mse_all.mean()), float(mae_all.mean())

    test_mse, test_mae = eval_loader(test_loader)
    train_mse, train_mae = eval_loader(train_loader)
    return {"test_mse": test_mse, "test_mae": test_mae, "train_mse": train_mse, "train_mae": train_mae}
