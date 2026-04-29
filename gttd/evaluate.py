from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .adapter import ResidualTemplateAdapter


def evaluate_gttd(cfg, model, get_test_dataloader, forecast, prepare_inputs) -> dict[str, float]:
    cfg_test = deepcopy(cfg)
    cfg_test.TEST.BATCH_SIZE = len(get_test_dataloader(cfg).dataset)
    test_loader = get_test_dataloader(cfg_test)
    adapter = ResidualTemplateAdapter(horizon=cfg.DATA.PRED_LEN, seq_len=cfg.DATA.SEQ_LEN)
    mse_all, mae_all = [], []

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

    return {"mse": float(np.concatenate(mse_all).mean()), "mae": float(np.concatenate(mae_all).mean())}
