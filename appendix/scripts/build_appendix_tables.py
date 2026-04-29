from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "appendix" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TABLE_ROWS = [
    {
        "table_id": "A1",
        "title": "Component ablation of the persistent residual-template adapter",
        "columns": "backbone,dataset,horizon,variant,mse,mae,delta_mse_vs_full_pct,latency_ms",
        "status": "template",
    },
    {
        "table_id": "A2",
        "title": "Sparse revealed-prefix sensitivity",
        "columns": "backbone,dataset,horizon,k,mse,mae,improvement_vs_zero_shot_pct",
        "status": "template",
    },
    {
        "table_id": "A3",
        "title": "Hyperparameter sensitivity",
        "columns": "parameter,value,backbone,dataset,horizon,mse,mae",
        "status": "template",
    },
    {
        "table_id": "A4",
        "title": "Result provenance and verification status",
        "columns": "backbone,dataset,horizon,source,verified_or_official,action",
        "status": "template",
    },
    {
        "table_id": "A5",
        "title": "Long-horizon tail-correction behavior",
        "columns": "backbone,dataset,horizon,variant,mse,mae,delta_mse_vs_full_pct",
        "status": "template",
    },
    {
        "table_id": "A6",
        "title": "Latency and memory overhead",
        "columns": "backbone,dataset,horizon,zero_shot_ms,adapter_ms,total_ms,peak_gpu_mb",
        "status": "template",
    },
    {
        "table_id": "A7",
        "title": "Stress tests under noisy or corrupted revealed prefixes",
        "columns": "backbone,dataset,horizon,stress_type,variant,mse,mae,stability_note",
        "status": "template",
    },
]


def main() -> None:
    out = OUT_DIR / "appendix_table_templates.csv"
    pd.DataFrame(TABLE_ROWS).to_csv(out, index=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

