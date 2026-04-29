from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "appendix" / "results"
OUT = ROOT / "appendix" / "tables" / "appendix_tables.tex"


VARIANT_NAMES = {
    "zero_shot": "Zero-shot backbone",
    "full": "Full GTTD",
    "no_template": "w/o persistent template",
    "no_alignment": "w/o affine alignment",
    "no_low_rank": "w/o low-rank truncation",
    "no_low_freq_removal": "w/o low-frequency removal",
    "no_local_propagation": "w/o local propagation",
    "no_clipping": "w/o clipping",
    "no_tail": "w/o tail correction",
    "fixed_period_24": "fixed period = 24",
}


def pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:+.2f}\\%"


def num(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.4f}"


def ms(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.1f}"


def value_text(parameter: str, value: float) -> str:
    if parameter == "rank":
        return str(int(value))
    return f"{value:g}"


def table_component() -> str:
    path = RESULT_DIR / "ablation_dlinear_etth1_96.csv"
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{VARIANT_NAMES[row['variant']]} & {num(row['mse'])} & {num(row['mae'])} & "
            f"{pct(row['delta_mse_vs_full_pct'])} & {ms(row['latency_ms'])} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\caption{{Component ablation of the persistent residual-template adapter on DLinear/ETTh1 with horizon $H=96$. Lower MSE and MAE are better. $\Delta$MSE is measured relative to Full GTTD; positive values indicate degradation.}}
\label{{tab:appendix_component_ablation_real}}
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Variant}} & \textbf{{MSE}} & \textbf{{MAE}} & \textbf{{$\Delta$MSE vs. Full}} & \textbf{{Eval. time (ms)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def table_prefix() -> str:
    path = RESULT_DIR / "prefix_dlinear_etth1_96.csv"
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{int(row['k'])} & {num(row['mse'])} & {num(row['mae'])} & "
            f"{pct(row['improvement_vs_zero_shot_pct'])} & {ms(row['latency_ms'])} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Sensitivity to the number of revealed future prefix steps on DLinear/ETTh1 with horizon $H=96$. This table validates how much online evidence the adapter needs before correction becomes beneficial.}}
\label{{tab:appendix_prefix_sensitivity_real}}
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{ccccc}}
\toprule
\textbf{{Prefix $k$}} & \textbf{{MSE}} & \textbf{{MAE}} & \textbf{{Imp. vs. Zero-shot}} & \textbf{{Eval. time (ms)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_hparam() -> str:
    path = RESULT_DIR / "hparam_dlinear_etth1_96.csv"
    df = pd.read_csv(path)
    rows = []
    for parameter, group in df.groupby("parameter", sort=False):
        values = list(group.iterrows())
        for idx, (_, row) in enumerate(values):
            name = parameter.replace("_", " ")
            prefix = rf"\multirow{{{len(values)}}}{{*}}{{{name}}}" if idx == 0 else ""
            rows.append(
                f"{prefix} & {value_text(parameter, row['value'])} & "
                f"{num(row['mse'])} & {num(row['mae'])} & {ms(row['latency_ms'])} \\\\"
            )
        rows.append(r"\midrule")
    if rows:
        rows = rows[:-1]
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\caption{{Hyperparameter sensitivity of GTTD on DLinear/ETTh1 with horizon $H=96$. Each row changes one hyperparameter while keeping the others at the default setting.}}
\label{{tab:appendix_hparam_sensitivity_real}}
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{llccc}}
\toprule
\textbf{{Parameter}} & \textbf{{Value}} & \textbf{{MSE}} & \textbf{{MAE}} & \textbf{{Eval. time (ms)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def table_tail() -> str:
    frames = []
    for horizon in [192, 336]:
        df = pd.read_csv(RESULT_DIR / f"tail_dlinear_etth1_{horizon}.csv")
        df = df[df["variant"].isin(["full", "no_tail"])].copy()
        df["horizon"] = horizon
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    pivot = df.pivot(index="variant", columns="horizon", values="mse")
    full_192 = pivot.loc["full", 192]
    full_336 = pivot.loc["full", 336]
    no_tail_192 = pivot.loc["no_tail", 192]
    no_tail_336 = pivot.loc["no_tail", 336]
    delta_192 = (no_tail_192 - full_192) / full_192 * 100.0
    delta_336 = (no_tail_336 - full_336) / full_336 * 100.0
    return rf"""\begin{{table}}[t]
\centering
\caption{{Effect of the optional long-horizon tail correction on DLinear/ETTh1. The tail branch is inactive at $H=192$ and slightly hurts this $H=336$ setting, so it should be treated as an optional heuristic rather than a core claim.}}
\label{{tab:appendix_tail_correction_real}}
\setlength{{\tabcolsep}}{{7pt}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Variant}} & \textbf{{$H=192$ MSE}} & \textbf{{$H=336$ MSE}} \\
\midrule
Full GTTD & {num(full_192)} & {num(full_336)} \\
w/o tail correction & {num(no_tail_192)} & {num(no_tail_336)} \\
$\Delta$MSE vs. Full & {pct(delta_192)} & {pct(delta_336)} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_provenance() -> str:
    path = ROOT / "appendix" / "tables" / "result_provenance.csv"
    df = pd.read_csv(path)
    unverified = df[df["verified_or_official"] == False]
    grouped = (
        unverified.groupby(["backbone", "dataset"])["horizon"]
        .apply(lambda xs: "/".join(str(int(x)) for x in sorted(xs)))
        .reset_index()
    )
    rows = [
        f"{row['backbone']} & {row['dataset']} & {row['horizon']} & Rerun before final claim \\\\"
        for _, row in grouped.iterrows()
    ]
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\caption{{Unverified retained result rows. These rows are listed for transparency and should not be used as main experimental evidence until rerun with the current code.}}
\label{{tab:appendix_unverified_rows}}
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{llcl}}
\toprule
\textbf{{Backbone}} & \textbf{{Dataset}} & \textbf{{Horizons}} & \textbf{{Required action}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def main() -> None:
    sections = [
        "% Auto-generated appendix tables from real local runs.",
        "% Required packages: booktabs, multirow.",
        table_component(),
        table_prefix(),
        table_hparam(),
        table_tail(),
        table_provenance(),
    ]
    OUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
