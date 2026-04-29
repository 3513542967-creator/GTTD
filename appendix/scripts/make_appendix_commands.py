from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "appendix" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SMOKE_RUNS = [
    "python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/ablation_dlinear_etth1_96.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/ablation_patchtst_etth1_96.csv",
]

COMPONENT_RUNS = [
    "python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/ablation_dlinear_etth1_96.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTm2 --horizon 96 --out appendix/results/ablation_dlinear_ettm2_96.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/ablation_patchtst_etth1_96.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone OLS --dataset ETTh1 --horizon 96 --out appendix/results/ablation_ols_etth1_96.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone MICN --dataset ETTh1 --horizon 96 --out appendix/results/ablation_micn_etth1_96.csv",
]

LONG_HORIZON_RUNS = [
    "python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 192 --variants full no_tail --out appendix/results/tail_dlinear_etth1_192.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone DLinear --dataset ETTh1 --horizon 336 --variants full no_tail --out appendix/results/tail_dlinear_etth1_336.csv",
    "python appendix/scripts/run_appendix_ablation.py --backbone PatchTST --dataset ETTh1 --horizon 336 --variants full no_tail --out appendix/results/tail_patchtst_etth1_336.csv",
]

PREFIX_RUNS = [
    "python appendix/scripts/run_prefix_sensitivity.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/prefix_dlinear_etth1_96.csv",
    "python appendix/scripts/run_prefix_sensitivity.py --backbone PatchTST --dataset ETTh1 --horizon 96 --out appendix/results/prefix_patchtst_etth1_96.csv",
]

HPARAM_RUNS = [
    "python appendix/scripts/run_hparam_sensitivity.py --backbone DLinear --dataset ETTh1 --horizon 96 --out appendix/results/hparam_dlinear_etth1_96.csv",
    "python appendix/scripts/run_hparam_sensitivity.py --backbone PatchTST --dataset ETTh1 --horizon 96 --parameters rank template_decay global_mix --out appendix/results/hparam_patchtst_etth1_96.csv",
]

VERIFY_GAPS = [
    "python scripts/run_gttd.py --backbone PatchTST --datasets weather --horizons 96 192 336 --out verify_patchtst_weather.csv",
    "python scripts/run_gttd.py --backbone MICN --datasets ETTh1 --horizons 96 --out verify_micn_etth1_96.csv",
    "python scripts/run_gttd.py --backbone OLS --datasets ETTh1 --horizons 96 --out verify_ols_etth1_96.csv",
]


def write_section(lines: list[str], title: str, commands: list[str]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    for cmd in commands:
        lines.append(f"```bash\n{cmd}\n```")
        lines.append("")


def main() -> None:
    lines = [
        "# Recommended Appendix Commands",
        "",
        "Run these commands from the repository root. Start with the smoke runs,",
        "then expand only after output schemas are confirmed.",
        "",
    ]
    write_section(lines, "Smoke Runs", SMOKE_RUNS)
    write_section(lines, "Component Ablation Runs", COMPONENT_RUNS)
    write_section(lines, "Sparse Prefix Sensitivity Runs", PREFIX_RUNS)
    write_section(lines, "Hyperparameter Sensitivity Runs", HPARAM_RUNS)
    write_section(lines, "Long-Horizon Tail Runs", LONG_HORIZON_RUNS)
    write_section(lines, "Verification Gap Runs", VERIFY_GAPS)
    out = OUT_DIR / "recommended_commands.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
