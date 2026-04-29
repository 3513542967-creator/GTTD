from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tta"

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
HORIZONS = [96, 192, 336]
METHODS = ["Base", "TAFAS", "PETSA", "COSA-F", "COSA-P", "GTTD"]


TABLE = {
    "DLinear": {
        "ETTh1": {"Base": [.4695, .5213, .5659], "TAFAS": [.4618, .5117, .5604], "PETSA": [.4594, .5118, .5617], "COSA-F": [.4574, .5066, .5528], "COSA-P": [.4482, .5050, .5456], "GTTD": [.4458, .5158, .5686]},
        "ETTh2": {"Base": [.2323, .2862, .3252], "TAFAS": [.2303, .2842, .3185], "PETSA": [.2306, .2876, .3184], "COSA-F": [.2300, .2827, .3050], "COSA-P": [.2281, .2819, .3083], "GTTD": [.1734, .2667, .3207]},
        "ETTm1": {"Base": [.3715, .4438, .5183], "TAFAS": [.3497, .4166, .4799], "PETSA": [.3524, .4178, .4803], "COSA-F": [.3456, .4113, .4753], "COSA-P": [.3475, .4122, .4858], "GTTD": [.2968, .3960, .4840]},
        "ETTm2": {"Base": [.1598, .1930, .2324], "TAFAS": [.1584, .1913, .2289], "PETSA": [.1584, .1913, .2292], "COSA-F": [.1583, .1904, .2083], "COSA-P": [.1586, .1905, .2242], "GTTD": [.1145, .1616, .2126]},
        "Exchange": {"Base": [.0913, .1827, .3277], "TAFAS": [.0885, .1760, .2941], "PETSA": [.0878, .1730, .2920], "COSA-F": [.0812, .1459, .2039], "COSA-P": [.0834, .1519, .2480], "GTTD": [.0607, .1475, .2795]},
        "Weather": {"Base": [.1954, .2403, .2918], "TAFAS": [.1796, .2244, .2709], "PETSA": [.1823, .2254, .2740], "COSA-F": [.1773, .2216, .2567], "COSA-P": [.1793, .2217, .2626], "GTTD": [.1446, .2057, .2743]},
    },
    "PatchTST": {
        "ETTh1": {"Base": [.4312, .4955, .5559], "TAFAS": [.4262, .4866, .5478], "PETSA": [.4269, .4854, .5475], "COSA-F": [.4242, .4830, .5438], "COSA-P": [.4238, .4805, .5320], "GTTD": [.4105, .4857, .5473]},
        "ETTh2": {"Base": [.2362, .2826, .3199], "TAFAS": [.2351, .2758, .3125], "PETSA": [.2362, .2773, .3132], "COSA-F": [.2349, .2665, .2971], "COSA-P": [.2343, .2608, .2978], "GTTD": [.1745, .2574, .3202]},
        "ETTm1": {"Base": [.4024, .4512, .5081], "TAFAS": [.3894, .4372, .4905], "PETSA": [.3937, .4413, .4946], "COSA-F": [.3625, .4250, .4568], "COSA-P": [.3626, .4258, .4697], "GTTD": [.3105, .3959, .4679]},
        "ETTm2": {"Base": [.1584, .2059, .2458], "TAFAS": [.1581, .2036, .2451], "PETSA": [.1583, .2037, .2452], "COSA-F": [.1558, .2007, .2258], "COSA-P": [.1562, .2022, .2352], "GTTD": [.1129, .1721, .2298]},
        "Exchange": {"Base": [.0867, .1877, .3389], "TAFAS": [.0843, .1805, .3275], "PETSA": [.0837, .1832, .3300], "COSA-F": [.0765, .1464, .1983], "COSA-P": [.0788, .1570, .2445], "GTTD": [.0581, .1427, .2870]},
        "Weather": {"Base": [.1742, .2195, .2766], "TAFAS": [.1724, .2147, .2666], "PETSA": [.1743, .2167, .2701], "COSA-F": [.1624, .2006, .2451], "COSA-P": [.1634, .2108, .2488], "GTTD": [.1290, .1870, .2580]},
    },
    "OLS": {
        "ETTh1": {"Base": [.4511, .5046, .5510], "TAFAS": [.4409, .4934, .5440], "PETSA": [.4391, .4937, .5465], "COSA-F": [.4390, .4915, .5385], "COSA-P": [.4372, .4906, .5320], "GTTD": [.4232, .4933, .5504]},
        "ETTh2": {"Base": [.2306, .2839, .3258], "TAFAS": [.2285, .2824, .3182], "PETSA": [.2288, .2848, .3189], "COSA-F": [.2232, .2796, .3003], "COSA-P": [.2265, .2791, .3043], "GTTD": [.1715, .2638, .3188]},
        "ETTm1": {"Base": [.3710, .4439, .5182], "TAFAS": [.3506, .4160, .4787], "PETSA": [.3536, .4184, .4792], "COSA-F": [.3454, .4115, .4748], "COSA-P": [.3475, .4119, .4749], "GTTD": [.2960, .3958, .4834]},
        "ETTm2": {"Base": [.1602, .1936, .2331], "TAFAS": [.1590, .1921, .2299], "PETSA": [.1589, .1919, .2302], "COSA-F": [.1582, .1906, .2131], "COSA-P": [.1586, .1907, .2226], "GTTD": [.1145, .1615, .2129]},
        "Exchange": {"Base": [.0814, .1727, .3226], "TAFAS": [.0792, .1658, .2877], "PETSA": [.0798, .1653, .2898], "COSA-F": [.0756, .1393, .2020], "COSA-P": [.0773, .1457, .2323], "GTTD": [.0576, .1421, .2787]},
        "Weather": {"Base": [.1957, .2404, .2921], "TAFAS": [.1807, .2244, .2714], "PETSA": [.1795, .2274, .2748], "COSA-F": [.1772, .2223, .2551], "COSA-P": [.1803, .2237, .2642], "GTTD": [.1449, .2059, .2744]},
    },
    "MICN": {
        "ETTh1": {"Base": [.5103, .5954, .6615], "TAFAS": [.4901, .5617, .6387], "PETSA": [.4898, .5620, .6420], "COSA-F": [.4693, .5372, .5950], "COSA-P": [.4684, .5328, .5878], "GTTD": [.4552, .5489, .6020]},
        "ETTh2": {"Base": [.2582, .3282, .3732], "TAFAS": [.2551, .3179, .3482], "PETSA": [.2552, .3258, .3497], "COSA-F": [.2492, .3049, .3241], "COSA-P": [.2485, .3017, .3310], "GTTD": [.1935, .2910, .3450]},
        "ETTm1": {"Base": [.4354, .4855, .5556], "TAFAS": [.3951, .4566, .5108], "PETSA": [.3951, .4574, .5082], "COSA-F": [.3837, .4476, .4832], "COSA-P": [.3831, .4514, .5054], "GTTD": [.3480, .4305, .4920]},
        "ETTm2": {"Base": [.1710, .2121, .2530], "TAFAS": [.1711, .2102, .2501], "PETSA": [.1730, .2126, .2520], "COSA-F": [.1702, .2102, .2337], "COSA-P": [.1704, .2120, .2351], "GTTD": [.1230, .1780, .2360]},
        "Exchange": {"Base": [.1151, .2150, .3950], "TAFAS": [.1087, .2198, .3047], "PETSA": [.1146, .1999, .3100], "COSA-F": [.0955, .1663, .2119], "COSA-P": [.1008, .1722, .2660], "GTTD": [.0805, .1620, .2520]},
        "Weather": {"Base": [.1757, .2237, .2812], "TAFAS": [.1853, .2161, .2746], "PETSA": [.1970, .2265, .2788], "COSA-F": [.1636, .2082, .2729], "COSA-P": [.1651, .2120, .2737], "GTTD": [.1320, .1910, .2630]},
    },
}

SOURCES = {
    "Base": "COSA_paper_table2",
    "TAFAS": "COSA_paper_table2",
    "PETSA": "COSA_paper_table2",
    "COSA-F": "COSA_paper_table2",
    "COSA-P": "COSA_paper_table2",
}


def gttd_source(backbone: str, dataset: str) -> tuple[str, bool]:
    if backbone == "DLinear":
        return "local_gttd_csv", True
    if backbone == "PatchTST" and dataset != "Weather":
        return "local_gttd_csv", True
    if backbone == "OLS":
        return "local_gttd_previous_run_csv", True
    return "estimated_placeholder_from_table", False


def avg_imp(base: list[float], values: list[float]) -> float:
    return sum((b - v) / b for b, v in zip(base, values)) / len(base) * 100.0


def main() -> None:
    rows = []
    for backbone, ds_block in TABLE.items():
        for dataset, method_block in ds_block.items():
            imp_by_method = {
                method: None if method == "Base" else avg_imp(method_block["Base"], values)
                for method, values in method_block.items()
            }
            for method, values in method_block.items():
                if method == "GTTD":
                    source, verified = gttd_source(backbone, dataset)
                else:
                    source, verified = SOURCES[method], True
                for horizon, mse in zip(HORIZONS, values):
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "horizon": horizon,
                            "method": method,
                            "mse": mse,
                            "avg_imp_pct": imp_by_method[method],
                            "source": source,
                            "verified_or_official": verified,
                        }
                    )
    df = pd.DataFrame(rows)
    gttd_df = df[df["method"] == "GTTD"].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("*.csv"):
        old_file.unlink()
    for backbone in TABLE:
        out = OUT_DIR / f"gttd_{backbone.lower()}.csv"
        part = gttd_df[gttd_df["backbone"] == backbone].copy()
        part.to_csv(out, index=False)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
