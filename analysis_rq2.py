#!/usr/bin/env python3
import argparse, os, pickle, re
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1) 读取 dense_log
# ==================================================
def _pick_last_file(run_dir: Path, pat: str) -> Path | None:
    files = sorted(run_dir.glob(pat), key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    return files[-1] if files else None

def load_dense_from_run(run_dir: str) -> Dict:
    d = Path(run_dir)
    for fname in ["dense_log_240.pkl", "dense_log.pkl"]:
        p = d / fname
        if p.exists() and p.stat().st_size > 0:
            with open(p, "rb") as f:
                dense = pickle.load(f)
            if not isinstance(dense, dict):
                raise ValueError(f"{p} is not a dict")
            print(f"[INFO] Loaded dense log from {p}")
            return dense
    env_last = _pick_last_file(d, "env_*.pkl")
    if env_last is None:
        raise FileNotFoundError(f"No dense_log(_240).pkl or env_*.pkl in {run_dir}")
    with open(env_last, "rb") as f:
        env = pickle.load(f)
    dense = getattr(env, "dense_log", None)
    if not isinstance(dense, dict):
        raise ValueError("env.dense_log is missing or not a dict")
    print(f"[INFO] Loaded dense log from {env_last} (env.dense_log)")
    return dense


# ==================================================
# 2) 提取月度数据：失业率、工资、产出
# ==================================================
def extract_monthly(dense: Dict) -> pd.DataFrame:
    world, actions, states = dense.get("world"), dense.get("actions"), dense.get("states")
    if not (isinstance(world, list) and isinstance(actions, list) and isinstance(states, list)):
        raise ValueError("dense_log missing expected keys")

    n_months = len(actions)
    rows = []
    for m in range(n_months):
        w = world[m + 1] if len(world) == n_months + 1 else world[m]
        a = actions[m]
        s = states[m + 1] if len(states) == n_months + 1 else states[m]

        price = float(w.get("Price", np.nan))
        agent_ids = [k for k in a.keys() if k != "p"]
        n_agents = len(agent_ids)

        employed = sum(1 for aid in agent_ids if (a.get(aid, {}) or {}).get("SimpleLabor", 0) >= 1)
        unemployment = 1.0 - employed / n_agents if n_agents > 0 else np.nan

        avg_wage = np.mean([s.get(aid, {}).get("skill", 0) * 168 for aid in agent_ids]) if agent_ids else np.nan
        total_output = sum(
            s.get(aid, {}).get("skill", 0.0) * 168
            for aid in agent_ids
            if (a.get(aid, {}) or {}).get("SimpleLabor", 0) >= 1
        )

        rows.append({
            "month": m + 1,
            "price": price,
            "unemployment": unemployment,
            "avg_wage": avg_wage,
            "real_output": total_output,
        })
    return pd.DataFrame(rows)


# ==================================================
# 3) 聚合到年度，计算工资通胀、GDP增长
# ==================================================
def monthly_to_annual(df: pd.DataFrame, years: int = 20) -> pd.DataFrame:
    out = []
    base_price = df[df["month"] <= 12]["price"].mean()

    for y in range(1, years + 1):
        seg = df[(df["month"] > (y-1)*12) & (df["month"] <= y*12)]
        if seg.empty: break
        year_price = seg["price"].mean()
        year_unemp = seg["unemployment"].mean()
        avg_wage = seg["avg_wage"].mean()
        total_real_output = seg["real_output"].sum()
        real_gdp = total_real_output * base_price

        if y > 1:
            prev = out[-1]
            wage_inflation = (avg_wage - prev["avg_wage"]) / prev["avg_wage"] if prev["avg_wage"] > 0 else 0
            real_gdp_growth = (real_gdp - prev["real_gdp"]) / prev["real_gdp"] if prev["real_gdp"] > 0 else 0
            unemp_growth = year_unemp - prev["unemployment"]
        else:
            wage_inflation = real_gdp_growth = unemp_growth = 0

        out.append({
            "year": y,
            "unemployment": year_unemp,
            "avg_wage": avg_wage,
            "wage_inflation": wage_inflation,
            "real_gdp": real_gdp,
            "real_gdp_growth": real_gdp_growth,
            "unemp_growth": unemp_growth,
        })
    return pd.DataFrame(out)


# ==================================================
# 4) 绘制 Figure 3：Phillips Curve & Okun’s Law
# ==================================================
def plot_macroeconomic_regularities(annual_map: Dict[str, pd.DataFrame], save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = {
        "LEN": {"marker": "o", "alpha": 0.6},
        "CATS": {"marker": "s", "alpha": 0.6},
        "Composite": {"marker": "^", "alpha": 0.6},
        "AI-Eco": {"marker": "x", "alpha": 0.6},
        "EconAgent": {"marker": "o", "alpha": 0.9, "c": "tab:blue"},
    }

    for label, df in annual_map.items():
        st = styles.get(label, {"marker": "o"})
        # === Phillips Curve ===
        axes[0].scatter(df["unemployment"], df["wage_inflation"], label=label, **st)
        # === Okun’s Law ===
        axes[1].scatter(df["unemp_growth"], df["real_gdp_growth"], label=label, **st)

        # Pearson r
        corr_ph = df["unemployment"].corr(df["wage_inflation"])
        corr_ok = df["unemp_growth"].corr(df["real_gdp_growth"])
        print(f"[INFO] {label}: Phillips r={corr_ph:.3f}, Okun r={corr_ok:.3f}")

    # ---- 图形样式 ----
    axes[0].set_title("Phillips Curve")
    axes[0].set_xlabel("Unemployment Rate")
    axes[0].set_ylabel("Wage Inflation")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Okun’s Law")
    axes[1].set_xlabel("Unemployment Rate Growth")
    axes[1].set_ylabel("Real GDP Growth")
    axes[1].grid(True, alpha=0.3)

    fig.legend(loc="upper center", ncol=min(5, len(annual_map)))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved macroeconomic regularities to {save_path}")


# ==================================================
# 5) CLI
# ==================================================
def main():
    parser = argparse.ArgumentParser("Plot Phillips Curve and Okun’s Law (Figure 3)")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="多个 run 目录，形式为 Label=Path，例如 EconAgent=/.../run-dir")
    parser.add_argument("--out", required=True, help="输出图片路径，例如 figs/figure3.png")
    args = parser.parse_args()

    annual_map = {}
    for item in args.runs:
        label, path = item.split("=", 1)
        print(f"\n=== Processing {label} ===")
        dense = load_dense_from_run(path)
        monthly = extract_monthly(dense)
        annual = monthly_to_annual(monthly, years=20)
        annual_map[label] = annual

    plot_macroeconomic_regularities(annual_map, args.out)


if __name__ == "__main__":
    main()
