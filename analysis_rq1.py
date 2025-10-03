#!/usr/bin/env python3
import argparse, os, pickle, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) 读取 run 目录里的最后一个 env 或 dense_log
# -----------------------------
def _pick_last_file(run_dir: Path, pat: str) -> Path | None:
    files = sorted(run_dir.glob(pat), key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    return files[-1] if files else None

def load_dense_from_run(run_dir: str) -> Dict:
    """优先 dense_log_240.pkl，其次 dense_log.pkl；若都不存在，退而从最后一个 env_xxx.pkl 里读 env.dense_log"""
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

    # 退路：从最后一个 env_*.pkl 里取 env.dense_log
    env_last = _pick_last_file(d, "env_*.pkl")
    if env_last is None:
        raise FileNotFoundError(f"No dense_log(_240).pkl and no env_*.pkl in {run_dir}")
    with open(env_last, "rb") as f:
        env = pickle.load(f)
    dense = getattr(env, "dense_log", None)
    if not isinstance(dense, dict):
        raise ValueError("env.dense_log is missing or not a dict")
    print(f"[INFO] Loaded dense log from {env_last} (env.dense_log)")
    return dense

# -----------------------------
# 2) 从 dense_log 按“月”提取：价格、就业、产出……
# -----------------------------
def extract_monthly(dense: Dict) -> pd.DataFrame:
    """
    期望结构：
      dense['world']   -> list[ dict(..., 'Price': float, ...)]
      dense['actions'] -> list[ dict(agent_id -> {'SimpleLabor': 0/1, ...}, 'p': ...)]
      dense['states']  -> list[ dict(agent_id -> {'skill': float, ...})]
    """
    world = dense.get("world")
    actions = dense.get("actions")
    states = dense.get("states")
    if not (isinstance(world, list) and isinstance(actions, list) and isinstance(states, list)):
        raise ValueError("dense_log doesn't have expected ['world','actions','states'] lists")

    # world/states 常比 actions 多 1（含初始态）
    n_months = len(actions)
    rows = []
    for m in range(n_months):
        w = world[m+1] if len(world) == n_months + 1 else world[m]
        a = actions[m]
        s = states[m+1] if len(states) == n_months + 1 else states[m]

        price = float(w.get("Price", float("nan")))

        # 代理 ID（排除 'p'）
        agent_ids = [k for k in a.keys() if k != "p"]
        n_agents = len(agent_ids)

        # —— 就业：以行动 SimpleLabor==1 为“工作”
        employed = 0
        # —— 真实产出：∑ (是否工作) × skill × 168
        total_real_output = 0.0
        # —— 平均工资：平均(skill × 168)
        wages = []

        for aid in agent_ids:
            ai = a.get(aid) or {}
            si = s.get(aid) or {}
            work_flag = int(bool(ai.get("SimpleLabor", 0)))
            skill = float(si.get("skill", 0.0))
            wage = skill * 168.0
            employed += work_flag
            total_real_output += work_flag * wage
            wages.append(wage)

        unemployment = 1.0 - (employed / n_agents) if n_agents > 0 else float("nan")
        avg_wage = float(np.mean(wages)) if wages else float("nan")

        rows.append({
            "month": m + 1,
            "price": price,
            "unemployment": unemployment,
            "avg_wage": avg_wage,
            "employed": employed,
            "real_output": total_real_output,   # 关键：真实产出（不含价格）
            "n_agents": n_agents,
        })
    return pd.DataFrame(rows)


# -----------------------------
# 3) 聚合到“年”（12 个月一组），计算通胀、名义/实际 GDP 及增长
# -----------------------------
def monthly_to_annual(df_monthly: pd.DataFrame, years: int = 20) -> pd.DataFrame:
    """
    - 年均价=当年12月 'price' 的均值；通胀=同比(今年均价/去年均价-1)
    - 年失业率=当年12月失业率的均值
    - 年真实产出=当年 'real_output' 求和
    - 名义GDP= 年真实产出 × 年均价
    - 实际GDP= 年真实产出 × 基期(第1年)均价
    - 名义GDP增长=同比
    """
    out = []
    base_price = df_monthly[df_monthly["month"] <= 12]["price"].mean()

    for y in range(1, years + 1):
        seg = df_monthly[(df_monthly["month"] > (y-1)*12) & (df_monthly["month"] <= y*12)]
        if seg.empty:
            break
        year_price = seg["price"].mean()
        year_unemp = seg["unemployment"].mean()
        total_real_output = seg["real_output"].sum()

        nominal_gdp = total_real_output * year_price
        real_gdp = total_real_output * base_price

        if y > 1:
            prev = out[-1]
            inflation = (year_price - prev["year_price"]) / prev["year_price"] if prev["year_price"] > 0 else 0.0
            nominal_gdp_growth = (nominal_gdp - prev["nominal_gdp"]) / prev["nominal_gdp"] if prev["nominal_gdp"] > 0 else 0.0
        else:
            inflation = 0.0
            nominal_gdp_growth = 0.0

        out.append({
            "year": y,
            "inflation": inflation,
            "unemployment": year_unemp,
            "year_price": year_price,
            "nominal_gdp": nominal_gdp,
            "real_gdp": real_gdp,
            "nominal_gdp_growth": nominal_gdp_growth,
        })
    return pd.DataFrame(out)


# -----------------------------
# 4) 画“论文风格”的 Figure 2（支持多条曲线）
# -----------------------------
def plot_figure2_multi(annual_map: Dict[str, pd.DataFrame], save_path: str):
    """
    annual_map: {label -> annual_df}
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # 颜色/线型可按论文图自定义（这里只给出简单示例）
    styles = {
        "LEN": {"ls": "--"},
        "CATS": {"ls": ":"},
        "Composite": {"ls": "-."},
        "AI-Eco": {"ls": (0, (3, 1, 1, 1))},
        "EconAgent": {"ls": "-", "lw": 2},
    }

    for label, df in annual_map.items():
        years = df["year"].values
        st = styles.get(label, {"ls": "-"})

        # (a) Inflation
        axes[0, 0].plot(years, df["inflation"].values, label=label, **st)

        # (b) Unemployment
        axes[0, 1].plot(years, df["unemployment"].values, label=label, **st)

        # (c) Nominal GDP
        axes[1, 0].plot(years, df["nominal_gdp"].values, label=label, **st)

        # (d) Nominal GDP Growth
        axes[1, 1].plot(years, df["nominal_gdp_growth"].values, label=label, **st)

    # 轴样式尽量贴近论文
    axes[0, 0].set_title("Inflation Rate");       axes[0, 0].axhline(0, color="k", ls="--", alpha=0.3)
    axes[0, 1].set_title("Unemployment Rate")
    axes[1, 0].set_title("Nominal GDP");          axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(6,6))
    axes[1, 1].set_title("Nominal GDP Growth");   axes[1, 1].axhline(0, color="k", ls="--", alpha=0.3)

    for ax in axes.ravel():
        ax.set_xlabel("year")
        ax.grid(True, alpha=0.3)

    # y 轴范围可按论文大致设置（你也可注释掉让它自适应）
    axes[0, 0].set_ylim(-0.25, 0.25)
    axes[0, 1].set_ylim(0.0, 0.12)
    axes[1, 1].set_ylim(-0.25, 0.25)

    # 图例放在顶部横排
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure saved to {save_path}")

# -----------------------------
# 5) CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Reproduce Figure 2 style plots (Inflation/Unemployment/Nominal GDP/Growth)")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help='多个 run 目录，形式为 Label=Path，例如 EconAgent=/.../gpt-3-perception-reflection-1-100agents-240months'
    )
    parser.add_argument("--out", required=True, help="输出图片路径，例如 figs/figure2.png")
    args = parser.parse_args()

    annual_map: Dict[str, pd.DataFrame] = {}
    for item in args.runs:
        if "=" not in item:
            raise ValueError("Each --runs item must be Label=Path")
        label, path = item.split("=", 1)
        dense = load_dense_from_run(path)
        monthly = extract_monthly(dense)
        annual = monthly_to_annual(monthly, years=20)
        annual_map[label] = annual

    plot_figure2_multi(annual_map, args.out)

if __name__ == "__main__":
    main()

