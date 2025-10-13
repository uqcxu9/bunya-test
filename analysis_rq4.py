#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind

# ========== 工具：读取 ==========
def _pick_last(d: Path, pat: str) -> Path | None:
    files = sorted(d.glob(pat), key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    return files[-1] if files else None

def load_dense(run_dir: str):
    d = Path(run_dir)
    for name in ["dense_log_240.pkl", "dense_log.pkl"]:
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            print(f"[INFO] Loaded {p}")
            with open(p, "rb") as f:
                return pickle.load(f)
    env_last = _pick_last(d, "env_*.pkl")
    if env_last:
        with open(env_last, "rb") as f:
            env = pickle.load(f)
        print(f"[INFO] Loaded {env_last} (env.dense_log)")
        return env.dense_log
    raise FileNotFoundError(f"No dense_log(_240).pkl or env_*.pkl in {run_dir}")

def load_env_for_age(run_dir: str):
    d = Path(run_dir)
    env_last = _pick_last(d, "env_*.pkl")
    if not env_last: 
        return None
    with open(env_last, "rb") as f:
        return pickle.load(f)

# ========== 主提取：构造面板 ==========
def extract_panel(dense: dict, run_dir: str) -> pd.DataFrame:
    """
    生成回归用的 agent-level 面板数据，严格贴合论文：
      目标变量：pw_i（工作倾向）、pc_i（消费倾向）
      自变量：vi, ci_hat, T(zi), zr, P, si, r
    其中：
      vi = expected monthly income = (expected skill 或 skill) * 168
      ci_hat = 上月“真实消费/消费比例”（这里用上一月的 pc_i 近似）
      T(zi), zr 来自 obs_*.pkl 的 tax_paid / lump_sum
      P = world["Price"], r = world["Interest Rate"]（按月前后填充）
    """
    from pathlib import Path
    import re
    world   = dense["world"]
    actions = dense["actions"]
    states  = dense["states"]
    n_months = len(actions)

    # ---------- 1) 读 obs_*.pkl（税/再分配） ----------
    obs_files = sorted(Path(run_dir).glob("obs_*.pkl"),
                       key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    obs_list = []
    for m in range(n_months):
        if m < len(obs_files):
            with open(obs_files[m], "rb") as f:
                obs_list.append(pickle.load(f))   # 预期：{"0": {...}, ..., "p": {...}}
        else:
            obs_list.append({})

    # ---------- 2) 预先按“月”抽取宏观量（避免对 Series 直接 groupby('agent') 的错误用法） ----------
    price_by_m = []
    rate_by_m  = []
    for m in range(n_months):
        w = world[m + 1] if len(world) == n_months + 1 else world[m]
        # Price
        try:
            price_by_m.append(float(w.get("Price", np.nan)))
        except Exception:
            price_by_m.append(np.nan)
        # Interest Rate（有缺口就先放 NaN）
        rv = w.get("Interest Rate", np.nan)
        try:
            rate_by_m.append(float(rv))
        except Exception:
            rate_by_m.append(np.nan)

    # 用“月维度”做前后填充，再映射到 agent 行
    r_series = pd.Series(rate_by_m).replace(0, np.nan).ffill().bfill()
    p_series = pd.Series(price_by_m)  # 一般齐全，这里不强制填充

    # ---------- 3) 逐月逐 agent 生成行 ----------
    rows = []
    for m in range(n_months):
        a = actions[m]
        s = states[m + 1] if len(states) == n_months + 1 else states[m]
        obs_m = obs_list[m] if m < len(obs_list) else {}

        for aid in [k for k in a.keys() if k != "p"]:
            act = a.get(aid, {}) or {}
            st  = s.get(aid, {}) or {}

            # 决策：消费比例（0..1）、是否劳动
            pc_now = float(act.get("SimpleConsumption", 0)) * 0.02
            labor  = 1 if float(act.get("SimpleLabor", 0)) >= 1 else 0

            # vi：期望月收入 —— 用 expected skill 优先（更贴论文“预期收入”表述），否则退回 skill
            exp_skill = st.get("expected skill", st.get("skill", 0.0))
            try:
                vi = float(exp_skill) * 168.0
            except Exception:
                vi = float(st.get("skill", 0.0)) * 168.0

            # si：当前储蓄（Coin）
            inv = st.get("inventory", {}) or {}
            try:
                si = float(inv.get("Coin", 0.0))
            except Exception:
                si = 0.0

            # 税/再分配：来自 obs_m[aid]
            # --- 税/再分配（带代理方差） --- #
    
            

            oi   = obs_m.get(aid, {}) if isinstance(obs_m, dict) else {}
            T_zi = float(oi.get("PeriodicBracketTax-tax_paid", 0.0)) if isinstance(oi, dict) else 0.0
            zr   = float(oi.get("PeriodicBracketTax-lump_sum", 0.0))  if isinstance(oi, dict) else 0.0

            rows.append({
                "month": m + 1,
                "agent": int(aid),
                # 先存“原始”决策，后续再平滑/构造 ci_hat
                "labor": labor,
                "pc_now": pc_now,
                # 自变量
                "vi": vi,
                "P": float(p_series.iloc[m]),
                "si": si,
                "T_zi": T_zi,
                "zr": zr,
                "r": float(r_series.iloc[m]),
            })

    df = pd.DataFrame(rows).sort_values(["agent", "month"]).reset_index(drop=True)

    # ---------- 4) ci_hat：上一月消费比例（作为“上一月真实消费”的近似） ----------
    df["ci_hat"] = df.groupby("agent")["pc_now"].shift(1).fillna(0.0)

    # ---------- 5) 倾向定义（论文里的 pw_i / pc_i） ----------
    # pc_i 就是“当月消费倾向”（0..1）
    df["pc_i"] = df["pc_now"]

    # pw_i 用 labor(0/1) 的 EWMA 近似 伯努利倾向（让它随时间平滑，符合“propensity”直觉）
    # 用 pandas 内置 ewm，避免手写 apply 返回索引对不齐的问题
    df["pw_i"] = df.groupby("agent")["labor"] \
                   .transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())

    # 最终只留回归要用的字段 + month/agent 方便后续分析
    df = df[[
        "month", "agent",
        "pw_i", "pc_i",        # 目标
        "vi", "ci_hat", "T_zi", "zr", "P", "si", "r"   # 自变量
    ]]

    return df


# ========== 回归（表 1） ==========
# ========== 回归（表 1） ==========
def regression_table1(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 新增：构造宏观变量波动项 =====
    df["P_dev"] = df["P"] / df["P"].mean() - 1
    df["r_dev"] = (df["r"] - df["r"].mean()) * 100.0

    # 变量顺序与论文一致（但替换掉 P 和 r）
    Xvars = ["vi", "ci_hat", "T_zi", "zr", "P_dev", "si", "r_dev"]

    out_rows = []
    for aid, g in df.groupby("agent"):
        for yname, dlabel in [("pw_i", "pw_i"), ("pc_i", "pc_i")]:
            y = g[yname].values
            X = g[Xvars].values
            # Z-score 标准化
            X_mean = X.mean(axis=0)
            X_std  = X.std(axis=0) + 1e-8
            Xn = (X - X_mean) / X_std
            Xn = sm.add_constant(Xn)
            try:
                model = sm.OLS(y, Xn).fit()
                # 逐变量显著性统计
                for i, var in enumerate(Xvars):
                    pval = float(model.pvalues[i+1])  # 跳过常数项
                    out_rows.append({
                        "agent": aid,
                        "decision": dlabel,
                        "variable": var,
                        "significant": (pval <= 0.05),
                        "pval": pval
                    })
            except Exception:
                continue

    res = pd.DataFrame(out_rows)
    table = (res[res["significant"]]
             .groupby(["decision", "variable"])
             .size()
             .unstack(fill_value=0)
             .reindex(columns=Xvars, fill_value=0))

    print("\n[Table 1-like Summary]")
    print(table)
    print("\n(论文值参考：")
    print(" pw_i : vi=60, ci_hat=37, T(zi)=60, zr=65, P=58, si=56, r=31")
    print(" pc_i : vi=65, ci_hat=73, T(zi)=51, zr=52, P=62, si=100, r=49 )")
    return table


# ========== 图 5 ==========
def plot_figure5(df: pd.DataFrame, dense: dict, env, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) Consumption vs. Age
    # 初始年龄 + month/12
    init_age = {}
    if env is not None:
        for i in range(getattr(env, "n_agents", 100)):
            ag = env.get_agent(str(i))
            init_age[i] = ag.endogenous.get("age", 25)
    df["age"] = df.apply(lambda r: init_age.get(r["agent"], 25) + r["month"]/12.0, axis=1)

    # 论文图的 4 个年龄段
    bins = [25, 30, 35, 40, 45]
    labels = ["[25,30)", "[30,35)", "[35,40)", "[40,45)"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    age_mean = (df.groupby("age_group")["pc_i"].mean()
                  .reindex(labels))  # 保序

    ax1.plot(range(len(labels)), age_mean.values, marker="o", linewidth=2)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Consumption Proportion")
    ax1.set_title("(a) Consumption v.s. Age")
    ax1.grid(alpha=0.3)

    # (b) Consumption vs. Unemployment
    # 月度失业率（用 SimpleLabor==1 的比例）
    unemp = []
    for m, a in enumerate(dense["actions"]):
        ids = [k for k in a.keys() if k != "p"]
        employed = sum(1 for k in ids if (a.get(k, {}) or {}).get("SimpleLabor", 0) >= 1)
        rate = 1 - employed / len(ids) if ids else 0.0
        unemp.append({"month": m+1, "unemployment": rate})
    u = pd.DataFrame(unemp)
    u["year"] = (u["month"] - 1)//12 + 1
    year_avg = u.groupby("year")["unemployment"].mean()
    low_y, high_y = int(year_avg.idxmin()), int(year_avg.idxmax())

    low_months  = range((low_y-1)*12 + 1, low_y*12 + 1)
    high_months = range((high_y-1)*12 + 1, high_y*12 + 1)

    # 在 df 上聚合 p^c_i
    low_pc  = df[df["month"].isin(low_months)]["pc_i"]
    high_pc = df[df["month"].isin(high_months)]["pc_i"]
    low_mean, high_mean = float(low_pc.mean()), float(high_pc.mean())

    t_stat, pval = ttest_ind(high_pc, low_pc, equal_var=False, nan_policy="omit")

    bars = ax2.bar(["Lowest", "Highest"], [low_mean, high_mean],
                   color=["#4CAF50", "#F44336"], alpha=0.85, width=0.6)
    # 显著性星号
    if pval < 0.001:
        y = max(low_mean, high_mean)*1.05
        ax2.text(0.5, y, "***", ha="center", fontsize=18)

    ax2.set_ylim(0, max(low_mean, high_mean)*1.25)
    ax2.set_xlabel("Unemployment Rate")
    ax2.set_ylabel("Consumption Propensity")
    ax2.set_title("(b) Consumption v.s. Unemployment")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure 5 saved to {out_path} (p={pval:.4g})")

# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser("RQ3 (Decision-Making Abilities) – Table 1 + Figure 5")
    ap.add_argument("--run", required=True, help="仿真输出目录（含 dense_log_*.pkl / env_*.pkl / obs_*.pkl）")
    ap.add_argument("--out_fig", default="figs/figure5_rq3.png", help="Figure 5 输出路径")
    ap.add_argument("--out_table", default="figs/table1_decision_rationality.csv", help="Table 1 CSV 输出路径")
    args = ap.parse_args()

    dense = load_dense(args.run)
    env   = load_env_for_age(args.run)
    df    = extract_panel(dense, args.run)

    # 回归表 1
    table1 = regression_table1(df)
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    table1.to_csv(args.out_table, float_format="%.0f")
    print(f"[OK] Table 1 saved to {args.out_table}")

    # 图 5
    plot_figure5(df, dense, env, args.out_fig)

if __name__ == "__main__":
    main()
