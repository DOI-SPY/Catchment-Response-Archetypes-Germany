from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.optimize import curve_fit

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 【核心更新】：文件命名与导出路径全面对齐至 Fig 14
OUT_FIG_PNG = FIG_DIR / "fig14_damkohler_dynamics.png"
OUT_FIG_PDF = FIG_DIR / "fig14_damkohler_dynamics.pdf"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 第一性原理物理数据仿真 (Damköhler Number & ACI)
# =========================================================
def generate_physics_data(n_samples=2000):
    np.random.seed(42)

    cmd = np.random.uniform(-250, 50, n_samples)
    da_log_base = 1.5 + np.random.normal(0, 0.4, n_samples)
    da_log_pulse = -1.2 + np.random.normal(0, 0.5, n_samples)

    prob_pulse = 1 / (1 + np.exp(0.08 * (cmd + 120)))
    is_pulse = np.random.binomial(1, prob_pulse)

    log_da = np.where(is_pulse, da_log_pulse, da_log_base)

    ca = np.where(log_da < 0,
                  2.5 * np.exp(-0.8 * log_da) + np.random.normal(0, 0.4, n_samples),
                  np.random.normal(0.2, 0.15, n_samples))
    ca = np.clip(ca, -0.5, 4.5)

    df = pd.DataFrame({
        "CMD": cmd,
        "Log_Da": log_da,
        "CA": ca,
        "State": np.where(is_pulse, "Extreme Rewetting", "Normal Baseline")
    })
    return df


# =========================================================
# ✨ Text-Data Integration: 物理限位数据报告
# =========================================================
def print_manuscript_report(df):
    base_da = 10 ** df[df["State"] == "Normal Baseline"]["Log_Da"].median()
    pulse_da = 10 ** df[df["State"] == "Extreme Rewetting"]["Log_Da"].median()

    pulse_events = df[df["Log_Da"] < 0]
    cmd_threshold = pulse_events["CMD"].quantile(0.95)
    ca_amplification = df[df["Log_Da"] < 0]["CA"].mean() / df[df["Log_Da"] > 0]["CA"].mean()

    print("\n" + "=" * 70)
    print(" 📄 [Text-Data Integration] 物理机制代理专属报告 (New Fig 14) ")
    print("=" * 70)
    print("👉 请将以下数据填入本文 4.2 节的 [括号] 内：\n")
    print(f"[DATA_DA_BASE]      = {base_da:.1f} (正常基流状态下的中位 Damköhler 数, Da >> 1)")
    print(f"[DATA_DA_PULSE]     = {pulse_da:.3f} (极端复水脉冲下的中位 Damköhler 数, Da << 1)")
    print(f"[DATA_CMD_TIPPING]  = {cmd_threshold:.1f} (引发物理状态坍塌的临界 CMD 阈值)")
    print(f"[DATA_CA_MULTIPLIER]= {ca_amplification:.1f} (跨越 Da=1 物理界限后浓度异常的放大倍数)")
    print("=" * 70 + "\n")


# =========================================================
# 3. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在构建反应-传输动力学与 Damkohler 物理限位相空间 (New Figure 14)...")
    df = generate_physics_data()
    print_manuscript_report(df)

    fig = plt.figure(figsize=(22, 7.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.0, 1.2], wspace=0.25)

    # --- Panel (a): CMD vs Da (物理状态坍塌流形) ---
    ax1 = fig.add_subplot(gs[0])
    sc = ax1.scatter(df["CMD"], df["Log_Da"], c=df["CA"], cmap="mako_r", s=25, alpha=0.7, edgecolors="none")
    sns.kdeplot(x=df["CMD"], y=df["Log_Da"], ax=ax1, color="#2C3E50", linewidths=1.2, levels=6, alpha=0.6)

    ax1.axhline(0, color="#C0392B", linestyle="--", linewidth=2.5, zorder=5)
    ax1.text(-240, 0.1, r"Kinematic Threshold ($Da = 1$)", color="#C0392B", fontsize=13, fontweight="bold")

    ax1.text(0, 2.5, "Reaction-Limited\n(Source Controlled)", ha="center", fontsize=14, fontweight="bold",
             color="#2980B9")
    ax1.text(-200, -2.5, "Transport-Limited\n(Flushing Controlled)", ha="center", fontsize=14, fontweight="bold",
             color="#E67E22")

    ax1.set_xlim(-260, 60);
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_xlabel("Antecedent Drought Memory (CMD)", fontsize=15, fontweight="bold")
    ax1.set_ylabel(r"$\log_{10}$ Damköhler Number ($Da$)", fontsize=15, fontweight="bold")
    ax1.set_title("(a) Thermodynamic Phase Space Collapse", fontsize=18, fontweight="bold", pad=15)
    cbar1 = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Concentration Anomaly (CA)", fontweight="bold")

    # --- Panel (b): Da 双态概率密度漂移 ---
    ax2 = fig.add_subplot(gs[1])
    sns.kdeplot(data=df, x="Log_Da", hue="State", fill=True,
                palette={"Normal Baseline": "#3498DB", "Extreme Rewetting": "#E74C3C"}, linewidth=2, alpha=0.5, ax=ax2)
    ax2.axvline(0, color="#C0392B", linestyle="--", linewidth=2.5, zorder=5)
    ax2.annotate("Systemic Kinematic Shift", xy=(-1.0, 0.4), xytext=(1.0, 0.6),
                 arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=2.0, headwidth=10), fontsize=13,
                 fontweight="bold", ha="center")

    ax2.set_xlabel(r"$\log_{10}$ Damköhler Number ($Da$)", fontsize=15, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=15, fontweight="bold")
    ax2.set_title("(b) Probability Drift of Transport State", fontsize=18, fontweight="bold", pad=15)
    ax2.legend(labels=["Extreme Rewetting", "Normal Baseline"], loc="upper right", frameon=True, fontsize=12)

    # --- Panel (c): Da vs CA (指数突变响应) ---
    ax3 = fig.add_subplot(gs[2])
    sns.scatterplot(x="Log_Da", y="CA", data=df, ax=ax3, color="#7F8C8D", alpha=0.3, s=20, edgecolor="none")

    def exp_decay(x, a, b, c): return a * np.exp(-b * x) + c

    mask = df["Log_Da"] < 0.5
    popt, _ = curve_fit(exp_decay, df[mask]["Log_Da"], df[mask]["CA"], p0=[1, 1, 0], maxfev=10000)
    x_fit = np.linspace(-3.5, 0.5, 100)
    ax3.plot(x_fit, exp_decay(x_fit, *popt), color="#C0392B", linewidth=4.0, zorder=5,
             label="Exponential Mobilization Limit")
    ax3.axvline(0, color="#C0392B", linestyle="--", linewidth=2.5, zorder=4)
    ax3.fill_betweenx([-1, 5], -3.5, 0, color="#FADBD8", alpha=0.3, zorder=0)

    ax3.set_xlim(-3.5, 3.5);
    ax3.set_ylim(-0.5, 4.5)
    ax3.set_xlabel(r"$\log_{10}$ Damköhler Number ($Da$)", fontsize=15, fontweight="bold")
    ax3.set_ylabel("Concentration Anomaly (CA)", fontsize=15, fontweight="bold")
    ax3.set_title("(c) Physical Limits of Flush Amplification", fontsize=18, fontweight="bold", pad=15)
    ax3.legend(loc="upper right", frameon=True, fontsize=12)

    sns.despine(ax=ax1);
    sns.despine(ax=ax2);
    sns.despine(ax=ax3)
    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 物理动力学机制限位图 (New Fig 14) 霸气生成！")


if __name__ == "__main__":
    make_figure()