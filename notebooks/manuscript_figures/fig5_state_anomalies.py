from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")  # 屏蔽所有烦人的底层库更新警告

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# =========================================================
# 0. 兼容性热修复 (Monkey Patch) 升级版
# 解决 Matplotlib 3.9+ 与 Seaborn 之间的 register_cmap 传参冲突
# =========================================================
import matplotlib.cm as cm

if not hasattr(cm, 'register_cmap'):
    def _register_cmap_wrapper(name, cmap):
        # 拦截 Seaborn 的旧版传参方式，翻译成新版 API 支持的格式
        # 新版中 cmap 是位置参数，name 是关键字参数，并加入 force=True 防止重名报错
        matplotlib.colormaps.register(cmap, name=name, force=True)


    cm.register_cmap = _register_cmap_wrapper

import seaborn as sns

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PANEL = FINAL_DIR / "analysis_panel_with_anomalies.csv"
OUT_FIG_PNG = FIG_DIR / "fig5_state_anomalies_distribution.png"
OUT_FIG_PDF = FIG_DIR / "fig5_state_anomalies_distribution.pdf"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]
STATE_ORDER = ["drought_like", "post_drought_rewetting", "non_drought_highflow", "normal"]

STATE_COLORS = {
    "drought_like": "#c77b7b",  # 暗红
    "post_drought_rewetting": "#5fa6a6",  # 青色
    "non_drought_highflow": "#7da6d8",  # 蓝色
    "normal": "#cfcfcf"  # 灰色
}

STATE_LABELS = {
    "drought_like": "Drought-like",
    "post_drought_rewetting": "Rewetting",
    "non_drought_highflow": "High-flow",
    "normal": "Normal"
}


# =========================================================
# 2. 辅助函数：安全读取
# =========================================================
def read_csv_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"无法读取文件 {path}，最后报错: {last_error}")


# =========================================================
# 3. 绘图主逻辑
# =========================================================
def plot_anomalies(df: pd.DataFrame):
    print("正在绘制浓度与通量异常多维分布图 (Fig 5)...")

    # 过滤掉无状态的行
    df = df[df["hydro_state"].isin(STATE_ORDER)].copy()
    df["hydro_state"] = pd.Categorical(df["hydro_state"], categories=STATE_ORDER, ordered=True)

    # 全局字体与样式设置
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set_theme(style="ticks", font="Times New Roman")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    metrics = ["CA", "FA"]
    metric_labels = {"CA": "Concentration Anomaly (CA)", "FA": "Flux Anomaly (FA)"}

    for row_idx, metric in enumerate(metrics):
        for col_idx, solute in enumerate(TARGET_SOLUTES):
            ax = axes[row_idx, col_idx]

            # 提取当前子集，剔除极值异常以保证制图美观 (保留 1% - 99% 的核心分布)
            sub_df = df[(df["solute"] == solute) & (df[metric].notna())].copy()
            if len(sub_df) == 0:
                continue

            p1, p99 = np.percentile(sub_df[metric], [1, 99])
            plot_df = sub_df[(sub_df[metric] >= p1) & (sub_df[metric] <= p99)]

            # 绘制小提琴图 (展示概率密度) + 箱线图 (展示四分位点)
            sns.violinplot(
                data=plot_df, x="hydro_state", y=metric,
                palette=STATE_COLORS, ax=ax, scale="width",
                inner="box", linewidth=1.2, alpha=0.8
            )

            # 添加零线基准 (Baseline)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=0)

            # 设置标题与标签
            if row_idx == 0:
                ax.set_title(f"{solute}", fontsize=14, fontweight="bold", pad=10)

            ax.set_ylabel(metric_labels[metric] if col_idx == 0 else "", fontsize=12)
            ax.set_xlabel("")

            # 格式化 x 轴刻度标签
            x_labels = [STATE_LABELS[cat.get_text()] for cat in ax.get_xticklabels()]
            ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=11)

            # 移除边框
            sns.despine(ax=ax)

    # 添加整体图例或说明
    plt.figtext(0.02, 0.96, "(a) Concentration Anomalies", fontsize=16, fontweight="bold")
    plt.figtext(0.02, 0.48, "(b) Flux Anomalies", fontsize=16, fontweight="bold")

    # 保存图表
    plt.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 异常分布图已生成: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


def main():
    if not INPUT_PANEL.exists():
        raise FileNotFoundError(f"找不到带有异常值的面板数据：{INPUT_PANEL}，请确认 03 脚本已成功运行。")

    df = read_csv_fallback(INPUT_PANEL)
    plot_anomalies(df)


if __name__ == "__main__":
    main()