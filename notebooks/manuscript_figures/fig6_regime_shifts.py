from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")  # 屏蔽所有烦人的底层库更新警告

from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.contour as mcontour

# =========================================================
# 0. 兼容性热修复 (Monkey Patch) 终极版
# 解决 Matplotlib 3.8/3.9+ 与旧版 Seaborn 之间的 API 世代冲突
# =========================================================
import matplotlib.cm as cm

# 补丁 1: 修复 register_cmap 传参冲突
if not hasattr(cm, 'register_cmap'):
    def _register_cmap_wrapper(name, cmap):
        matplotlib.colormaps.register(cmap, name=name, force=True)


    cm.register_cmap = _register_cmap_wrapper

# 补丁 2: 修复 KDEPlot 中 QuadContourSet 缺失 collections 属性的崩溃
if not hasattr(mcontour.QuadContourSet, 'collections'):
    # 当 Seaborn 呼叫 .collections[0] 时，把 self 包装在列表里返还，完美兼容旧逻辑
    mcontour.QuadContourSet.collections = property(lambda self: [self])

import seaborn as sns

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SLOPES = FINAL_DIR / "catchment_cq_slopes_matrix.csv"
OUT_FIG_PNG = FIG_DIR / "fig6_regime_shifts.png"
OUT_FIG_PDF = FIG_DIR / "fig6_regime_shifts.pdf"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]
COLORS = {"NO3N": "#2f7f73", "PO4P": "#c48a3a", "DOC": "#7b5ea7"}


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
def plot_regime_shifts(df: pd.DataFrame):
    print("正在绘制流域水质体制跃迁图 (Fig 6)...")

    # 全局字体与样式设置
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set_theme(style="ticks", font="Times New Roman")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.25)

    # 核心目标变量
    x_var = "delta_beta_post_drought_rewetting"
    y_var = "delta_cv_ratio_post_drought_rewetting"

    for ax, solute in zip(axes, TARGET_SOLUTES):
        # 提取并清理当前溶质的有效数据
        sub_df = df[(df["solute"] == solute)].copy()
        plot_df = sub_df.dropna(subset=[x_var, y_var])

        if len(plot_df) == 0:
            continue

        # 1. 绘制带有透明度的基础散点，反映大样本点云分布
        sns.scatterplot(
            data=plot_df, x=x_var, y=y_var,
            ax=ax, color=COLORS[solute], alpha=0.4, s=40, edgecolor="white", linewidth=0.5
        )

        # 2. 叠加核密度等高线 (KDE Contour) 展现群聚核心 (重心所在)
        sns.kdeplot(
            data=plot_df, x=x_var, y=y_var,
            ax=ax, color=COLORS[solute], linewidths=1.5, alpha=0.8, levels=5
        )

        # 3. 绘制学术十字准星 (Baseline = 0)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2, zorder=0)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, zorder=0)

        # 4. 图表修饰
        ax.set_title(f"{solute} Regime Shifts", fontsize=15, fontweight="bold", pad=15)
        ax.set_xlabel(r"Shift in c-Q Slope ($\Delta\beta$)", fontsize=13)
        if solute == "NO3N":
            ax.set_ylabel(r"Shift in Chemostatic Index ($\Delta CV_c / CV_q$)", fontsize=13)
        else:
            ax.set_ylabel("")

        # 5. 在四个象限打上机理解释标签 (顶刊范式，直接帮助审稿人理解图表物理意义)
        # 第一象限 (右上)
        ax.text(0.95, 0.95, "Flushing &\nSource Depleted", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color="#555", style="italic")
        # 第二象限 (左上)
        ax.text(0.05, 0.95, "Dilution &\nSource Depleted", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, color="#555", style="italic")
        # 第三象限 (左下)
        ax.text(0.05, 0.05, "Dilution &\nTransport Limited", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=11, color="#555", style="italic")
        # 第四象限 (右下)
        ax.text(0.95, 0.05, "Flushing &\nTransport Limited", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11, color="#555", style="italic")

        # 动态自适应坐标轴范围，居中十字线
        x_max = max(abs(plot_df[x_var].min()), abs(plot_df[x_var].max())) * 1.1
        y_max = max(abs(plot_df[y_var].min()), abs(plot_df[y_var].max())) * 1.1
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)

        sns.despine(ax=ax)

    # 整体保存
    plt.tight_layout()
    plt.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 体制跃迁图已生成: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


def main():
    if not INPUT_SLOPES.exists():
        raise FileNotFoundError(f"找不到 c-Q 斜率特征矩阵：{INPUT_SLOPES}")

    df = read_csv_fallback(INPUT_SLOPES)
    plot_regime_shifts(df)


if __name__ == "__main__":
    main()