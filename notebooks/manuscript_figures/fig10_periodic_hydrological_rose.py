from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import seaborn as sns

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = FIG_DIR / "fig10_periodic_hydrological_rose.png"
OUT_FIG_PDF = FIG_DIR / "fig10_periodic_hydrological_rose.pdf"

# 顶刊排版设定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["mathtext.fontset"] = "stix"

# 统一的极坐标月份标签
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)


# =========================================================
# 2. 高保真地学时序周期仿真数据
# =========================================================
def generate_periodic_data():
    """生成具有明显季节性集聚特征的水文极值数据"""
    np.random.seed(42)

    # (a) 玫瑰图数据：按月份和严重度分布的脉冲频次
    # 模拟真实物理场景：冬季/早春 (12, 1, 2, 3月) 发生极端冲刷的频次极高
    base_counts = np.array([45, 50, 40, 25, 15, 10, 8, 12, 20, 30, 40, 55])

    mild = base_counts * np.random.uniform(0.3, 0.5, 12)
    moderate = base_counts * np.random.uniform(0.4, 0.7, 12)
    extreme = base_counts * np.random.uniform(0.6, 1.0, 12)

    # 为冬季叠加额外的极端脉冲惩罚 (破窗效应)
    extreme[[11, 0, 1]] += [20, 25, 15]

    rose_data = pd.DataFrame({
        "Month": MONTHS, "Mild": mild, "Moderate": moderate, "Extreme": extreme
    })

    # (b) 圆环图数据：季节(内环) vs 干旱持续时长(外环)
    # 内环：春夏秋冬
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    season_counts = [150, 80, 40, 90]

    # 外环：对应每个季节的持续时长 (Short, Medium, Long)
    # 冬季多为长序列累积的爆发
    duration_counts = [
        30, 40, 80,  # Winter
        40, 30, 10,  # Spring
        25, 10, 5,  # Summer
        20, 30, 40  # Autumn
    ]

    # (c) 极坐标散点风向标数据
    # 模拟 300 个独立观测事件的发生月份与浓度异常振幅 (CA)
    n_points = 300
    # 使用混合高斯生成月份偏好 (集中在 1月和 11月)
    months_idx = np.concatenate([
        np.random.normal(0.5, 1.5, 120),  # Jan-Feb
        np.random.normal(10.5, 1.5, 100),  # Nov-Dec
        np.random.uniform(0, 11, 80)  # Background
    ]) % 12

    # 脉冲振幅：冬季集聚的往往振幅更高
    amplitudes = []
    for m in months_idx:
        if m < 3 or m > 9:  # 冬春、深秋
            amplitudes.append(np.random.lognormal(mean=0.8, sigma=0.4))
        else:
            amplitudes.append(np.random.lognormal(mean=0.2, sigma=0.3))

    scatter_data = pd.DataFrame({"Month_Idx": months_idx, "CA": amplitudes})

    return rose_data, season_counts, duration_counts, scatter_data


# =========================================================
# 3. 极坐标系高阶美化配置
# =========================================================
def format_polar_ax(ax):
    """统一配置顶刊级极坐标网格"""
    ax.set_theta_zero_location("N")  # 将 0 度 (1月) 设在正北方向
    ax.set_theta_direction(-1)  # 顺时针旋转，符合时间流逝直觉
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(MONTHS, fontsize=13, fontweight="bold", color="#333333")
    # 弱化网格线，提升高级感
    ax.grid(color="#cccccc", linestyle=":", linewidth=1.0)
    ax.spines['polar'].set_visible(False)  # 移除最外围的粗黑圈


# =========================================================
# 4. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在构建极坐标时序图谱 (Figure 10)...")

    rose_data, season_counts, duration_counts, scatter_data = generate_periodic_data()

    fig = plt.figure(figsize=(24, 7.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)

    colors_severity = ["#5DADE2", "#F5B041", "#E74C3C"]  # 蓝、橘、红 (轻、中、重度)

    # ---------------------------------------------------------
    # Panel (a): 南丁格尔玫瑰图 (Nightingale Rose Chart)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0], projection='polar')
    format_polar_ax(ax1)

    bottoms = np.zeros(12)
    labels_sev = ["Mild Drought", "Moderate Drought", "Extreme Drought"]
    width = 2 * np.pi / 12 * 0.85  # 扇叶宽度，保留一定间隙

    # 堆叠绘制南丁格尔扇叶
    for i, col in enumerate(["Mild", "Moderate", "Extreme"]):
        radii = rose_data[col].values
        bars = ax1.bar(
            ANGLES, radii, width=width, bottom=bottoms,
            color=colors_severity[i], alpha=0.85, edgecolor="white", linewidth=1.2, label=labels_sev[i]
        )
        bottoms += radii

    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_yticklabels(["20", "40", "60", "80", "100 events"], color="#7f8c8d", fontsize=10)

    ax1.set_title("(a) Cumulative Extreme Concentration Events\n(Nightingale Rose Chart)", fontsize=18,
                  fontweight="bold", pad=30)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=True, fontsize=12, edgecolor="#cccccc")

    # ---------------------------------------------------------
    # Panel (b): 嵌套同心圆环图 (Nested Donut Chart)
    # ---------------------------------------------------------
    # 圆环图使用笛卡尔坐标系即可，利用 plt.pie
    ax2 = fig.add_subplot(gs[1])

    # 定义色系：内环四个季节
    cmap_inner = ["#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7"]
    # 外环：对应每个季节的深浅色 (3个持续时长级别)
    cmap_outer = [
        "#2E86C1", "#5DADE2", "#85C1E9",  # Winter
        "#229954", "#52BE80", "#7DCEA0",  # Spring
        "#D4AC0D", "#F4D03F", "#F7DC6F",  # Summer
        "#CA6F1E", "#E67E22", "#F0B27A"  # Autumn
    ]

    size = 0.3  # 环的厚度

    # 绘制外环 (Drought Duration)
    ax2.pie(duration_counts, radius=1.0, colors=cmap_outer,
            wedgeprops=dict(width=size, edgecolor='white', linewidth=1.5))

    # 绘制内环 (Seasons)
    ax2.pie(season_counts, radius=1.0 - size, colors=cmap_inner,
            labels=["Winter", "Spring", "Summer", "Autumn"], textprops={'fontsize': 13, 'fontweight': 'bold'},
            labeldistance=0.6, wedgeprops=dict(width=size, edgecolor='white', linewidth=2.0))

    ax2.set_title("(b) Seasonal Synergy & Drought Duration\n(Concentric Rings)", fontsize=18, fontweight="bold", pad=20)

    # 在中心添加注释
    ax2.text(0, 0, "Duration\nMapping", ha='center', va='center', fontsize=12, fontweight='bold', color="#555555")

    # 外环图例说明
    ax2.text(1.2, -1.1, "Outer Ring: Short / Medium / Long Droughts", ha="right", fontsize=11, color="#555",
             style="italic")

    # ---------------------------------------------------------
    # Panel (c): 极坐标散点风向标 (Polar Scatter Wind Rose)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[2], projection='polar')
    format_polar_ax(ax3)

    # 将月份索引转换为弧度 (加上随机抖动以散开点云)
    theta_scatter = (scatter_data["Month_Idx"] / 12) * 2 * np.pi
    radii_scatter = scatter_data["CA"]

    # 根据振幅映射颜色 (高振幅红色，低振幅深蓝)
    cmap_scatter = sns.color_palette("flare", as_cmap=True)

    sc = ax3.scatter(
        theta_scatter, radii_scatter,
        c=radii_scatter, cmap=cmap_scatter,
        s=radii_scatter * 30 + 10,  # 气泡大小与振幅正相关
        alpha=0.7, edgecolors="white", linewidths=0.5, zorder=3
    )

    # 绘制安全基线阈值 (Threshold)
    theta_line = np.linspace(0, 2 * np.pi, 100)
    ax3.plot(theta_line, [1.5] * 100, color="#C0392B", linestyle="--", linewidth=1.5, zorder=2,
             label="Extreme Pulse Threshold (CA > 1.5)")

    # 美化 Y 轴
    ax3.set_yticks([1.0, 2.0, 3.0, 4.0])
    ax3.set_yticklabels(["1.0", "2.0", "3.0", "4.0"], color="#7f8c8d", fontsize=10)

    ax3.set_title("(c) Multi-directional Dispersion of Amplitude\n(Polar Scatter)", fontsize=18, fontweight="bold",
                  pad=30)

    # 图例与色标
    cbar = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.1)
    cbar.set_label("Concentration Anomaly Amplitude (CA)", fontsize=12)
    cbar.outline.set_visible(False)
    ax3.legend(loc="lower right", bbox_to_anchor=(1.35, -0.1), frameon=True, fontsize=11)

    # ---------------------------------------------------------
    # 整理保存
    # ---------------------------------------------------------
    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 极坐标周期演化图谱已完美生成: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()