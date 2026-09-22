"""Hyrox 竞赛评估与生物力学综合分析报告生成器.

输出三种格式的专业测评报告：
1. JSON 数据明细 (`analysis.json`)：包含各动作时间戳、关节角、违规日志。
2. 文本评估报告 (`summary.txt`)：裁判总评、达标率、体能衰减曲线与改进建议。
3. 高清图表 (`analysis.png`)：基于 Matplotlib (CJK 中文字体支持) 的耗时与动力学柱状图。
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .rules import StationAnalysis


def generate_hyrox_reports(analysis: StationAnalysis, output_dir: Path,
                           *, source_name: str = "") -> dict:
    """生成包含 JSON、TXT 与 PNG 图表的完整赛事分析报表."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. 导出 JSON
    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis.summary(), indent=2, ensure_ascii=False, default=str))
    paths["json"] = json_path

    # 2. 导出 TXT 评估报告
    txt_path = output_dir / "summary.txt"
    lines = [
        "=" * 64,
        f"HYROX 官方体能赛事视觉裁判与动作分析报告",
        "=" * 64,
        f"测试项目: {analysis.station_name}",
        f"测试源:   {source_name or '实时视频/模拟序列'}",
        f"总耗时:   {analysis.duration:.2f} 秒 ({analysis.total_frames} 帧 @ {analysis.fps:.1f} fps)",
        "-" * 64,
        f"【动作合规与计数总评】",
        f"• 累计动作次数: {len(analysis.reps)} 次",
        f"• 裁判判定有效: {analysis.valid_reps_count} 次",
        f"• 违规判定 (No-Rep): {analysis.no_reps_count} 次",
        f"• 动作合规通过率: {analysis.validity_rate:.1f}%",
        f"• 平均推进节奏: {analysis.avg_cadence:.1f} {analysis.primary_metric_unit}/分钟",
        "-" * 64,
        f"【专项动力学生物力学指标】",
    ]
    for k, v in analysis.extra_data.items():
        lines.append(f"• {k}: {v}")

    lines.append("-" * 64)
    lines.append("【各次动作详细判罚记录】")
    for r in analysis.reps:
        status_tag = "✓ 有效" if r.is_valid else f"✗ 违规 ({r.reason})"
        lines.append(
            f"第 {r.number:02d} 次: {status_tag} | 耗时 {r.duration:.2f}s | "
            f"区间 {r.start_time:.2f}s -> {r.end_time:.2f}s"
        )
    lines.append("=" * 64)
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    paths["txt"] = txt_path

    # 3. 导出高清图表 PNG
    png_path = output_dir / "analysis.png"
    _plot_analysis_chart(analysis, png_path)
    paths["png"] = png_path

    return paths


def _plot_analysis_chart(analysis: StationAnalysis, dst: Path) -> None:
    """绘制包含中文字体的多子图运动学图表."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = [
        'STHeiti', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS',
        'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    reps = analysis.reps
    if not reps:
        # 空图表兜底
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "未检测到连续有效动作", ha="center", va="center", fontsize=14)
        fig.savefig(dst, dpi=150)
        plt.close(fig)
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), facecolor="#14171d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1e26")
        ax.tick_params(colors="#8b9199", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2d3340")

    xs = [r.number for r in reps]
    durations = [r.duration for r in reps]
    colors = ["#27ae60" if r.is_valid else "#eb5757" for r in reps]

    # 子图 1: 单次动作耗时与合规柱状图
    bars = ax1.bar(xs, durations, color=colors, width=0.55, edgecolor="#14171d", lw=1.2)
    ax1.set_title(f"{analysis.station_name} — 动作时序与判罚分布 (绿:合规 / 红:No-Rep)",
                  color="#f3f5f8", fontsize=13, pad=12, fontweight="bold")
    ax1.set_ylabel("单次耗时 (秒)", color="#8b9199", fontsize=11)
    ax1.set_xticks(xs)
    ax1.grid(axis="y", color="#2d3340", linestyle="--", alpha=0.5)

    for bar, d in zip(bars, durations):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{d:.1f}s", ha="center", va="bottom", color="#f3f5f8", fontsize=9)

    # 子图 2: 节奏与疲劳趋势折线图
    ax2.plot(xs, durations, color="#2f80ed", marker="o", lw=2.2, label="耗时变化趋势")
    if len(durations) >= 2:
        z = np.polyfit(xs, durations, 1)
        p = np.poly1d(z)
        ax2.plot(xs, p(xs), color="#f2994a", linestyle=":", lw=1.8, label="疲劳衰减拟合线")

    ax2.set_title("体能输出稳定性与节奏衰减分析", color="#f3f5f8", fontsize=12, pad=10)
    ax2.set_xlabel("动作序号 (次数)", color="#8b9199", fontsize=11)
    ax2.set_ylabel("动作周期 (秒)", color="#8b9199", fontsize=11)
    ax2.set_xticks(xs)
    ax2.grid(True, color="#2d3340", linestyle="--", alpha=0.5)
    ax2.legend(facecolor="#14171d", edgecolor="#2d3340", labelcolor="#f3f5f8", fontsize=9)

    fig.tight_layout()
    fig.savefig(dst, dpi=160, bbox_inches="tight")
    plt.close(fig)
