"""Hyrox 赛事级专业暗黑风格中文数据看板 (Side Panel) 渲染器.

呈现内容：
1. 顶部：Hyrox 赛事站点标题与计时。
2. 核心卡片：当前累计有效次数 (Valid Reps)、合规率 (Pass %)、平均频率 (Pace/SPM)。
3. 实时裁判警示条：合规绿色通告 vs 违规红色高亮 (No-Rep Warning Banner)。
4. 历史动作瀑布柱状图：每次动作耗时展示（合规绿柱 / 违规红柱），动态揭示体能衰减。
5. 站点专有动力学度量（如深蹲屈角、跳跃间距、躯干驱动角、冲程幅度）。
"""

from __future__ import annotations

import cv2
import numpy as np
from .rules import StationAnalysis, ViolationType
from .text import draw, measure, resolve_font


def _rgb(color_hex: str) -> tuple[int, int, int]:
    return tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))


class HyroxPanelRenderer:
    """Hyrox 实时数据看板渲染引擎."""

    def __init__(self, analysis: StationAnalysis, width: int, height: int, *, cfg):
        self.analysis = analysis
        self.width = width
        self.height = height
        self.cfg = cfg
        self.font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)

    def render_frame_panel(self, frame_idx: int) -> np.ndarray:
        """根据当前视频帧索引生成右侧数据看板图像 (BGR, height x width x 3)."""
        w, h = self.width, self.height
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        # 背景底色
        bg_rgb = _rgb(self.cfg.PANEL_BG)
        panel[:] = bg_rgb[::-1]

        t = frame_idx / self.analysis.fps if self.analysis.fps else 0.0

        # 当前已完成的动作
        reps_done = [r for r in self.analysis.reps if r.end_frame <= frame_idx]
        current_rep = next((r for r in self.analysis.reps
                           if r.start_frame <= frame_idx <= r.end_frame), None)

        valid_count = sum(1 for r in reps_done if r.is_valid)
        total_count = len(reps_done)
        pass_pct = (valid_count / total_count * 100.0) if total_count > 0 else 100.0

        margin_x = int(w * 0.06)
        y = int(h * 0.04)

        # ── 1. 顶部站点标题与时间 ─────────────────────────────────────────────
        draw(panel, "HYROX 官方体能赛事视觉裁判系统", self.font, size=15,
             xy=(margin_x, y), color=_rgb(self.cfg.PANEL_MUTED))
        y += 24
        draw(panel, self.analysis.station_name, self.font, size=21,
             xy=(margin_x, y), color=_rgb(self.cfg.PANEL_FG))
        y += 36

        # 分割线
        cv2.line(panel, (margin_x, y), (w - margin_x, y), (40, 48, 60), 1, cv2.LINE_AA)
        y += 18

        # ── 2. 核心大数字数据卡片 ─────────────────────────────────────────────
        card_w = (w - margin_x * 2 - 16) // 2
        card_h = int(h * 0.14)

        # 卡片 1: 有效计数
        cv2.rectangle(panel, (margin_x, y), (margin_x + card_w, y + card_h),
                      (24, 30, 40), -1)
        cv2.rectangle(panel, (margin_x, y), (margin_x + card_w, y + card_h),
                      (45, 55, 72), 1, cv2.LINE_AA)
        draw(panel, self.analysis.primary_metric_name, self.font, size=14,
             xy=(margin_x + 16, y + 16), color=_rgb(self.cfg.PANEL_MUTED))
        draw(panel, f"{valid_count}", self.font, size=46,
             xy=(margin_x + 16, y + 42), color=_rgb(self.cfg.PANEL_SUCCESS))

        # 卡片 2: 合规率 / 桨频
        cv2.rectangle(panel, (margin_x + card_w + 16, y), (w - margin_x, y + card_h),
                      (24, 30, 40), -1)
        cv2.rectangle(panel, (margin_x + card_w + 16, y), (w - margin_x, y + card_h),
                      (45, 55, 72), 1, cv2.LINE_AA)

        if "SPM" in self.analysis.primary_metric_unit:
            sub_title = "实时推进频率"
            sub_val = f"{self.analysis.avg_cadence:.0f}"
            unit = "SPM"
        else:
            sub_title = "动作合规通过率"
            sub_val = f"{pass_pct:.0f}%"
            unit = f"无效: {total_count - valid_count}"

        draw(panel, sub_title, self.font, size=14,
             xy=(margin_x + card_w + 32, y + 16), color=_rgb(self.cfg.PANEL_MUTED))
        draw(panel, sub_val, self.font, size=42,
             xy=(margin_x + card_w + 32, y + 44), color=_rgb(self.cfg.PANEL_ACCENT))
        draw(panel, unit, self.font, size=13,
             xy=(margin_x + card_w + 32, y + 96), color=_rgb(self.cfg.PANEL_MUTED))

        y += card_h + 18

        # ── 3. 实时裁判警示条 ─────────────────────────────────────────────────
        banner_h = int(h * 0.08)
        if current_rep and not current_rep.is_valid:
            # 违规醒目红底报警条
            cv2.rectangle(panel, (margin_x, y), (w - margin_x, y + banner_h),
                          (30, 30, 160), -1)
            cv2.rectangle(panel, (margin_x, y), (w - margin_x, y + banner_h),
                          (60, 60, 230), 2, cv2.LINE_AA)
            draw(panel, "[警示] 裁判判定: NO-REP 违规", self.font, size=16,
                 xy=(margin_x + 16, y + 12), color=(255, 255, 255))
            draw(panel, current_rep.reason, self.font, size=14,
                 xy=(margin_x + 16, y + 36), color=(255, 220, 220))
        elif current_rep and current_rep.is_valid:
            # 动作进行中 - 绿底有效提示条
            cv2.rectangle(panel, (margin_x, y), (w - margin_x, y + banner_h),
                          (20, 80, 30), -1)
            cv2.rectangle(panel, (margin_x, y), (w - margin_x, y + banner_h),
                          (40, 160, 60), 1, cv2.LINE_AA)
            draw(panel, f"[合规] 正在进行第 {current_rep.number} 次动作", self.font, size=15,
                 xy=(margin_x + 16, y + 14), color=(255, 255, 255))
            draw(panel, "姿态动力学合规中 · 准备计数", self.font, size=13,
                 xy=(margin_x + 16, y + 38), color=(200, 255, 210))
        else:
            # 巡航待机条
            cv2.rectangle(panel, (margin_x, y), (w - margin_x, y + banner_h),
                          (22, 28, 36), -1)
            draw(panel, "[巡检] 实时动作捕捉巡检中", self.font, size=14,
                 xy=(margin_x + 16, y + 16), color=_rgb(self.cfg.PANEL_MUTED))
            draw(panel, f"比赛计时: {t:.1f}s · 帧数: {frame_idx}", self.font, size=13,
                 xy=(margin_x + 16, y + 38), color=(140, 150, 165))

        y += banner_h + 24

        # ── 4. 历史动作时序瀑布柱状图 ─────────────────────────────────────────
        draw(panel, "动作时序与疲劳耗时分布 (秒)", self.font, size=16,
             xy=(margin_x, y), color=_rgb(self.cfg.PANEL_FG))
        y += 26

        graph_h = int(h * 0.28)
        graph_w = w - margin_x * 2
        # 画布底板
        cv2.rectangle(panel, (margin_x, y), (margin_x + graph_w, y + graph_h),
                      (18, 22, 28), -1)
        cv2.rectangle(panel, (margin_x, y), (margin_x + graph_w, y + graph_h),
                      (36, 44, 56), 1, cv2.LINE_AA)

        if reps_done:
            durations = [r.duration for r in reps_done]
            max_d = max(max(durations), 3.0)
            n_bars = len(reps_done)
            bar_slot = graph_w / max(1, n_bars)
            bar_w = max(4, int(bar_slot * 0.65))

            for i, r in enumerate(reps_done):
                bx = int(margin_x + i * bar_slot + (bar_slot - bar_w) / 2)
                bh = int((r.duration / max_d) * (graph_h - 40))
                by = y + graph_h - bh - 24

                # 合规绿柱，违规红柱
                b_color = _rgb(self.cfg.PANEL_SUCCESS)[::-1] if r.is_valid \
                    else _rgb(self.cfg.PANEL_DANGER)[::-1]
                cv2.rectangle(panel, (bx, by), (bx + bar_w, y + graph_h - 24),
                              b_color, -1)

                # 标注序号与耗时
                if n_bars <= 15:
                    draw(panel, f"{r.number}", self.font, size=11,
                         xy=(bx + bar_w // 2, y + graph_h - 20),
                         color=_rgb(self.cfg.PANEL_MUTED), anchor="ct")
                    draw(panel, f"{r.duration:.1f}", self.font, size=10,
                         xy=(bx + bar_w // 2, by - 14),
                         color=_rgb(self.cfg.PANEL_FG), anchor="ct")
        else:
            draw(panel, "动作序列建立中...", self.font, size=14,
                 xy=(margin_x + graph_w // 2, y + graph_h // 2),
                 color=_rgb(self.cfg.PANEL_MUTED), anchor="cm")

        y += graph_h + 20

        # ── 5. 站点专有动力学拓展指标 ─────────────────────────────────────────
        draw(panel, "专项生物力学关键指标", self.font, size=16,
             xy=(margin_x, y), color=_rgb(self.cfg.PANEL_FG))
        y += 26

        for k, v in self.analysis.extra_data.items():
            draw(panel, f"• {k}: {v}", self.font, size=13,
                 xy=(margin_x + 8, y), color=_rgb(self.cfg.PANEL_MUTED))
            y += 20

        # 底部官方署名
        draw(panel, "HYROX VISION · 运动视觉智能引擎", self.font, size=12,
             xy=(w - margin_x, h - 20), color=(80, 90, 105), anchor="rb")

        return panel
