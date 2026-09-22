"""Hyrox 视频帧合成与数据看板并置渲染器.

将左侧视频画面（骨架叠加、关节角度高亮、违规标框）与
右侧 Hyrox 实时专业侧边看板 (Side Panel) 无缝拼接并编码输出。
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from rich.console import Console

from .panel import HyroxPanelRenderer
from .rules import StationAnalysis, ViolationType
from .skeleton import draw_skeleton
from .text import draw, resolve_font


def render_hyrox_video(video_path: Path | None, frames: list[dict],
                       analysis: StationAnalysis, output_path: Path,
                       *, cfg, console: Console | None = None) -> dict:
    """合成完整带有骨骼姿态、动作标记与右侧看板的视频文件."""
    n_frames = len(frames)
    fps = analysis.fps or 30.0

    # 确定输入源尺寸
    cap = None
    if video_path and video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # 无物理源视频时，生成高品质赛场暗黑底布
        src_w, src_h = 1080, 1080

    target_h = cfg.EXPORT_HEIGHT or src_h
    # 保持原视频宽高比缩放
    video_w = int(src_w * (target_h / src_h))
    panel_w = int(target_h * 0.68)  # 侧边面板宽度
    total_w = video_w + panel_w
    total_h = target_h

    # 初始化看板渲染器
    panel_renderer = HyroxPanelRenderer(analysis, panel_w, total_h, cfg=cfg)
    font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    font_scale = 0.6
    frames_written = 0

    for idx in range(n_frames):
        # 1. 获取原画帧
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[:2] != (total_h, video_w):
                frame = cv2.resize(frame, (video_w, total_h))
        else:
            # 合成赛场网格底色
            frame = np.zeros((total_h, video_w, 3), dtype=np.uint8)
            frame[:] = (20, 24, 30)
            # 弱网格线
            for gx in range(0, video_w, 80):
                cv2.line(frame, (gx, 0), (gx, total_h), (28, 33, 40), 1)
            for gy in range(0, total_h, 80):
                cv2.line(frame, (0, gy), (video_w, gy), (28, 33, 40), 1)

        # 2. 绘制姿态骨架
        frame_data = frames[idx] if idx < len(frames) else {}
        persons = frame_data.get("persons") or []

        # 检查当前帧是否在某个违规动作区间中
        current_rep = next((r for r in analysis.reps
                           if r.start_frame <= idx <= r.end_frame), None)

        highlight_color = None
        if current_rep:
            if not current_rep.is_valid:
                highlight_color = (40, 40, 240)   # 红色高亮违规
            else:
                highlight_color = (80, 220, 80)   # 绿色高配合规

        if persons and cfg.DRAW_SKELETON:
            draw_skeleton(frame, persons[0], video_w, total_h,
                          draw_face=cfg.DRAW_FACE, thickness=cfg.LINE_THICKNESS,
                          radius=cfg.POINT_RADIUS, highlight_color=highlight_color)

        # 3. 视频画面左上角 HUD
        cv2.rectangle(frame, (16, 16), (280, 68), (10, 12, 16), -1)
        cv2.rectangle(frame, (16, 16), (280, 68), (40, 50, 65), 1)
        draw(frame, f"HYROX 实时姿态分析", font, size=14,
             xy=(28, 24), color=(180, 195, 210))
        t_now = idx / fps
        draw(frame, f"耗时: {t_now:.2f}s  |  帧: {idx}/{n_frames}", font, size=12,
             xy=(28, 46), color=(120, 135, 150))

        # 4. 若为 No-Rep，在画面中央顶部显示醒目横幅
        if current_rep and not current_rep.is_valid:
            box_w = int(video_w * 0.70)
            box_x = (video_w - box_w) // 2
            cv2.rectangle(frame, (box_x, 80), (box_x + box_w, 140), (20, 20, 180), -1)
            cv2.rectangle(frame, (box_x, 80), (box_x + box_w, 140), (50, 50, 255), 2)
            draw(frame, f"[违规] {current_rep.reason}", font, size=17,
                 xy=(box_x + box_w // 2, 95), color=(255, 255, 255), anchor="ct")

        # 5. 生成右侧看板并拼接
        panel_img = panel_renderer.render_frame_panel(idx)

        # 水平合并
        composite = np.hstack([frame, panel_img])
        writer.write(composite)
        frames_written += 1

    if cap:
        cap.release()
    writer.release()

    return {
        "frames_written": frames_written,
        "width": total_w,
        "height": total_h,
        "fps": fps,
    }
