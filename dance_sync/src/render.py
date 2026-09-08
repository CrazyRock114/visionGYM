"""Drawing the returned poses back onto the frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .panel import SimilarityPanel
from .skeleton import KPT_NAMES, draw_hud, draw_person, visible_parts
from .text import resolve_font
from .video import VideoInfo


def render(
    info: VideoInfo,
    by_index: dict[int, list[dict]],
    n_returned: int,
    dst: Path,
    *,
    cfg,
    console,
    similarity=None,
    picture_times=(),
) -> dict:
    """Write an overlay video and return per-run stats.

    When the clip was decimated (more source frames than the model posed), the
    frames in between have no result of their own. Holding the last pose across
    the gap beats flickering, and the overlay marks those frames `(held)` so a
    measured frame is never confused with a carried one.
    """
    edges, points = visible_parts(cfg.DRAW_FACE)

    stride = max(1, round(info.n_frames / max(n_returned, 1)))
    hold_limit = stride - 1

    # Same height, double the width: the clip keeps its native aspect ratio and
    # the panel takes an equal share beside it.
    panel = None
    out_w = info.width
    font_name = None
    if cfg.SIDE_PANEL and similarity is not None:
        font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        font_name = font[2] if font else "OpenCV Hershey"
        panel = SimilarityPanel(similarity, info.width, info.height, cfg=cfg, font=font,
                                picture_times=picture_times)
        out_w = info.width * 2

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (out_w, info.height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    last_persons: list[dict] | None = None
    held_for = 0
    n_drawn = n_held = n_people = 0
    max_people = 0

    columns = [
        TextColumn("[cyan]rendering[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
    ]
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task("render", total=info.n_frames)
            for i in range(info.n_frames):
                ok, frame = cap.read()
                if not ok:
                    break

                if i in by_index:
                    persons, held = by_index[i], False
                    last_persons, held_for = persons, 0
                elif last_persons is not None and held_for < hold_limit:
                    persons, held = last_persons, True
                    held_for += 1
                else:
                    persons, held = [], False

                h, w = frame.shape[:2]
                for person in persons:
                    draw_person(
                        frame, person, w, h,
                        edges=edges, points=points,
                        thickness=cfg.LINE_THICKNESS, radius=cfg.POINT_RADIUS,
                        held=held, draw_bbox=cfg.DRAW_BBOX, draw_label=cfg.DRAW_TRACK_LABEL,
                        bbox_color=cfg.BBOX_COLOR, color_by=cfg.COLOR_BY,
                    )

                if persons:
                    n_drawn += 1
                    n_people += len(persons)
                    n_held += bool(held)
                    max_people = max(max_people, len(persons))

                if cfg.DRAW_HUD:
                    hud = f"frame {i}/{info.n_frames}  |  {len(persons)} person(s)"
                    draw_hud(frame, hud + ("  (held)" if held else ""))

                if panel is not None:
                    frame = np.hstack([frame, panel.draw(i)])

                writer.write(frame)
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()

    return {
        "frames_written": info.n_frames,
        "frames_with_pose": n_drawn,
        "frames_held": n_held,
        "mean_people_per_posed_frame": round(n_people / n_drawn, 2) if n_drawn else 0.0,
        "max_people_in_a_frame": max_people,
        "hold_limit": hold_limit,
        "output_width": out_w,
        "side_panel": panel is not None,
        "panel_font": font_name,
    }


def plot_joint(frames: list[dict], fps: float, joint: str, dst: Path) -> bool:
    """Plot one joint's height per track. Returns False if it never appeared.

    `track_id` is what makes this meaningful: without it `persons[0]` is a list
    slot, not a person, and with three dancers on screen the series would jump
    between people every time the detection order changed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = KPT_NAMES.index(joint)

    tracks: dict = {}
    for f in frames:
        for p in f["persons"]:
            kpts = p.get("kpts_xy", [])
            if j < len(kpts) and tuple(kpts[j]) != (0, 0):
                # y grows downward in image coordinates; flip so "up" reads as up.
                tracks.setdefault(p.get("track_id"), []).append(
                    (f["index"] / fps, 1.0 - kpts[j][1])
                )

    if not tracks:
        return False

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for tid, series in sorted(tracks.items(), key=lambda kv: (kv[0] is None, kv[0])):
        ts, ys = zip(*series)
        ax.plot(ts, ys, lw=1.5, label=f"track {tid}")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{joint} height (normalized, up = higher)")
    ax.set_title(f"{joint} over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    return True
