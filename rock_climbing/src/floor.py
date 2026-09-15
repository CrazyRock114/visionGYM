"""The ground, so the clock can start when the climber leaves it.

A boulder starts when both feet are off the floor — not when both feet are on
holds. Those are different rules and only the first one is the gym's: a foot
smeared flat against the wall, on no hold at all, is a legitimate placement and
the climb has still begun.

Finding the floor is the same problem as finding the holds, so it is the same
model with a different prompt. What comes back is not a region but a *line*: the
top edge of the floor, per column, which is the height the feet have to clear.
Per column rather than one number because the wall-floor junction is not level
in a frame — on our test clips it drops 35px from one side to the other, which
a single threshold would get wrong by that much at one end or the other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .holds import FrameMasks, decode_label_map, instance_mask


class Floor:
    """The floor's top edge, as a height per column. Normalized throughout."""

    def __init__(self, edge: np.ndarray):
        self.edge = np.asarray(edge, dtype=np.float32)   # one y per column

    @property
    def resolution(self) -> int:
        return len(self.edge)

    def top_at(self, x: float) -> float:
        """The floor's height at normalized x. 1.0 (off-frame) where unknown."""
        if not len(self.edge):
            return 1.0
        i = int(round(min(max(x, 0.0), 1.0) * (len(self.edge) - 1)))
        value = float(self.edge[i])
        return value if np.isfinite(value) else 1.0

    def is_clear(self, x: float, y: float, clearance: float) -> bool:
        """True when the point sits above the floor by at least *clearance*."""
        return y < self.top_at(x) - clearance

    def as_list(self) -> list:
        return [None if not np.isfinite(v) else round(float(v), 5) for v in self.edge]

    @classmethod
    def from_list(cls, values: list) -> "Floor":
        return cls(np.array([np.nan if v is None else float(v) for v in values],
                            dtype=np.float32))


def _edge_from_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Top-most floor pixel per column, resampled to *resolution* columns."""
    h, w = mask.shape
    edge = np.full(w, np.nan, dtype=np.float32)
    # argmax on a boolean column gives the first True; any() guards the all-False
    # columns, where argmax would silently return 0 and put the floor at the top
    # of the frame.
    has = mask.any(axis=0)
    first = mask.argmax(axis=0)
    edge[has] = first[has] / h

    if w != resolution:
        xs = np.linspace(0, w - 1, resolution)
        edge = np.interp(xs, np.arange(w), edge)
    return edge


def consensus(per_frame: list[FrameMasks], *, resolution: int, min_score: float,
              min_area: float) -> Floor | None:
    """One floor edge from several frames, by per-column median.

    Median rather than mean, and per column rather than per frame, because the
    climber standing on the mat takes a bite out of it: in that frame those
    columns report the floor starting *below* where it does, or not at all. Over
    a dozen frames the climber is somewhere different each time, so for any given
    column most frames see the real junction and the median lands on it.
    """
    edges = []
    for result in per_frame:
        usable = [i for i in result.items
                  if float(i.get("score") or 0.0) >= min_score
                  and i.get("instance_id") is not None]
        if not usable:
            continue
        # The floor is the big low thing, so the biggest instance wins and a
        # stray patch of grey wall cannot stand in for it. `area` is the mask's
        # own share of the frame, which is the honest measure — a floor seen
        # edge-on fills a wide, shallow box far larger than the mat inside it —
        # with the box as the fallback for a reply that omits it.
        def size(instance):
            if instance.get("area") is not None:
                return float(instance["area"])
            _, _, w, h = instance["bbox_xywh"]
            return float(w * h)

        best = max(usable, key=size)
        if size(best) < min_area:
            continue
        mask = instance_mask(decode_label_map(result.mask), best)
        if mask is not None:
            edges.append(_edge_from_mask(mask, resolution))

    if not edges:
        return None

    stacked = np.vstack(edges)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stacked, axis=0)

    # Columns nobody ever saw floor in: fill from their neighbours, so a foot
    # over a gap is still measured against something rather than treated as
    # airborne by default.
    finite = np.isfinite(median)
    if not finite.any():
        return None
    if not finite.all():
        median = np.interp(np.arange(len(median)), np.flatnonzero(finite), median[finite])
    return Floor(median)


def segment(client, frames, *, model: str, prompt: str, resolution: int,
            min_score: float, min_area: float, workers: int, console=None):
    """Segment the floor on the sampled frames and reduce them to one edge."""
    from .holds import segment_frames

    per_frame, usages = segment_frames(
        client, frames, model=model, prompt=prompt, min_score=0.0,
        workers=workers, console=None)
    if console:
        counts = [len(f.items) for f in per_frame]
        console.print(f"  floor: {sum(1 for c in counts if c)}/{len(counts)} frames "
                      f"returned a mask")
    return consensus(per_frame, resolution=resolution, min_score=min_score,
                     min_area=min_area), usages


def draw(frame, floor: Floor, *, color, thickness: int):
    """The floor line, so the rule the clock uses is visible rather than implied."""
    if floor is None:
        return frame
    h, w = frame.shape[:2]
    points = []
    for x in range(w):
        y = floor.top_at(x / max(w - 1, 1))
        if y < 1.0:
            points.append((x, int(round(y * h))))
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color,
                      thickness, cv2.LINE_AA)
    return frame


# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str,
               settings: dict) -> Path:
    stat = video.stat()
    key = json.dumps({
        "video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "model": model, "prompt": prompt, "settings": settings,
    }, sort_keys=True)
    return cache_dir / f"floor.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, floor: Floor | None, usages: list, *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp,
        "edge": floor.as_list() if floor is not None else None,
        "usages": usages,
    }))


def load_cache(path: Path):
    """Read a cached floor edge, or None if missing or unreadable.

    A cached *miss* — the model found no floor — is a real result and is
    honoured, so a clip with no visible ground does not re-ask every run.
    """
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    edge = blob.get("edge")
    floor = Floor.from_list(edge) if edge else None
    return floor, blob.get("usages") or [], blob.get("created", "unknown")
