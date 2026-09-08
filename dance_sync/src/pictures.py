"""Pictures — the shapes the dancers land in, and how well they match.

In dance the choreography is a sequence of *pictures*: you travel, you arrive,
you hold for an instant, you travel again. The travelling is where bodies
legitimately differ — different legs cover ground differently — and the picture
is what was actually choreographed and what an audience reads. Scoring every
frame equally spends most of its attention on the transitions.

So this module scores the pictures only, and it splits the question the way a
dancer would:

**Did we land in the same shape?** Each dancer is compared **at their own
arrival**, not at a common frame. That is the whole point — if one dancer
arrives two frames late but lands in exactly the same shape, the shape is
correct and only the timing is off. Comparing them at the same instant would
charge that to the shape.

**Did we land at the same time?** The spread of the arrival times, in
milliseconds. One number per picture, independent of the shape.

A picture is found, not defined: it is a local minimum of the dancer's own
angular-speed envelope — the moment the body stops moving. No choreography
needs to be segmented and no notion of "a move" is required, which is the same
trick audio onset detection uses to avoid defining "a note".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Picture:
    """One moment the group arrived at a shape."""

    index: int
    frames: dict            # slot -> the frame that dancer arrived on
    time_seconds: float     # mean arrival
    spread_ms: float        # max-min arrival spread
    sd_ms: float
    shape_percent: float    # mean pairwise shape agreement, each at own arrival
    pair_shape: dict        # "a-b" -> percent
    offsets_ms: dict        # slot -> arrival relative to the group mean
    timing_percent: float = 0.0    # from the arrival spread
    score_percent: float = 0.0     # the blend — this is the similarity score
    pair_score: dict = field(default_factory=dict)
    dancer_score: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "time_seconds": round(self.time_seconds, 3),
            "frames": {str(k): int(v) for k, v in self.frames.items()},
            "spread_ms": round(self.spread_ms, 1),
            "sd_ms": round(self.sd_ms, 1),
            "shape_percent": round(self.shape_percent, 1),
            "timing_percent": round(self.timing_percent, 1),
            "score_percent": round(self.score_percent, 1),
            "pair_shape_percent": {k: round(v, 1) for k, v in self.pair_shape.items()},
            "pair_score_percent": {k: round(v, 1) for k, v in self.pair_score.items()},
            "dancer_score_percent": {str(k): round(v, 1) for k, v in self.dancer_score.items()},
            "offsets_ms": {str(k): round(v, 1) for k, v in self.offsets_ms.items()},
        }


@dataclass
class PictureAnalysis:
    fps: float
    pictures: list = field(default_factory=list)
    per_dancer_detected: dict = field(default_factory=dict)
    slots: list = field(default_factory=list)
    unmatched: int = 0
    cluster_frames: int = 0
    timing_tolerance_ms: float = 0.0
    timing_null_ms: float = 0.0
    weights: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.pictures)

    def summary(self) -> dict:
        if not self.pictures:
            return {"pictures": 0, "detected_per_dancer": self.per_dancer_detected}
        shape = np.array([p.shape_percent for p in self.pictures])
        spread = np.array([p.spread_ms for p in self.pictures])
        drift = {s: float(np.mean([p.offsets_ms[s] for p in self.pictures]))
                 for s in self.slots}
        absdrift = {s: float(np.mean([abs(p.offsets_ms[s]) for p in self.pictures]))
                    for s in self.slots}
        return {
            "pictures": self.n,
            "detected_per_dancer": self.per_dancer_detected,
            "unmatched_detections": self.unmatched,
            "cluster_window_frames": self.cluster_frames,
            "shape_percent": {
                "mean": round(float(shape.mean()), 1),
                "median": round(float(np.median(shape)), 1),
                "min": round(float(shape.min()), 1),
                "max": round(float(shape.max()), 1),
            },
            "timing_spread_ms": {
                "median": round(float(np.median(spread)), 1),
                "mean": round(float(spread.mean()), 1),
                "p90": round(float(np.percentile(spread, 90)), 1),
                "max": round(float(spread.max()), 1),
                "share_under_50ms": round(float(np.mean(spread < 50)), 3),
            },
            "per_dancer_offset_ms": {str(k): round(v, 1) for k, v in drift.items()},
            "per_dancer_abs_offset_ms": {str(k): round(v, 1) for k, v in absdrift.items()},
            "timing_percent": {
                "mean": round(float(np.mean([p.timing_percent for p in self.pictures])), 1),
                "median": round(float(np.median([p.timing_percent for p in self.pictures])), 1),
            },
            "score_percent": {
                "mean": round(float(np.mean([p.score_percent for p in self.pictures])), 1),
                "median": round(float(np.median([p.score_percent for p in self.pictures])), 1),
                "min": round(float(np.min([p.score_percent for p in self.pictures])), 1),
                "max": round(float(np.max([p.score_percent for p in self.pictures])), 1),
            },
            "construction": {
                "formula": " + ".join(f"{v:.2f} x {k}" for k, v in self.weights.items()),
                "timing_anchors_ms": {"perfect_under": self.timing_tolerance_ms,
                                      "zero_at": self.timing_null_ms},
                "note": ("shape is read at each dancer's own arrival, so landing late "
                         "in the right shape costs timing and not shape"),
            },
        }

    def held_series(self, n_frames: int) -> np.ndarray:
        """The score as a staircase — each picture's value held until the next.

        A picture-based score has no value between pictures, and interpolating
        would invent a reading for the travelling. Holding says what is true:
        this is the score as of the last thing the group landed.
        """
        out = np.full(n_frames, np.nan)
        for pic in self.pictures:
            frame = int(round(pic.time_seconds * self.fps))
            if 0 <= frame < n_frames:
                out[frame:] = pic.score_percent
        return out


def _detect(envelope: np.ndarray, *, min_gap: int, prominence: float) -> np.ndarray:
    """Positions of stillness points — the arrivals — in a speed envelope.

    A local minimum, required to be the lowest point within *min_gap* either
    side and to sit at least *prominence* of the clip's speed range below the
    local high. The prominence test is what separates arriving at a shape from
    merely slowing down mid-travel.
    """
    v = -np.nan_to_num(envelope, nan=np.inf)
    span = float(np.nanmax(envelope) - np.nanmin(envelope))
    if not np.isfinite(span) or span <= 0:
        return np.array([], int)

    out = []
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] and v[i] > v[i + 1]:
            lo, hi = max(0, i - min_gap), min(len(v), i + min_gap + 1)
            window = v[lo:hi]
            if v[i] == np.max(window) and (v[i] - np.min(window)) >= prominence * span:
                out.append(i)
    return np.array(out, int)


def analyze(similarity, *, cfg) -> PictureAnalysis:
    """Find the pictures and score each one for shape and for timing."""
    fps = similarity.fps or 30.0
    slots = list(similarity.slots)
    result = PictureAnalysis(fps=fps, slots=slots,
                             cluster_frames=int(cfg.PICTURE_CLUSTER_FRAMES))
    if not similarity.speed_series or len(slots) < 2:
        return result

    indices = similarity.indices
    posture = next((c for c in similarity.components if c.name == "posture"), None)
    if posture is None or not indices.size:
        return result

    weights = similarity.feature_weights
    typical, zone, scale = posture.typical, posture.zone, posture.scale
    hold = int(cfg.PICTURE_HOLD_FRAMES)

    def held_pose(slot: int, position: int) -> np.ndarray:
        """The shape at an arrival, averaged over the frames it is held for.

        A picture is held, not passed through, so the pose either side of the
        arrival is the same shape — averaging over it is a better estimate than
        trusting one frame, and measured on this clip it lifts the shape score
        4-5 points by removing single-frame keypoint noise. Averaged as
        directions rather than as numbers, so a segment sitting near +/-180
        degrees does not average to nonsense.
        """
        series = similarity.angle_series[slot]
        lo, hi = max(0, position - hold), min(len(series), position + hold + 1)
        window = series[lo:hi]
        if not np.isfinite(window).any():
            return series[position]
        with np.errstate(invalid="ignore"):
            return np.arctan2(np.nanmean(np.sin(window), axis=0),
                              np.nanmean(np.cos(window), axis=0))

    def shape_difference(a_row: np.ndarray, b_row: np.ndarray) -> float:
        """The posture difference between two poses, on the posture scale.

        Deliberately the same calibration as the continuous posture score — same
        weights, same per-part dead zones, same normalization — so a picture's
        shape percent is comparable with the frame-by-frame number rather than
        being a second scale nobody can relate to it.
        """
        per = np.abs(np.arctan2(np.sin(a_row - b_row), np.cos(a_row - b_row)))
        per = np.maximum(np.degrees(per) - zone, 0.0) / typical
        ok = np.isfinite(per)
        w = np.where(ok, weights, 0.0)
        total = w.sum()
        if total <= 0:
            return float("nan")
        return float((np.where(ok, per, 0.0) * w).sum() / total * scale)

    span = max(posture.null_distance - posture.tolerance, 1e-9)

    def to_percent(diff: float) -> float:
        return float(np.clip(1.0 - (diff - posture.tolerance) / span, 0.0, 1.0) * 100.0)

    # Timing anchors, in milliseconds. Perfect below the tolerance and zero at
    # the null, the same two-anchor shape the other scores use.
    #
    # The tolerance is the *measurement* limit: an arrival is a discrete frame,
    # so each dancer's is +/-1 frame (33 ms) and a spread carries about 47 ms of
    # that. Scoring below it would score rounding.
    #
    # The null is the cluster window — the point at which two arrivals stop
    # being called the same picture at all. Anchoring past it would be
    # meaningless, because a spread that large never appears by construction.
    tol_ms = float(cfg.PICTURE_TIMING_TOLERANCE_MS)
    null_ms = (float(cfg.PICTURE_CLUSTER_FRAMES) / fps * 1000.0
               if cfg.PICTURE_TIMING_NULL_MS is None else float(cfg.PICTURE_TIMING_NULL_MS))
    result.timing_tolerance_ms, result.timing_null_ms = tol_ms, null_ms
    t_span = max(null_ms - tol_ms, 1e-9)

    def timing_percent(spread_ms: float) -> float:
        return float(np.clip(1.0 - (spread_ms - tol_ms) / t_span, 0.0, 1.0) * 100.0)

    w_shape = float(cfg.PICTURE_WEIGHTS["shape"])
    w_time = float(cfg.PICTURE_WEIGHTS["timing"])
    total_w = w_shape + w_time
    w_shape, w_time = w_shape / total_w, w_time / total_w
    result.weights = {"shape": w_shape, "timing": w_time}

    # 1. every dancer's own arrivals, as positions in the scored-frame array
    detected: dict[int, np.ndarray] = {}
    for slot in slots:
        env = similarity.speed_series[slot]
        weighted = _weighted_envelope(env, weights)
        detected[slot] = _detect(weighted, min_gap=int(cfg.PICTURE_MIN_GAP_FRAMES),
                                 prominence=float(cfg.PICTURE_PROMINENCE))
    result.per_dancer_detected = {str(s): int(len(detected[s])) for s in slots}

    # 2. group arrivals that belong to the same picture. The reference dancer is
    #    whoever detected the most, so the sequence is anchored on the clearest
    #    reading rather than on an arbitrary slot.
    anchor = max(slots, key=lambda s: len(detected[s]))
    window = int(cfg.PICTURE_CLUSTER_FRAMES)
    used = {s: set() for s in slots}
    pictures: list[Picture] = []

    for pos in detected[anchor]:
        frames = {}
        for slot in slots:
            candidates = [p for p in detected[slot]
                          if abs(indices[p] - indices[pos]) <= window
                          and p not in used[slot]]
            if not candidates:
                frames = {}
                break
            pick = min(candidates, key=lambda p: abs(indices[p] - indices[pos]))
            frames[slot] = pick
        if not frames:
            continue
        for slot, p in frames.items():
            used[slot].add(p)

        times = np.array([indices[frames[s]] / fps for s in slots])
        poses = {s: held_pose(s, frames[s]) for s in slots}
        pair_shape = {}
        for a, b in itertools.combinations(slots, 2):
            pair_shape[f"{a}-{b}"] = to_percent(shape_difference(poses[a], poses[b]))
        mean_t = float(times.mean())
        spread = float((times.max() - times.min()) * 1000)
        offsets = {s: float((indices[frames[s]] / fps - mean_t) * 1000) for s in slots}
        t_pct = timing_percent(spread)

        # Per pair: that pair's shape, and the timing of just those two.
        pair_score = {}
        for a, b in itertools.combinations(slots, 2):
            gap = abs(offsets[a] - offsets[b])
            pair_score[f"{a}-{b}"] = (w_shape * pair_shape[f"{a}-{b}"]
                                      + w_time * timing_percent(gap))
        # Per dancer: the mean of the pairs they are in, which keeps the
        # per-dancer numbers on the same footing as everywhere else.
        dancer_score = {
            s: float(np.mean([v for k, v in pair_score.items()
                              if s in (int(k.split("-")[0]), int(k.split("-")[1]))]))
            for s in slots
        }

        pictures.append(Picture(
            index=len(pictures),
            frames={s: int(indices[frames[s]]) for s in slots},
            time_seconds=mean_t,
            spread_ms=spread,
            sd_ms=float(times.std() * 1000),
            shape_percent=float(np.mean(list(pair_shape.values()))),
            pair_shape=pair_shape,
            offsets_ms=offsets,
            timing_percent=t_pct,
            score_percent=w_shape * float(np.mean(list(pair_shape.values()))) + w_time * t_pct,
            pair_score=pair_score,
            dancer_score=dancer_score,
        ))

    result.pictures = pictures
    result.unmatched = sum(len(detected[s]) - len(used[s]) for s in slots)
    return result


def _weighted_envelope(speeds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Collapse per-segment speeds into one number per frame."""
    ok = np.isfinite(speeds)
    w = np.where(ok, weights, 0.0)
    total = w.sum(axis=1)
    return np.where(total > 0,
                    (np.where(ok, speeds, 0.0) * w).sum(axis=1)
                    / np.where(total > 0, total, 1.0), np.nan)


def frame_weight(pics, n_frames: int, *, sigma_frames: float, floor: float) -> np.ndarray:
    """How much each frame matters, peaking at the pictures.

    A Gaussian bump on every picture, plus a floor so the travelling still
    counts for something. This is what lets the pictures *weight* the metric
    rather than replace it: the reading stays continuous and scrubbable, while
    the moments the choreography actually specifies dominate the aggregate.
    """
    w = np.full(n_frames, float(floor))
    if not pics.pictures or sigma_frames <= 0:
        return w
    x = np.arange(n_frames)
    for pic in pics.pictures:
        centre = pic.time_seconds * pics.fps
        w = np.maximum(w, np.exp(-0.5 * ((x - centre) / sigma_frames) ** 2))
    return w


def timing_series(pics, n_frames: int) -> np.ndarray:
    """Each frame's nearest picture's timing score, for blending.

    Not held as a staircase and not interpolated between pictures — just "the
    timing of the landing nearest this moment", which is only ever used where
    the picture weight is high, i.e. near that landing.
    """
    out = np.full(n_frames, np.nan)
    if not pics.pictures:
        return out
    centres = np.array([p.time_seconds * pics.fps for p in pics.pictures])
    values = np.array([p.timing_percent for p in pics.pictures])
    nearest = np.argmin(np.abs(np.arange(n_frames)[:, None] - centres[None, :]), axis=1)
    return values[nearest]


def contribute(similarity, pics, *, cfg) -> None:
    """Fold the pictures into the continuous metric, in place.

    Two things change and neither of them discretizes anything:

    **Timing at the landings comes from the pictures.** The speed-envelope
    timing component measures whether two dancers are moving at the same rate,
    which is a proxy; the arrival spread at a picture measures whether they
    landed together, which is the thing. So near a picture the timing series is
    the picture's timing score, fading back to the envelope in between. The
    crossfade is the picture weight, so there is no seam and no step.

    **The aggregate is picture-weighted.** The average, the leaderboard and the
    bands are computed with the per-frame weights, so the headline number is
    driven by the landings while the trace stays a continuous per-frame reading
    you can scrub against a timestamp.
    """
    if not pics.pictures:
        return

    n_frames = similarity.n_frames
    weight = frame_weight(pics, n_frames,
                          sigma_frames=cfg.PICTURE_WEIGHT_SIGMA_FRAMES,
                          floor=cfg.PICTURE_WEIGHT_FLOOR)
    similarity.frame_weight = weight
    similarity.source = "frames, picture-weighted"

    timing = next((c for c in similarity.components if c.name == "timing"), None)
    posture = next((c for c in similarity.components if c.name == "posture"), None)
    if timing is None or posture is None:
        return

    idx = similarity.indices
    pic_timing = timing_series(pics, n_frames)[idx]
    blend = weight[idx]

    # The picture's timing is the group's arrival spread, so every pair gets the
    # same value near a landing. A per-pair crossfade would need the pair's own
    # spread, which `Picture.pair_score` has but mixes with that pair's shape;
    # blending that in here would double-count shape inside the timing
    # component.
    timing.pairwise = np.where(np.isfinite(pic_timing)[:, None],
                               blend[:, None] * pic_timing[:, None]
                               + (1 - blend[:, None]) * timing.pairwise,
                               timing.pairwise)
    timing.score = timing.pairwise.mean(axis=1)
    timing.series = np.full(n_frames, np.nan)
    timing.series[idx] = timing.score
    timing.picture_blended = True

    # Rebuild the summary from the components, unchanged in form.
    similarity.pairwise = (posture.pairwise * posture.weight
                           + timing.pairwise * timing.weight)
    similarity.score = similarity.pairwise.mean(axis=1)
    similarity.series = np.full(n_frames, np.nan)
    similarity.series[idx] = similarity.score
    from .similarity import _interpolate_gaps, _nan_smooth
    similarity.smoothed = _nan_smooth(
        _interpolate_gaps(similarity.series, cfg.SIMILARITY_INTERPOLATE_MAX_GAP),
        cfg.SIMILARITY_SMOOTH_FRAMES, cfg.SIMILARITY_SMOOTH_KERNEL)

    similarity.pairs_series = np.full((n_frames, len(similarity.pair_labels)), np.nan)
    similarity.pairs_series[idx] = similarity.pairwise

    per_dancer = np.stack(
        [similarity.pairwise[:, [i for i, pr in enumerate(similarity.pair_labels)
                                 if s in pr]].mean(axis=1)
         for s in similarity.slots], axis=1)
    similarity.dancer_series = np.full((n_frames, len(similarity.slots)), np.nan)
    similarity.dancer_series[idx] = per_dancer

    # Running mean, picture-weighted, so the leaderboard converges on the
    # landings rather than on the transitions.
    similarity.dancer_running = np.full((n_frames, len(similarity.slots)), np.nan)
    cw = np.cumsum(blend)
    running = np.cumsum(per_dancer * blend[:, None], axis=0) / np.maximum(cw, 1e-9)[:, None]
    similarity.dancer_running[idx] = running
    for column in range(similarity.dancer_running.shape[1]):
        col = similarity.dancer_running[:, column]
        filled = np.nan
        for i in range(n_frames):
            if np.isfinite(col[i]):
                filled = col[i]
            elif np.isfinite(filled):
                col[i] = filled

    from .similarity import _rank, _band_segments
    similarity.ranking = _rank(
        per_dancer, similarity.slots,
        block=max(1, int(round(cfg.RANK_BOOTSTRAP_BLOCK_SECONDS * (similarity.fps or 30.0)))),
        n_boot=cfg.RANK_BOOTSTRAP_SAMPLES, seed=cfg.RANK_BOOTSTRAP_SEED,
        tie_band=cfg.RANK_TIE_BAND, sample_weights=blend)
    _band_segments(similarity, cfg)
