"""How alike the dancers' poses are, as one number per frame.

Two metrics live here. Both answer "how different are these two bodies right
now", and both are calibrated the same way; they differ in what they measure.

**`"angles"` (the default) compares limb directions.** Each pose becomes a set
of segment angles measured inside the dancer's own torso frame: which way the
thigh points, how the forearm sits, how far the torso leans off vertical. The
score is a weighted mean angular error, in degrees.

**`"positions"` compares joint locations** — the original metric. Each pose is
moved to put its mid-hip at the origin and divided by its torso length, and the
score is a weighted mean joint distance in torso lengths.

Angles are the default because three separate problems with positions all come
from one root — a joint's position is not a property of that joint:

* **The chain compounds.** A shoulder off by a little puts the elbow off by that
  same amount *plus* its own error, and the wrist off by all three. One bad
  shoulder gets counted three times. Angles are measured per segment, so an
  error lands once, on the segment that has it.
* **Proportions leak in.** Torso-length scaling fixes overall size but not the
  ratio of arm to torso, so a longer-armed dancer reads as "arms always further
  out" in an identical pose. A direction has no length to leak.
* **The scale divisor is noisy.** Torso length swings 26-36% within a single
  dancer on this clip, and being a mostly-vertical measurement its noise lands
  on the mostly-vertical part of the disagreement. Angles need no divisor at
  all, which removes the problem rather than shrinking it.

Angles also survive body rotation far better, which measurement showed to be the
largest confound here: shoulder-width-over-torso swings 5x within each dancer,
meaning they turn constantly, and a few degrees of facing moves 2D joint
positions a long way.

Shared by both metrics:

**Isotropic units.** Coordinates arrive normalized to [0, 1] against the frame,
separately in x and y. On a 720x1280 clip one unit of x is 720 px and one of y
is 1280 px, so any geometry taken on the raw pairs — a distance *or* an angle —
comes out distorted. Everything multiplies back up by width and height first.

**Deliberate weights.** Left flat, the eight extremity joints take 88% of the
score while carrying 4-9x the frame-to-frame noise of the shoulders and hips. So
segments are weighted in tiers: the core carries the pose, the hands and feet
contribute without dominating. See `config.ANGLE_WEIGHTS`.

**A tolerance band.** Below `SIMILARITY_TOLERANCE` two poses count as identical
rather than being scored on a difference nobody can see and the model cannot
resolve. The default is the *measured* noise floor — each segment against its
own short-window smoothed path.

**A measured zero.** Turning a difference into a percentage needs to know what
"completely different" is. Two upright human bodies are never really unlike each
other, so a plain threshold reads 41% agreement on two unrelated poses. The far
anchor is therefore measured: the median difference between the dancers at
moments far enough apart to be independent. That is the null, and the score is
how far the pair has closed the gap from it:

    score = 100 * clamp(1 - (difference - tolerance) / (null - tolerance), 0, 1)

0% is "no more alike than two random moments of this dance"; 100% is identical
to within the tolerance band.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}

# The joints the "positions" metric compares: shoulders down. The face is
# excluded for the same reason it is not drawn — head keypoints are noisy and
# say nothing about what the body is doing.
BODY_JOINTS = tuple(range(5, 17))

POSITION_TIERS = {
    "left_shoulder": "shoulder", "right_shoulder": "shoulder",
    "left_hip": "hip", "right_hip": "hip",
    "left_knee": "knee", "right_knee": "knee",
    "left_elbow": "elbow", "right_elbow": "elbow",
    "left_wrist": "wrist", "right_wrist": "wrist",
    "left_ankle": "ankle", "right_ankle": "ankle",
}

# The "angles" metric's features: (name, tier, from-joint, to-joint).
#
# Every direction is measured inside the dancer's own torso frame — up is
# mid-hip to mid-shoulder — so it says where a limb points *relative to the
# body*, not relative to the screen. `torso_lean` is the exception and is
# deliberately measured against image vertical: leaning is part of the move, and
# a frame defined by the torso cannot see its own tilt.
#
# `shoulder_line` is the shoulder axis inside that frame — in 2D, the best
# available read on upper-body twist.
# The head is in, and the two features chosen for it are measured, not guessed.
# Candidates and their signal-to-noise (typical disagreement over own jitter):
#
#   mid_shoulder -> mid_ear   10.2   38 px   <- "neck", kept
#   right_ear -> left_ear     10.0   34 px   <- "head_twist", kept
#   mid_shoulder -> nose      10.0   41 px      duplicates neck, noisier
#   right_eye -> left_eye      8.7   14 px      same information, 2x shorter
#   mid_ear -> nose            8.8    8 px      far too short to aim reliably
#
# For scale: the thigh scores 8.9 and the torso 14.6, so the head features sit
# above the thigh. The ear line is preferred over the eye line purely on length
# — a 34 px segment gives a steadier direction than a 14 px one — and the nose
# is dropped because everything it adds, `neck` already carries.
ANGLE_FEATURES: tuple[tuple[str, str, str, str], ...] = (
    ("torso_lean", "torso", "mid_hip", "mid_shoulder"),
    ("neck", "neck", "mid_shoulder", "mid_ear"),
    ("head_twist", "head_twist", "right_ear", "left_ear"),
    ("shoulder_line", "shoulder_line", "right_shoulder", "left_shoulder"),
    ("left_thigh", "thigh", "left_hip", "left_knee"),
    ("right_thigh", "thigh", "right_hip", "right_knee"),
    ("left_upper_arm", "upper_arm", "left_shoulder", "left_elbow"),
    ("right_upper_arm", "upper_arm", "right_shoulder", "right_elbow"),
    ("left_forearm", "forearm", "left_elbow", "left_wrist"),
    ("right_forearm", "forearm", "right_elbow", "right_wrist"),
    ("left_shin", "shin", "left_knee", "left_ankle"),
    ("right_shin", "shin", "right_knee", "right_ankle"),
)


@dataclass
class SimilarityAnalysis:
    """Per-frame similarity, plus everything needed to explain it."""

    fps: float
    n_frames: int
    n_subjects: int
    pair_labels: list[tuple[int, int]]
    null_distance: float
    metric: str = "angles"
    unit: str = "degrees"
    tolerance: float = 0.0
    noise_floor: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    feature_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_share: dict = field(default_factory=dict)
    indices: np.ndarray = field(default_factory=lambda: np.array([], int))
    distance: np.ndarray = field(default_factory=lambda: np.array([]))       # (n, pairs)
    pairwise: np.ndarray = field(default_factory=lambda: np.array([]))       # (n, pairs) %
    score: np.ndarray = field(default_factory=lambda: np.array([]))          # (n,) raw %
    series: np.ndarray = field(default_factory=lambda: np.array([]))         # (n_frames,)
    smoothed: np.ndarray = field(default_factory=lambda: np.array([]))       # (n_frames,)
    pairs_series: np.ndarray = field(default_factory=lambda: np.array([]))   # (n_frames, pairs)
    slots: list[int] = field(default_factory=list)
    dancer_series: np.ndarray = field(default_factory=lambda: np.array([]))
    dancer_running: np.ndarray = field(default_factory=lambda: np.array([]))
    ranking: list[dict] = field(default_factory=list)
    components: list = field(default_factory=list)
    window_frames: tuple[int, int] | None = None
    source: str = "frames"
    frame_weight: np.ndarray = field(default_factory=lambda: np.array([]))
    angle_series: dict = field(default_factory=dict)   # slot -> (n_scored, n_features)
    speed_series: dict = field(default_factory=dict)
    band_cuts: tuple = ()
    band_labels: tuple[str, ...] = ()
    band_of_frame: np.ndarray = field(default_factory=lambda: np.array([]))
    segments: list[dict] = field(default_factory=list)
    excluded: dict = field(default_factory=dict)

    @property
    def n_scored(self) -> int:
        return len(self.indices)

    @property
    def coverage(self) -> float:
        return self.n_scored / self.n_frames if self.n_frames else 0.0

    def weighted_mean(self) -> float:
        """The average, weighted toward the pictures.

        A flat average over every frame gives the travelling the same say as the
        landings. This is the headline number when pictures are in play.
        """
        if not self.n_scored:
            return float("nan")
        if self.frame_weight.size == 0:
            return float(self.score.mean())
        w = self.frame_weight[self.indices]
        return float((self.score * w).sum() / max(w.sum(), 1e-9))

    def times(self) -> np.ndarray:
        return self.indices / self.fps if self.fps else self.indices.astype(float)

    def summary(self) -> dict:
        if not self.n_scored:
            return {"scored_frames": 0, "excluded": self.excluded}
        pair_stats = {
            f"{a}-{b}": {
                "mean": round(float(self.pairwise[:, i].mean()), 2),
                "median": round(float(np.median(self.pairwise[:, i])), 2),
            }
            for i, (a, b) in enumerate(self.pair_labels)
        }
        return {
            "metric": self.metric,
            "unit": self.unit,
            "scored_frames": self.n_scored,
            "coverage": round(self.coverage, 4),
            "window_frames": list(self.window_frames) if self.window_frames else None,
            "window_seconds": ([round(self.window_frames[0] / self.fps, 3),
                                round(self.window_frames[1] / self.fps, 3)]
                               if self.window_frames and self.fps else None),
            "null_distance": round(self.null_distance, 4),
            "tolerance": round(self.tolerance, 4),
            "noise_floor": round(self.noise_floor, 4),
            "median_matched_distance": round(float(np.median(self.distance)), 4),
            "weight_share_percent": self.feature_share,
            "score_percent": {
                "mean": round(float(self.score.mean()), 2),
                "picture_weighted_mean": round(self.weighted_mean(), 2),
                "median": round(float(np.median(self.score)), 2),
                "min": round(float(self.score.min()), 2),
                "max": round(float(self.score.max()), 2),
                "p05": round(float(np.percentile(self.score, 5)), 2),
                "p95": round(float(np.percentile(self.score, 95)), 2),
            },
            "pairs": pair_stats,
            "bands": {
                "cuts_percent": [round(v, 1) for v in self.band_cuts],
                "labels": list(self.band_labels),
                "note": ("cuts are percentiles of THIS clip, so every clip yields "
                         "roughly the same share of each band — good for finding the "
                         "weakest moments, not for judging the run"),
                "segments": self.segments,
            },
            "summary_construction": {
                "formula": "summary% = " + " + ".join(
                    f"{c.weight:.2f} x {c.name}%" for c in self.components),
                "note": ("the components are blended as percentages, not as raw "
                         "differences: degrees and degrees-per-second are not "
                         "commensurable, but both are calibrated against their own "
                         "measured null so both read 0 at chance and 100 at identical"),
            },
            "components": {c.name: c.summary() for c in self.components},
            "ranking": self.ranking,
            "excluded": self.excluded,
        }


# ── feature extraction ───────────────────────────────────────────────────────

def _pixels(kpts: np.ndarray, width: int, height: int) -> np.ndarray:
    """Keypoints in isotropic pixel units."""
    return np.stack([kpts[:, 0] * width, kpts[:, 1] * height], axis=1)


def _point(px: np.ndarray, name: str) -> np.ndarray:
    """A named landmark, including the two midpoints the torso frame needs."""
    if name == "mid_hip":
        return (px[J["left_hip"]] + px[J["right_hip"]]) / 2
    if name == "mid_shoulder":
        return (px[J["left_shoulder"]] + px[J["right_shoulder"]]) / 2
    if name == "mid_ear":
        return (px[J["left_ear"]] + px[J["right_ear"]]) / 2
    return px[J[name]]


def _torso_length(px: np.ndarray) -> float:
    return float(np.linalg.norm(_point(px, "mid_shoulder") - _point(px, "mid_hip")))


def angle_features(px: np.ndarray, min_segment_px: float) -> np.ndarray:
    """Every segment's direction, in radians. NaN for a segment too short to aim.

    Directions are expressed inside the torso frame, so the result describes the
    pose of the body rather than its orientation on screen — except
    ``torso_lean``, which is the frame's own tilt against image vertical and so
    has to be measured outside it.

    A segment shorter than *min_segment_px* is dropped rather than measured: a
    forearm pointing at the camera projects to a couple of pixels, and its
    apparent direction is then noise. Dropping it costs that segment's weight on
    that frame, which the weighted mean redistributes.
    """
    hip = _point(px, "mid_hip")
    up = _point(px, "mid_shoulder") - hip
    torso = float(np.linalg.norm(up))

    out = np.full(len(ANGLE_FEATURES), np.nan)
    if torso < min_segment_px:
        return out

    # Torso frame: `up` along the spine, `right` perpendicular to it.
    u = up / torso
    r = np.array([-u[1], u[0]])

    for i, (name, _tier, a, b) in enumerate(ANGLE_FEATURES):
        if name == "torso_lean":
            # Against image vertical (0 = upright), signed, so leaning left and
            # leaning right are different poses rather than the same magnitude.
            out[i] = np.arctan2(u[0], -u[1])
            continue
        vec = _point(px, b) - _point(px, a)
        if float(np.linalg.norm(vec)) < min_segment_px:
            continue
        out[i] = np.arctan2(float(vec @ r), float(vec @ u))
    return out


def _angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute angle between two sets of directions, in radians."""
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _unwrap_nan(values: np.ndarray) -> np.ndarray:
    """Unwrap each column past +/-pi, leaving gaps as gaps.

    ``np.unwrap`` works on differences, so one NaN would poison every sample
    after it. Interpolating across the gap first keeps the branch choices right
    and the mask is restored afterwards.
    """
    out = np.array(values, dtype=float, copy=True)
    for c in range(out.shape[1]):
        col = out[:, c]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        filled = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
        out[:, c] = np.where(ok, np.unwrap(filled), np.nan)
    return out


def speed_features(angles: np.ndarray, fps: float, smooth_frames: int) -> np.ndarray:
    """How fast each segment is turning, in degrees per second.

    This is the timing signal. Magnitude only — the *direction* a limb is
    turning is choreography, and mixing it in here would make the timing
    component a second posture score.

    Smoothed before differentiating, and deliberately so: a raw frame-to-frame
    difference of a signal whose own noise floor is ~1.8 degrees produces
    ~54 deg/s of pure noise at 30 fps, which would swamp the real speeds. A
    centered average over TIMING_SMOOTH_FRAMES first, then a central
    difference, keeps the envelope and drops the jitter.
    """
    smoothed = _moving_average(_unwrap_nan(angles), smooth_frames)
    return np.degrees(np.abs(np.gradient(smoothed, axis=0) * fps))


# ── helpers ──────────────────────────────────────────────────────────────────

def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0, edge-padded, NaNs ignored."""
    if window <= 1 or values.size == 0:
        return values.copy()
    flat = values.reshape(len(values), -1)
    pad = window // 2
    kernel = np.ones(window)

    present = np.isfinite(flat)
    padded_vals = np.pad(np.where(present, flat, 0.0), ((pad, pad), (0, 0)), mode="edge")
    padded_mask = np.pad(present.astype(float), ((pad, pad), (0, 0)), mode="edge")

    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        total = np.convolve(padded_vals[:, c], kernel, mode="valid")[:len(flat)]
        count = np.convolve(padded_mask[:, c], kernel, mode="valid")[:len(flat)]
        out[:, c] = np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)
    return out.reshape(values.shape)


def _smooth_scale(values: np.ndarray, window: int) -> np.ndarray:
    """Stabilize the per-frame torso length without freezing it.

    Torso length is the "positions" metric's divisor, and measured per frame it
    swings 26-36% within one dancer here: bending forward foreshortens the spine
    and inflates every normalized coordinate at once. A single median over the
    clip would fix that but could not follow a dancer walking toward the camera,
    which is exactly what the centre dancer does (torso 121 -> 177 px). A moving
    average keeps the slow depth change and drops the per-frame wobble.
    """
    return _moving_average(values.reshape(-1, 1), window).ravel()


def _interpolate_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short interior gaps by linear interpolation.

    Only *interior* gaps, and only short ones. The frames before the last dancer
    walks in were never scored at all — there is nothing to interpolate between
    — so those stay undefined, and so does anything longer than *max_gap*, where
    a straight line really would be inventing a stretch of dance that was never
    measured.
    """
    if max_gap <= 0:
        return values.copy()

    out = values.copy()
    present = np.flatnonzero(np.isfinite(values))
    if present.size < 2:
        return out

    for a, b in zip(present[:-1], present[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            out[a + 1:b] = np.linspace(values[a], values[b], gap + 2)[1:-1]
    return out


def _nan_smooth(values: np.ndarray, window: int, kernel: str = "gaussian") -> np.ndarray:
    """Centered smoothing that steps over gaps rather than through them.

    The kernel matters more than the width. A boxcar — a plain moving average —
    has a poor stopband, so it *ripples*: it leaves high-frequency wiggle behind
    while flattening the peaks it does catch. A Gaussian of the same width
    removes more of the wiggle and keeps more of the amplitude, which is not a
    trade-off but strictly better. Measured on the reference clip, counting
    local turns of at least 4 points and the range they span:

        kernel   width   turns/s   p05-p95   r with raw
        boxcar      45      1.71     39-64         0.54
        gaussian    45      1.34     36-70         0.81

    ``window`` is the full width either way; the Gaussian takes it as ±3σ, so
    the two are comparable in reach and lag.

    Display only. The raw per-frame series is what the averages, the ranking and
    `similarity.json` are computed from — smoothing changes the trace, the bands
    cut from it and the readout sampled off it, and nothing else.
    """
    if window <= 1:
        return values.copy()

    if kernel == "gaussian":
        sigma = window / 6.0
        half = max(1, int(3 * sigma))
        offsets = np.arange(-half, half + 1)
        k = np.exp(-0.5 * (offsets / sigma) ** 2)
    elif kernel == "boxcar":
        k = np.ones(window)
    else:
        raise ValueError(f"unknown smoothing kernel {kernel!r}")

    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)
    total = np.convolve(filled, k, mode="same")
    count = np.convolve(present.astype(float), k, mode="same")

    out = np.full_like(values, np.nan)
    usable = present & (count > 0)
    out[usable] = total[usable] / count[usable]
    return out


def _at_offset(full: np.ndarray, indices: np.ndarray, shift: int, n_frames: int) -> np.ndarray:
    """A dancer's features at frame ``i + shift`` for each scored frame ``i``.

    Indexed by real frame number rather than by position in the scored list,
    because the scored frames have gaps — a two-position step is not always a
    two-frame step. Frames that fall outside the clip, or that were never
    scored, come back as NaN and drop out of the minimum.
    """
    target = indices + shift
    inside = (target >= 0) & (target < n_frames)
    out = np.full((len(indices), full.shape[1]), np.nan)
    out[inside] = full[target[inside]]
    return out


def _rank(dancer_scores: np.ndarray, slots: list[int], *, block: int, n_boot: int,
          seed: int, tie_band: float, sample_weights: np.ndarray | None = None) -> list[dict]:
    """Rank the dancers by agreement with the rest, with an honest tie test.

    The mean alone would put this clip's three dancers in a confident 1-2-3, and
    two of those places would be noise. So the spread is estimated by a **block**
    bootstrap — resampling one-second blocks rather than individual frames,
    because consecutive frames of a dance are nowhere near independent and
    resampling them singly reports a false precision of a few tenths of a
    percent.

    Two dancers whose win probability lands inside *tie_band* of even share a
    rank, competition-style (1, 2, 2), rather than being ordered by a margin the
    data does not support.
    """
    n = len(slots)
    sw = (np.ones(len(dancer_scores)) if sample_weights is None
          else np.asarray(sample_weights, dtype=float))
    means = (dancer_scores * sw[:, None]).sum(axis=0) / max(sw.sum(), 1e-9)
    order = list(np.argsort(-means))

    n_blocks = len(dancer_scores) // block
    boot = None
    if n_blocks >= 2:
        trimmed = dancer_scores[:n_blocks * block]
        # All blocks are the same length, so the mean of a resampled set of
        # blocks equals the mean of their block means — one vectorized gather
        # instead of n_boot concatenations.
        bw = sw[:n_blocks * block].reshape(n_blocks, block)
        block_means = ((trimmed.reshape(n_blocks, block, n) * bw[:, :, None]).sum(axis=1)
                       / np.maximum(bw.sum(axis=1), 1e-9)[:, None])
        rng = np.random.default_rng(seed)
        boot = block_means[rng.integers(0, n_blocks, (n_boot, n_blocks))].mean(axis=1)

    rows: list[dict] = []
    rank = 1
    for position, i in enumerate(order):
        tied_with = []
        if boot is not None:
            for j in order:
                if j == i:
                    continue
                p = float((boot[:, i] > boot[:, j]).mean())
                if abs(p - 0.5) <= tie_band:
                    tied_with.append(slots[j])

        if position > 0:
            previous = order[position - 1]
            if not (boot is not None and slots[previous] in tied_with):
                rank = position + 1

        row = {
            "dancer": slots[i],
            "rank": rank,
            "mean": round(float(means[i]), 2),
            "median": round(float(np.median(dancer_scores[:, i])), 2),
            "weighted": sample_weights is not None,
            "tied_with": tied_with,
        }
        if boot is not None:
            lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
            row["ci95"] = [round(float(lo), 2), round(float(hi), 2)]
            row["p_ranks_first"] = round(float((np.argmax(boot, axis=1) == i).mean()), 3)
        rows.append(row)
    return rows


# Head keypoints are compared now, so they have to be on screen too — the model
# returns all 17 whatever is visible, and an off-frame ear is a guess.
CHECKED_JOINTS = tuple(sorted(set(BODY_JOINTS) | {J["left_ear"], J["right_ear"]}))


def _in_frame(kpts: np.ndarray) -> bool:
    """Every compared joint inside the frame.

    A dancer half out of shot still gets a full 17-keypoint pose — the model
    always returns all of them — but the missing limbs are guesses, and a
    similarity score built on a guess is worse than no score.
    """
    checked = kpts[list(CHECKED_JOINTS)]
    return bool((checked >= 0.0).all() and (checked <= 1.0).all())



@dataclass
class Component:
    """One scored axis of similarity — posture or timing."""

    name: str
    unit: str
    weight: float = 0.0
    null_distance: float = 0.0
    tolerance: float = 0.0
    noise_floor: float = 0.0
    share: dict = field(default_factory=dict)
    match_window: int = 0
    picture_blended: bool = False
    dead_zone: dict = field(default_factory=dict)
    typical: np.ndarray = field(default_factory=lambda: np.array([]))
    zone: np.ndarray = field(default_factory=lambda: np.array([]))
    scale: float = 1.0
    ranking: list[dict] = field(default_factory=list)
    distance: np.ndarray = field(default_factory=lambda: np.array([]))
    pairwise: np.ndarray = field(default_factory=lambda: np.array([]))
    score: np.ndarray = field(default_factory=lambda: np.array([]))
    series: np.ndarray = field(default_factory=lambda: np.array([]))
    smoothed: np.ndarray = field(default_factory=lambda: np.array([]))
    dancer_series: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> dict:
        if not self.score.size:
            return {"name": self.name, "unit": self.unit}
        return {
            "name": self.name,
            "unit": self.unit,
            "summary_weight": round(self.weight, 4),
            "null_distance": round(self.null_distance, 4),
            "tolerance": round(self.tolerance, 4),
            "noise_floor": round(self.noise_floor, 4),
            "match_window_frames": self.match_window,
            "dead_zone_degrees": self.dead_zone,
            "median_matched_distance": round(float(np.median(self.distance)), 4),
            "weight_share_percent": self.share,
            "score_percent": {
                "mean": round(float(self.score.mean()), 2),
                "median": round(float(np.median(self.score)), 2),
                "min": round(float(self.score.min()), 2),
                "max": round(float(self.score.max()), 2),
                "p05": round(float(np.percentile(self.score, 5)), 2),
                "p95": round(float(np.percentile(self.score, 95)), 2),
            },
            "ranking": self.ranking,
        }


def _score_component(name, F, per_feature, indices, *, weights, tiers, names, unit,
                     n_frames, fps, cfg, pair_labels, share_mode, tolerance_cfg,
                     fallback, angular, match_window=0, dead_zone=None) -> Component:
    """Turn one feature set into a calibrated 0-100 score.

    The same four steps for both components, which is what puts their
    percentages on one scale and makes the summary blend meaningful:

    1. a weighted difference per pair per frame,
    2. a measured noise floor, which the tolerance band defaults to,
    3. a measured null from moments far enough apart to be independent,
    4. the two anchors mapped onto 0-100.
    """
    comp = Component(name=name, unit=unit)

    # Make the weights mean what they say. Features do not disagree on
    # comparable scales — the forearms differ by 40 degrees typically while the
    # torso differs by 5.5 — so weighting raw units lets the loudest feature
    # dominate whatever weight it was given. Dividing each feature by its own
    # typical disagreement first makes a weight of 0.35 buy a 0.35 share, and
    # rescaling by the weighted mean typical value puts the result back into
    # interpretable units rather than leaving it dimensionless.
    # A per-feature dead zone: how far each segment may differ before it counts
    # as differing at all. Per feature rather than one global figure, because
    # the forgiveness a body part deserves is a property of that body part — a
    # forearm 20 degrees off is nothing, a torso 20 degrees off is a different
    # shape. A single composite tolerance cannot express that: it forgives a
    # small *average*, so a large arm error can be cancelled by a still torso.
    zone = (np.zeros(len(names)) if dead_zone is None
            else np.array([float(dead_zone[t]) for t in tiers]))

    def raw(a, b):
        return np.maximum(per_feature(a, b) - zone, 0.0)

    # Normalization uses each feature's *raw* typical disagreement, measured
    # before the dead zone. Using the dead-zoned value instead is a trap: a
    # segment whose dead zone exceeds its natural variation has a median of
    # zero, gets divided by the floor, and its rare large differences explode —
    # on this clip that handed the shin a 26% share on a 0.20 weight. Dividing
    # by the raw scale keeps the divisor stable, so forgiving a segment makes it
    # count for *less*, which is the whole point.
    typical = np.ones(len(names))
    scale = 1.0
    if share_mode:
        measured = np.nanmedian(
            np.concatenate([per_feature(F[a], F[b]) for a, b in pair_labels]), axis=0)
        floorv = max(float(np.nanmedian(measured)) * 0.1, 1e-6)
        typical = np.where(np.isfinite(measured), np.maximum(measured, floorv), 1.0)
        scale = float((weights * typical).sum())
        # A dead zone bigger than the segment's own typical variation forgives
        # everything and silently removes the segment; cap it so a mis-set
        # number degrades instead of disappearing.
        over = zone > typical * cfg.ANGLE_TOLERANCE_CAP
        if over.any():
            zone = np.where(over, typical * cfg.ANGLE_TOLERANCE_CAP, zone)

    def difference(a, b):
        """Weighted mean difference, renormalized over the features present.

        Renormalizing is what makes a dropped feature cost its weight rather
        than score as perfect agreement.
        """
        per = raw(a, b) / typical
        ok = np.isfinite(per)
        w = np.where(ok, weights, 0.0)
        total = w.sum(axis=-1)
        return np.where(total > 0,
                        (np.where(ok, per, 0.0) * w).sum(axis=-1)
                        / np.where(total > 0, total, 1.0) * scale,
                        np.nan)

    # A match window lets each frame compare against the best-matching instant
    # within +/-k frames instead of only the same instant. Without it, a
    # one-frame offset during a fast move craters the posture score — which is
    # timing error being charged to posture, exactly the contamination the
    # component split exists to remove. Measured on this clip, +/-2 frames
    # (67 ms) cuts posture's frame-to-frame jitter by about a third.
    #
    # The window is applied to the **null as well as the matched pairs**, and
    # that is not optional: giving matched pairs a best-of-five choice while the
    # baseline gets one would hand the score a free improvement that means
    # nothing. Windowing both keeps the two anchors on the same footing, which
    # is why the real gain is smaller than a matched-only test suggests.
    #
    # Timing deliberately gets no window. Sliding it in time to find a better
    # match would erase the thing it measures.
    full = {slot: np.full((n_frames, series.shape[1]), np.nan)
            for slot, series in F.items()}
    for slot, series in F.items():
        full[slot][indices] = series
    shifts = range(-match_window, match_window + 1) if match_window else (0,)

    matched = []
    for a, b in pair_labels:
        best = np.full(len(indices), np.inf)
        for sh in shifts:
            other = F[b] if sh == 0 else _at_offset(full[b], indices, sh, n_frames)
            best = np.fmin(best, np.nan_to_num(difference(F[a], other), nan=np.inf))
        matched.append(np.where(np.isfinite(best), best, np.nan))
    comp.distance = np.stack(matched, axis=1)
    comp.match_window = int(match_window)
    comp.typical, comp.zone, comp.scale = typical, zone, scale
    if dead_zone is not None:
        comp.dead_zone = dict(dead_zone)

    # The noise floor, measured: every feature against its own 3-frame smoothed
    # path. Real movement is smooth over 100 ms at 30 fps; single-frame wobble
    # is not.
    floors = []
    for slot in F:
        series = F[slot]
        if angular:
            unwrapped = _unwrap_nan(series)
            residual = np.degrees(np.abs(unwrapped - _moving_average(unwrapped, 3)))
        elif series.ndim == 3:
            residual = np.linalg.norm(series - _moving_average(series, 3), axis=2)
        else:
            residual = np.abs(series - _moving_average(series, 3))
        residual = residual / typical * scale
        ok = np.isfinite(residual)
        w = np.where(ok, weights, 0.0)
        total = w.sum(axis=1)
        floors.append(np.where(total > 0,
                               (np.where(ok, residual, 0.0) * w).sum(axis=1)
                               / np.where(total > 0, total, 1.0),
                               np.nan))
    comp.noise_floor = float(np.nanmedian(np.concatenate(floors)))
    comp.tolerance = (comp.noise_floor if tolerance_cfg is None else float(tolerance_cfg))

    # The null anchor, exhaustive rather than sampled so it is reproducible:
    # every cross-dancer pairing of two moments at least NULL_LAG_SECONDS apart.
    # That lag is what makes the samples independent — adjacent frames of a
    # dance are nearly the same pose, so a short lag would measure the
    # choreography instead of the baseline.
    lag = np.abs(indices[:, None] - indices[None, :]) >= cfg.NULL_LAG_SECONDS * fps
    if lag.any():
        pool = []
        for a, b in pair_labels:
            best = np.full((len(indices), len(indices)), np.inf)
            for sh in shifts:
                other = F[b] if sh == 0 else _at_offset(full[b], indices, sh, n_frames)
                best = np.fmin(best, np.nan_to_num(
                    difference(F[a][:, None], other[None, :]), nan=np.inf))
            pool.append(np.where(np.isfinite(best), best, np.nan)[lag])
        comp.null_distance = float(np.nanmedian(np.concatenate(pool)))
    else:
        comp.null_distance = float(fallback)

    # The 100% anchor, optionally re-set to the best agreement these bodies
    # actually reach rather than to "identical".
    #
    # Why it was needed: posture was anchored at 0 degrees — exact equality —
    # while timing was anchored at its own noise floor, so the two components
    # were not asking the same question. Exact equality is also below what the
    # instrument can resolve: the noise floor is 1.55 degrees and the dancers'
    # best frame is 1.71, i.e. at their best they are within measurement noise
    # of each other, and the score still called that 92%.
    #
    # max() with the noise floor, because no ceiling can claim resolution the
    # measurement does not have.
    if cfg.SIMILARITY_CEILING_PERCENTILE is not None and comp.distance.size:
        comp.tolerance = max(comp.tolerance, float(np.percentile(
            comp.distance, cfg.SIMILARITY_CEILING_PERCENTILE)))

    span = max(comp.null_distance - comp.tolerance, 1e-9)
    comp.pairwise = np.clip(1.0 - (comp.distance - comp.tolerance) / span, 0.0, 1.0) * 100.0
    comp.score = comp.pairwise.mean(axis=1)

    comp.series = np.full(n_frames, np.nan)
    comp.series[indices] = comp.score
    comp.smoothed = _nan_smooth(
        _interpolate_gaps(comp.series, cfg.SIMILARITY_INTERPOLATE_MAX_GAP),
        cfg.SIMILARITY_SMOOTH_FRAMES, cfg.SIMILARITY_SMOOTH_KERNEL)

    slots = sorted(F)
    comp.dancer_series = np.full((n_frames, len(slots)), np.nan)
    comp.dancer_series[indices] = np.stack(
        [comp.pairwise[:, [i for i, p in enumerate(pair_labels) if s in p]].mean(axis=1)
         for s in slots], axis=1)

    # What each tier actually contributed — its weight times how much it really
    # disagrees — so the weighting is checkable rather than asserted.
    contribution = np.zeros(len(names))
    for a, b in pair_labels:
        contribution += np.nan_to_num(
            np.nanmean(raw(F[a], F[b]) / typical, axis=0)) * weights
    if contribution.sum() > 0:
        comp.share = {
            tier: round(float(100 * contribution[[t == tier for t in tiers]].sum()
                              / contribution.sum()), 1)
            for tier in dict.fromkeys(tiers)
        }
    return comp


# ── the analysis ─────────────────────────────────────────────────────────────

def _band_segments(analysis, cfg) -> None:
    """Cut the display series into three bands and merge runs into segments.

    The cuts are percentiles of this clip, so the bands find the weakest and
    strongest stretches of *this* run rather than judging it — every clip yields
    roughly the same share of each band, including a flawless one. The absolute
    score is what says whether the run was good.
    """
    fps = analysis.fps or 30.0
    n_frames = analysis.n_frames
    analysis.segments = []

    display = analysis.smoothed
    finite = display[np.isfinite(display)]
    if not finite.size:
        return

    # Any number of bands: the cuts are len(labels) - 1 percentiles, so adding a
    # category needs no code change.
    cuts = np.percentile(finite, cfg.BAND_PERCENTILES)
    analysis.band_cuts = tuple(float(v) for v in np.atleast_1d(cuts))
    analysis.band_labels = tuple(cfg.BAND_LABELS)

    band = np.full(n_frames, -1, int)
    ok = np.isfinite(display)
    band[ok] = np.searchsorted(np.asarray(analysis.band_cuts), display[ok])
    analysis.band_of_frame = band

    min_frames = max(1, int(round(cfg.BAND_MIN_SECONDS * fps)))
    i = 0
    while i < n_frames:
        if band[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n_frames and band[j + 1] == band[i]:
            j += 1
        if j - i + 1 >= min_frames:
            analysis.segments.append({
                "band": int(band[i]),
                "label": cfg.BAND_LABELS[band[i]],
                "start_frame": int(i), "end_frame": int(j),
                "start_seconds": round(i / fps, 2), "end_seconds": round((j + 1) / fps, 2),
                "duration_seconds": round((j - i + 1) / fps, 2),
                "mean_score": round(float(np.nanmean(display[i:j + 1])), 1),
            })
        i = j + 1


def analyze(frames: list[dict], *, width: int, height: int, fps: float, n_frames: int,
            cfg) -> SimilarityAnalysis:
    """Score every frame that has all subjects fully in shot."""
    metric = cfg.SIMILARITY_METRIC
    excluded = {"outside_window": 0, "missing_subject": 0, "out_of_frame": 0,
                "degenerate_torso": 0}

    # The scored window. A routine usually has an intro and an outro that are
    # freestyle, and stretches in the middle can be too — scoring them measures
    # people deliberately doing different things. Bounds are given in seconds
    # and clamped to the clip.
    #
    # This bounds the **null anchor** as well, and that is the part that matters
    # most: freestyle is more varied than choreography, so letting it into the
    # baseline would raise the "unrelated" distance and inflate every score
    # inside the window for free.
    if cfg.SCORE_WINDOW is None:
        lo, hi = 0, n_frames - 1
    else:
        start, end = cfg.SCORE_WINDOW
        lo = 0 if start is None else max(0, int(round(start * fps)))
        hi = n_frames - 1 if end is None else min(n_frames - 1, int(round(end * fps)))

    slots_seen: set[int] = set()
    for frame in frames:
        for person in frame["persons"]:
            if person.get("slot") is not None:
                slots_seen.add(person["slot"])
    slots = sorted(slots_seen)
    n_subjects = len(slots)

    # Pass 1 — raw pixel keypoints for every frame with everyone fully in shot.
    raw: dict[int, dict[int, np.ndarray]] = {}
    for frame in frames:
        if not (lo <= frame["index"] <= hi):
            excluded["outside_window"] += 1
            continue
        per_slot: dict[int, np.ndarray] = {}
        rejected = False
        for person in frame["persons"]:
            slot = person.get("slot")
            if slot is None:
                continue
            kpts = np.asarray(person["kpts_xy"], dtype=float)
            if not _in_frame(kpts):
                rejected = True
                continue
            px = _pixels(kpts, width, height)
            if _torso_length(px) < cfg.MIN_TORSO_PX:
                excluded["degenerate_torso"] += 1
                continue
            per_slot[slot] = px

        if len(per_slot) == n_subjects and n_subjects >= 2:
            raw[frame["index"]] = per_slot
        elif rejected:
            excluded["out_of_frame"] += 1
        else:
            excluded["missing_subject"] += 1

    pair_labels = [tuple(p) for p in itertools.combinations(slots, 2)]

    if metric == "angles":
        names = [name for name, *_ in ANGLE_FEATURES]
        tiers = [tier for _n, tier, *_ in ANGLE_FEATURES]
        weight_map, unit = cfg.ANGLE_WEIGHTS, "degrees"
    else:
        names = [KPT_NAMES[j] for j in BODY_JOINTS]
        tiers = [POSITION_TIERS[n] for n in names]
        weight_map, unit = cfg.POSITION_WEIGHTS, "torso lengths"

    weights = np.array([float(weight_map[t]) for t in tiers])
    weights = weights / weights.sum()

    analysis = SimilarityAnalysis(
        fps=fps, n_frames=n_frames, n_subjects=n_subjects, pair_labels=pair_labels,
        null_distance=0.0, metric=metric, unit=unit, excluded=excluded, slots=slots,
        window_frames=(lo, hi),
        feature_names=names, feature_weights=weights,
        series=np.full(n_frames, np.nan), smoothed=np.full(n_frames, np.nan),
        pairs_series=np.full((n_frames, len(pair_labels)), np.nan),
        dancer_series=np.full((n_frames, len(slots)), np.nan),
        dancer_running=np.full((n_frames, len(slots)), np.nan),
    )
    if not raw or not pair_labels:
        return analysis

    indices = np.array(sorted(raw))

    # Pass 2 — features. Positions need a scale; angles do not, which is one of
    # the reasons they are the default.
    F: dict[int, np.ndarray] = {}
    for slot in slots:
        stack = np.stack([raw[i][slot] for i in indices])
        if metric == "angles":
            F[slot] = np.stack([angle_features(px, cfg.MIN_SEGMENT_PX) for px in stack])
        else:
            torso = np.array([_torso_length(px) for px in stack])
            if cfg.SCALE_MODE == "smoothed":
                torso = _smooth_scale(torso, cfg.SCALE_SMOOTH_FRAMES)
            elif cfg.SCALE_MODE == "median":
                torso = np.full_like(torso, float(np.median(torso)))
            hip = (stack[:, J["left_hip"]] + stack[:, J["right_hip"]]) / 2
            F[slot] = (stack[:, list(BODY_JOINTS)] - hip[:, None, :]) / torso[:, None, None]

    # Timing always reads angular speed, whatever SIMILARITY_METRIC compares —
    # "are we moving together" is a question about limb rotation rates, and
    # joint positions are the wrong units for it.
    angles = {slot: np.stack([angle_features(px, cfg.MIN_SEGMENT_PX)
                              for px in np.stack([raw[i][slot] for i in indices])])
              for slot in slots}
    speeds = {slot: speed_features(angles[slot], fps, cfg.TIMING_SMOOTH_FRAMES)
              for slot in slots}

    if metric == "angles":
        def posture_per_feature(a, b):
            return np.degrees(_angle_difference(a, b))
        posture_unit = "degrees"
    else:
        def posture_per_feature(a, b):
            return np.linalg.norm(a - b, axis=-1)
        posture_unit = "torso lengths"

    def timing_per_feature(a, b):
        # Speed *magnitude* only. Direction of change is choreography, not
        # timing: two dancers moving the same limb at the same rate in opposite
        # directions are in time with each other and in the wrong shape, and
        # that is the posture component's business.
        return np.abs(a - b)

    posture = _score_component(
        "posture", F, posture_per_feature, indices,
        weights=weights, tiers=tiers, names=names, unit=posture_unit,
        n_frames=n_frames, fps=fps, cfg=cfg, pair_labels=pair_labels,
        share_mode=(metric == "angles" and cfg.ANGLE_WEIGHT_MODE == "share"),
        tolerance_cfg=cfg.SIMILARITY_TOLERANCE,
        fallback=cfg.NULL_DISTANCE_FALLBACK[metric],
        angular=(metric == "angles"),
        match_window=cfg.POSTURE_MATCH_WINDOW,
        dead_zone=cfg.ANGLE_TOLERANCE if metric == "angles" else None,
    )
    timing = _score_component(
        "timing", speeds, timing_per_feature, indices,
        weights=weights, tiers=tiers, names=names, unit="deg/s",
        n_frames=n_frames, fps=fps, cfg=cfg, pair_labels=pair_labels,
        share_mode=(cfg.ANGLE_WEIGHT_MODE == "share"),
        tolerance_cfg=cfg.TIMING_TOLERANCE,
        fallback=cfg.NULL_DISTANCE_FALLBACK["timing"],
        angular=False,
    )

    # ── the summary score ────────────────────────────────────────────────────
    # One number for the panel, built from the two components as an explicit
    # weighted mean of their *percentages*. Blending the percentages rather than
    # the raw differences is what makes this legitimate: degrees and degrees per
    # second are not commensurable, but both components are already calibrated
    # against their own measured null, so both read 0 at chance and 100 at
    # identical. That shared scale is the only thing that makes them addable.
    comp_weights = np.array([cfg.SUMMARY_WEIGHTS[c.name] for c in (posture, timing)],
                            dtype=float)
    comp_weights = comp_weights / comp_weights.sum()
    posture.weight, timing.weight = float(comp_weights[0]), float(comp_weights[1])

    pairwise = posture.pairwise * posture.weight + timing.pairwise * timing.weight
    score = pairwise.mean(axis=1)

    analysis.angle_series = angles
    analysis.speed_series = speeds
    analysis.components = [posture, timing]
    analysis.null_distance = posture.null_distance
    analysis.tolerance = posture.tolerance
    analysis.noise_floor = posture.noise_floor
    analysis.feature_share = posture.share
    analysis.indices = indices
    analysis.distance = posture.distance
    analysis.pairwise = pairwise
    analysis.score = score
    analysis.series[indices] = score
    analysis.pairs_series[indices] = pairwise
    analysis.smoothed = _nan_smooth(
        _interpolate_gaps(analysis.series, cfg.SIMILARITY_INTERPOLATE_MAX_GAP),
        cfg.SIMILARITY_SMOOTH_FRAMES, cfg.SIMILARITY_SMOOTH_KERNEL,
    )

    # Per dancer: agreement with everyone else, which for three dancers is the
    # mean of the two pairs they appear in. This is inescapably relative — it
    # measures agreement with the others, not correctness — so if two dancers
    # drift the same way together, the third is the one that looks wrong.
    per_dancer = np.stack(
        [pairwise[:, [i for i, p in enumerate(pair_labels) if s in p]].mean(axis=1)
         for s in slots],
        axis=1,
    )
    analysis.dancer_series[indices] = per_dancer
    running = np.cumsum(per_dancer, axis=0) / np.arange(1, len(indices) + 1)[:, None]
    analysis.dancer_running[indices] = running
    # Carry the last running value across gaps, so the leaderboard does not
    # blank out on a frame the group score has no value for.
    for column in range(analysis.dancer_running.shape[1]):
        col = analysis.dancer_running[:, column]
        filled = np.nan
        for i in range(n_frames):
            if np.isfinite(col[i]):
                filled = col[i]
            elif np.isfinite(filled):
                col[i] = filled

    # Rank each component as well as the summary. On this clip the summary
    # ranking is a three-way tie, and that is the honest answer for one
    # number — but it conceals the actual finding, which is that the dancers
    # separate on posture and are indistinguishable on timing. A leaderboard
    # per axis says that; a single tied leaderboard does not.
    for comp in analysis.components:
        comp.ranking = _rank(
            comp.dancer_series[indices], slots,
            block=max(1, int(round(cfg.RANK_BOOTSTRAP_BLOCK_SECONDS * fps))),
            n_boot=cfg.RANK_BOOTSTRAP_SAMPLES, seed=cfg.RANK_BOOTSTRAP_SEED,
            tie_band=cfg.RANK_TIE_BAND,
        )

    _band_segments(analysis, cfg)

    analysis.ranking = _rank(
        per_dancer, slots,
        block=max(1, int(round(cfg.RANK_BOOTSTRAP_BLOCK_SECONDS * fps))),
        n_boot=cfg.RANK_BOOTSTRAP_SAMPLES, seed=cfg.RANK_BOOTSTRAP_SEED,
        tie_band=cfg.RANK_TIE_BAND,
    )
    return analysis
