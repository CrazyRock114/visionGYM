"""The route: SAM 3.1 segmentation of the holds, agreed across sampled frames.

One call segments one frame, and one frame is not the route. The climber covers
a different part of the wall in every one of them, and SAM's own mask edge moves
a pixel or two between looks at the same hold. So the route is built from a
handful of evenly-spaced frames and then reduced:

  * instances are clustered across calls by IoU — one cluster per real hold
  * a cluster is kept only if it appears in enough of the samples, which is what
    separates a hold that was occluded twice from a one-frame false positive
  * the surviving masks are pixel-voted into one shape, so the drawn outline is
    the agreement between samples rather than any single frame's noisy edge

SAM 3.1 returns one *label map* per frame rather than a mask per instance: a
single 8-bit PNG whose pixel value is the instance's own `instance_id`, 0 where
nothing matched. So a frame's masks are one decode, and an instance's mask is
`labels == instance_id` — see `decode_label_map`.

What comes back is one dict per hold, everything normalized to [0, 1]:

    {"bbox": [x, y, w, h], "polygon": [[x, y], ...], "score": float,
     "appearances": int, "appearance_fraction": float, "id": int}

`id` is assigned bottom-to-top, so hold 1 is the start and the last id is the top.
"""

from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

CONTENT_OBJECT_MASKS = ("img.segment.masks", "vid.segment.masks")

# Bumped when the request or the response shape this module reads changes, so a
# cache written against the old one is a miss rather than a confusing crash.
CONTRACT = "label-map-v1"

MASK_FORMAT = "png"   # the gateway's default; sent explicitly so it is on record


# ── the wire format ──────────────────────────────────────────────────────────

@dataclass
class FrameMasks:
    """One frame's segmentation: the instance rows, and the label map they index into.

    They are kept together because that is what the contract is now — a row
    carries no pixels of its own, only the `instance_id` naming its value in the
    frame's single label map — and separating them loses the join.
    """

    items: list[dict] = field(default_factory=list)
    mask: dict | None = None      # the label map, still PNG-encoded


def decode_label_map(mask: dict | None) -> np.ndarray | None:
    """Decode a label map into a uint8 array: pixel = ``instance_id``, 0 = no object.

    One 8-bit PNG carries every instance in the frame, so a frame costs one
    decode however many holds are on the wall. `data` is a `data:` URI; only the
    part after the comma is base64.
    """
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    data = mask.get("data")
    if not isinstance(data, str):
        return None
    raw = base64.b64decode(data.split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    if labels.ndim == 3:   # a grayscale PNG some decoders hand back as 3 channels
        labels = labels[:, :, 0]
    return labels.astype(np.uint8)


def instance_mask(labels: np.ndarray | None, item: dict) -> np.ndarray | None:
    """The boolean mask of one instance, cut out of its frame's label map."""
    if labels is None:
        return None
    instance_id = item.get("instance_id")
    if instance_id is None:
        return None
    mask = labels == int(instance_id)
    return mask if mask.any() else None


def unwrap(payload: dict) -> FrameMasks:
    """Pull the instances and the label map out of the response envelope.

    The envelope is flat — `model`, the input's details, then `content` — so the
    payload is read straight off the reply. The tag is checked rather than
    assumed: a wrong `content.object` means the request went somewhere other than
    image segmentation, which is clearer to say here than to let it surface as a
    KeyError three functions later.
    """
    content = payload.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}")

    got = content.get("object")
    if got not in CONTENT_OBJECT_MASKS:
        raise RuntimeError(
            f"Expected a segmentation payload ({' or '.join(CONTENT_OBJECT_MASKS)}), "
            f"got {got!r}."
        )
    return FrameMasks(items=content.get("items") or [], mask=content.get("mask"))


def request_masks(client, *, model: str, image_b64: str, prompt: str) -> tuple[dict, dict | None]:
    """One blocking call: segment everything matching *prompt* in one frame."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }],
        }],
        response_format={"type": "json_object"},
        extra_body={"method": "segment",
                    "method_params": {"prompt": prompt, "mask_format": MASK_FORMAT}},
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


# ── sampling the clip ────────────────────────────────────────────────────────

def sample_frames(video: Path, n: int) -> list[tuple[int, np.ndarray]]:
    """*n* evenly-spaced frames, as BGR arrays, with their source indices.

    Spaced across the whole clip rather than taken from the start: the point is
    to see each hold with the climber somewhere else, and at the start they are
    all still on the ground.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"{video} reports no frames")

    # Inset from both ends: the first and last frames of a handheld start/stop
    # are the ones most likely to be blurred or half-covered.
    indices = np.linspace(total * 0.05, total * 0.95, num=min(n, total)).astype(int)
    out: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            out.append((int(idx), frame))
    cap.release()
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed on a sampled frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def segment_frames(client, frames, *, model: str, prompt: str, min_score: float,
                   workers: int, console=None) -> tuple[list[FrameMasks], list[dict]]:
    """Segment every sampled frame, in parallel. Returns (per-frame results, usages).

    The calls are independent, so they go out together — twelve frames one after
    another is twelve round trips of latency for no reason. A frame whose call
    fails contributes nothing rather than failing the run: the consensus step
    below is built to work from however many samples actually came back.

    The score filter drops rows only; the frame's label map is kept whole, since
    it is one image for every instance and the ids of the dropped ones are simply
    never looked up.
    """
    def one(item):
        index, frame = item
        payload, usage = request_masks(
            client, model=model, image_b64=encode_jpeg(frame), prompt=prompt)
        result = unwrap(payload)
        result.items = [i for i in result.items
                        if float(i.get("score") or 0.0) >= min_score]
        return index, result, usage

    results: list[tuple[int, FrameMasks, dict | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, result, usage in pool.map(one, frames):
            results.append((index, result, usage))
            if console:
                console.print(f"  frame [dim]{index}[/]: {len(result.items)} instances")

    results.sort(key=lambda r: r[0])
    return [r[1] for r in results], [r[2] for r in results if r[2]]


# ── consensus ────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    """Intersection over union for two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _cluster(per_frame: list[FrameMasks], *, iou_threshold: float) -> list[dict]:
    """Greedy IoU clustering of instances across calls. One cluster per hold.

    Each instance is matched against the *running mean* box of each cluster
    rather than against any one member, so a cluster's identity settles as it
    gathers members instead of being pinned to whichever frame happened first.

    A member is kept as ``(call_index, instance)``: the instance's pixels live in
    *its own frame's* label map, so which call it came from is part of its
    identity now rather than incidental bookkeeping.
    """
    clusters: list[dict] = []
    for call_idx, result in enumerate(per_frame):
        for inst in result.items:
            box = inst.get("bbox_xywh")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            box = [float(v) for v in box]

            best, best_iou = -1, iou_threshold
            for ci, cluster in enumerate(clusters):
                score = _iou(box, np.mean(cluster["boxes"], axis=0).tolist())
                if score > best_iou:
                    best, best_iou = ci, score

            if best >= 0:
                clusters[best]["boxes"].append(box)
                clusters[best]["instances"].append((call_idx, inst))
                clusters[best]["calls"].add(call_idx)
            else:
                clusters.append({"boxes": [box], "instances": [(call_idx, inst)],
                                 "calls": {call_idx}})
    return clusters


def _vote_masks(masks: list[np.ndarray], boxes: list[list[float]], *, raster_size: int,
                vote_fraction: float, min_area_px: int, fill_holes: bool):
    """Pixel-vote one cluster's masks into a single normalized polygon.

    The vote happens in a grid over the cluster's own bounding box, not over the
    whole frame: a hold is a couple of percent of the image, so a frame-sized
    grid would resolve it into a dozen pixels. Locally, the same raster budget
    gives the shape a few hundred across.
    """
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes)
    ys1 = max(b[1] + b[3] for b in boxes)

    pad = 0.01
    x0, y0 = max(0.0, xs0 - pad), max(0.0, ys0 - pad)
    x1, y1 = min(1.0, xs1 + pad), min(1.0, ys1 + pad)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    long_side = max(w, h)
    W = max(8, int(round(raster_size * w / long_side)))
    H = max(8, int(round(raster_size * h / long_side)))

    accum = np.zeros((H, W), dtype=np.int32)
    voters = 0
    for mask in masks:
        mh, mw = mask.shape
        # Crop the full-frame mask to this cluster's window, in that mask's own
        # pixels, then resample to the shared grid — so every sample lands in
        # the same coordinate frame regardless of what resolution it came at.
        cx0, cx1 = int(round(x0 * mw)), int(round(x1 * mw))
        cy0, cy1 = int(round(y0 * mh)), int(round(y1 * mh))
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(mw, max(cx1, cx0 + 1)), min(mh, max(cy1, cy0 + 1))
        crop = mask[cy0:cy1, cx0:cx1].astype(np.uint8)
        if crop.size == 0:
            continue
        accum += cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        voters += 1

    if voters == 0:
        return None

    threshold = max(1, int(np.ceil(voters * vote_fraction)))
    binary = (accum >= threshold).astype(np.uint8)
    if not binary.any():
        return None

    labeled, n = ndimage.label(binary)
    if n == 0:
        return None
    areas = ndimage.sum(binary, labeled, index=range(1, n + 1)).astype(int)
    largest = int(np.argmax(areas)) + 1
    if int(areas[largest - 1]) < min_area_px:
        return None

    cleaned = labeled == largest
    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)
    cleaned = cleaned.astype(np.uint8) * 255

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    eps = max(0.5, 0.005 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = contour.reshape(-1, 2)
    if len(approx) < 3:
        return None

    polygon = [[float(px) / max(W - 1, 1) * w + x0,
                float(py) / max(H - 1, 1) * h + y0] for px, py in approx]

    arr = np.asarray(polygon, dtype=float)
    px0, py0 = arr.min(axis=0)
    px1, py1 = arr.max(axis=0)
    bbox = [float(px0), float(py0), float(px1 - px0), float(py1 - py0)]
    return polygon, bbox


def consensus(per_frame: list[FrameMasks], *, iou_threshold: float,
              min_appearance: float, raster_size: int, vote_fraction: float,
              min_area_px: int, fill_holes: bool) -> list[dict]:
    """Cluster, filter by appearance rate, and vote each survivor into one shape.

    Each frame's label map is decoded once, up front, rather than per instance:
    one PNG holds every hold in that frame, so decoding it per cluster member
    would decode the same image as many times as there are holds on the wall.
    """
    n_calls = len([f for f in per_frame if f is not None])
    if n_calls == 0:
        return []

    label_maps = [decode_label_map(f.mask) if f is not None else None for f in per_frame]

    holds = []
    for cluster in _cluster(per_frame, iou_threshold=iou_threshold):
        appearances = len(cluster["calls"])
        fraction = appearances / n_calls
        if fraction < min_appearance:
            continue

        masks = [m for m in (instance_mask(label_maps[call_idx], inst)
                             for call_idx, inst in cluster["instances"]) if m is not None]
        voted = _vote_masks(
            masks, cluster["boxes"], raster_size=raster_size,
            vote_fraction=vote_fraction, min_area_px=min_area_px, fill_holes=fill_holes)
        if voted is None:
            # No usable mask agreement; fall back to the mean box so the hold is
            # not silently lost. Touch tests degrade to the box + its margin.
            polygon, bbox = [], np.mean(cluster["boxes"], axis=0).tolist()
        else:
            polygon, bbox = voted

        scores = [float(i.get("score") or 0.0) for _, i in cluster["instances"]]
        holds.append({
            "bbox": [float(v) for v in bbox],
            "polygon": polygon,
            "score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "appearances": appearances,
            "appearance_fraction": round(fraction, 4),
        })
    return holds


def _area(box) -> float:
    return float(box[2] * box[3])


def _containment(inner, outer) -> float:
    """Fraction of *inner*'s area that lies inside *outer*. Normalized xywh."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = aw * ah
    return float(inter / area) if area > 1e-9 else 0.0


def suppress_contained(holds: list[dict], *, max_containment: float | None
                       ) -> tuple[list[dict], list[tuple[dict, dict, float]]]:
    """Drop a hold that sits mostly inside a better one. ``(kept, dropped)``.

    `consensus` clusters by IoU, which is symmetric and therefore blind to a
    small box nested in a large one: a blob covering a fifth of a hold's area
    scores about 0.28 against it — under HOLD_IOU_THRESHOLD — and survives as a
    hold in its own right. Containment is the asymmetric test that catches it.
    It asks how much of the *smaller* box is inside the larger, which is ~95%
    for a knob segmented off its own hold and near zero for two holds merely set
    close together, so it separates the two cases IoU cannot.

    This matters beyond one spurious number on one wall. Ids are assigned by
    position, so a phantom hold shifts every id above it and that clip's
    numbering stops agreeing with its siblings' — see `compare.build`.

    Ranked by score, then area: the survivor is the detection SAM was surest of
    and, failing that, the whole hold rather than the piece of it. Each dropped
    entry is returned with the hold that swallowed it and by how much, because a
    suppression is a judgement about the wall and should be visible.
    """
    if not holds or max_containment is None:
        return list(holds), []

    order = sorted(range(len(holds)),
                   key=lambda i: (-float(holds[i].get("score") or 0.0),
                                  -_area(holds[i]["bbox"])))
    kept: list[int] = []
    dropped: list[tuple[dict, dict, float]] = []
    for i in order:
        box = holds[i]["bbox"]
        covered = max(((j, _containment(box, holds[j]["bbox"])) for j in kept),
                      key=lambda p: p[1], default=None)
        if covered is not None and covered[1] >= max_containment:
            dropped.append((holds[i], holds[covered[0]], covered[1]))
        else:
            kept.append(i)
    return [holds[i] for i in sorted(kept)], dropped


def route_lean(holds: list[dict]) -> float:
    """``dx/dy`` over the hold centroids. Negative means the route leans right.

    Image coordinates put y=0 at the top, so climbing is y *decreasing*: a
    negative slope means x grows as the climber rises, i.e. a route running from
    the bottom-left to the top-right.
    """
    if len(holds) < 3:
        return 0.0
    xs = np.array([h["bbox"][0] + h["bbox"][2] / 2 for h in holds])
    ys = np.array([h["bbox"][1] + h["bbox"][3] / 2 for h in holds])
    variance = float(np.var(ys))
    if variance < 1e-9:
        return 0.0
    return float(np.cov(xs, ys)[0, 1] / variance)


def assign_ids(holds: list[dict], *, band: float = 0.025,
               direction: str = "auto") -> tuple[list[dict], str]:
    """Number the holds bottom-to-top, ordering same-height holds along the route.

    The numbering has to be a property of the *wall*, not of the climber, or two
    people's runs cannot be compared hold for hold. Sorting on height alone is
    already climber-independent but it is not stable: on a real route several
    holds sit within a few thousandths of the same height — on this wall two of
    them are 0.0008 apart — and which one comes first is then decided by mask
    noise, so a re-run can silently renumber the route.

    So holds within *band* of each other in height are treated as one row and
    ordered left to right along the route's own lean. A route running
    bottom-left to top-right reads left to right; a mirrored route reads right
    to left, so hold 1 is the first hold of the climb either way.

    Returns ``(holds, direction)`` with the direction actually used.
    """
    if direction == "auto":
        # A route with no lean to speak of still needs a rule; left-to-right is
        # the arbitrary-but-fixed default, and it is recorded in run.json.
        direction = "rtl" if route_lean(holds) > 0 else "ltr"
    if direction not in ("ltr", "rtl"):
        raise ValueError(f"direction must be 'auto', 'ltr' or 'rtl' (got {direction!r})")
    sign = 1.0 if direction == "ltr" else -1.0

    def centre(hold):
        return (hold["bbox"][0] + hold["bbox"][2] / 2,
                hold["bbox"][1] + hold["bbox"][3] / 2)

    # Greedy rows from the ground up: take the lowest hold not yet placed, sweep
    # in everything within `band` of it, order that row along the lean. Banding
    # against the row's own base rather than against a fixed grid keeps a row
    # from chaining upward across the whole wall one hold at a time.
    remaining = sorted(holds, key=lambda h: -centre(h)[1])
    ordered: list[dict] = []
    while remaining:
        base_y = centre(remaining[0])[1]
        row = [h for h in remaining if base_y - centre(h)[1] <= band]
        row.sort(key=lambda h: sign * centre(h)[0])
        ordered.extend(row)
        remaining = [h for h in remaining if h not in row]

    for i, hold in enumerate(ordered, start=1):
        hold["id"] = i
    holds[:] = ordered
    return holds, direction


# ── the wall, as a region ────────────────────────────────────────────────────

def wall_mask(holds: list[dict], *, res: int, dilate: float) -> np.ndarray | None:
    """A binary mask of the route's actual footprint, dilated.

    A rectangle around the holds is the wrong shape for this: a route runs on a
    diagonal, so its bounding box is mostly the empty triangle beside it, and
    somebody standing in that triangle would score as "on the wall" exactly like
    the climber. The real hold pixels do not have that problem. The dilation is
    what lets a torso spanning the gap between two holds still register.
    """
    if not holds:
        return None

    mask = np.zeros((res, res), dtype=np.uint8)
    for hold in holds:
        polygon = hold.get("polygon")
        if polygon and len(polygon) >= 3:
            pts = np.asarray([[p[0] * (res - 1), p[1] * (res - 1)] for p in polygon],
                             dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        else:
            bx, by, bw, bh = hold["bbox"]
            cv2.rectangle(mask,
                          (int(bx * (res - 1)), int(by * (res - 1))),
                          (int((bx + bw) * (res - 1)), int((by + bh) * (res - 1))),
                          1, thickness=-1)

    px = max(0, int(round(dilate * res)))
    if px:
        mask = cv2.dilate(mask, np.ones((2 * px + 1, 2 * px + 1), np.uint8))
    return mask if mask.any() else None


def wall_overlap(bbox_xywh, mask: np.ndarray) -> float:
    """Fraction of a normalized ``[x, y, w, h]`` box that sits on wall pixels."""
    if mask is None:
        return 0.0
    res = mask.shape[0]
    x, y, w, h = bbox_xywh
    x0 = max(0, int(round(x * (res - 1))))
    y0 = max(0, int(round(y * (res - 1))))
    x1 = min(res, int(round((x + w) * (res - 1))) + 1)
    y1 = min(res, int(round((y + h) * (res - 1))) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = mask[y0:y1, x0:x1]
    return float(region.sum()) / float(region.size) if region.size else 0.0


def filter_by_pose_region(holds: list[dict], points: np.ndarray, *,
                          margin: float) -> tuple[list[dict], int]:
    """Drop holds outside the convex hull of where the climber's body went.

    The prompt asks for a colour, and a colour is not a route: the same orange
    appears on the wall next door and above the finish. What separates this
    route from the rest of the orange in the frame is that the climber's body
    passed over it, so the body's own trajectory is the filter.
    """
    if len(points) < 3:
        return holds, 0

    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return holds, 0

    # Rasterize the hull and dilate it, rather than scaling the vertices about
    # the centroid: a scale moves a far vertex further than a near one, so the
    # margin would mean a different distance on every edge — and on a short clip,
    # where the hull is small, it would barely move the boundary at all.
    res = 512
    verts = np.round(points[hull.vertices] * (res - 1)).astype(np.int32)
    grid = np.zeros((res, res), dtype=np.uint8)
    cv2.fillPoly(grid, [verts.reshape(-1, 1, 2)], 1)
    px = max(1, int(round(margin * res)))
    grid = cv2.dilate(grid, np.ones((2 * px + 1, 2 * px + 1), np.uint8))

    kept, dropped = [], 0
    for hold in holds:
        x, y, w, h = hold["bbox"]
        cx = min(res - 1, max(0, int(round((x + w / 2) * (res - 1)))))
        cy = min(res - 1, max(0, int(round((y + h / 2) * (res - 1)))))
        if grid[cy, cx]:
            kept.append(hold)
        else:
            dropped += 1

    # Dropping everything means the filter, not the route, is wrong — a clip
    # where the pose failed should not silently produce an empty wall.
    return (kept, dropped) if kept else (holds, 0)


# ── cache ────────────────────────────────────────────────────────────────────

def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str,
               settings: dict) -> Path:
    """Where this exact route lives.

    Keyed on the clip's identity *and* every setting that shapes the result, so
    a cache hit can only mean "same clip, same prompt, same consensus rules".
    """
    stat = video.stat()
    key = json.dumps({
        "video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "model": model, "prompt": prompt, "settings": settings, "contract": CONTRACT,
    }, sort_keys=True)
    return cache_dir / f"holds.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, holds: list[dict], raw: list[FrameMasks],
               usages: list[dict], *, stamp: str) -> None:
    """Write the voted route, plus the per-frame instance rows behind it.

    The label maps are deliberately *not* stored: the consensus they fed is
    already in `holds`, and a full-frame PNG per sample would grow the file by an
    order of magnitude to serve nothing but the instance counts printed on a
    cache hit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp, "holds": holds,
        "raw": [{"items": f.items} for f in raw], "usages": usages}))


def load_cache(path: Path):
    """Read cached holds, or None if missing or unreadable."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        raw = [FrameMasks(items=f.get("items") or []) for f in blob.get("raw") or []]
        return blob["holds"], raw, blob.get("usages") or [], \
            blob.get("created", "unknown")
    except (json.JSONDecodeError, AttributeError, KeyError, OSError):
        return None
