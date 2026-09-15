"""COCO-17 skeleton definition and per-frame drawing."""

from __future__ import annotations

import cv2

# The 17 keypoints, in the order the model returns them.
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FACE_KPTS = frozenset({0, 1, 2, 3, 4})   # nose, eyes, ears

# (a, b, part); part picks the colour. Limbs are coloured by body part rather
# than by side, because on a wall the question is what the arms and the legs are
# doing, and a left/right split cuts across that.
SKELETON = [
    (5, 6, "torso"),
    (5, 7, "arms"), (7, 9, "arms"),
    (6, 8, "arms"), (8, 10, "arms"),
    (5, 11, "torso"), (6, 12, "torso"),
    (11, 12, "torso"),
    (11, 13, "legs"), (13, 15, "legs"),
    (12, 14, "legs"), (14, 16, "legs"),
    (0, 1, "head"), (0, 2, "head"), (1, 3, "head"), (2, 4, "head"),
    (3, 5, "head"), (4, 6, "head"),
]


def visible_parts(draw_face: bool):
    """The edges and keypoint indices to draw."""
    if draw_face:
        return SKELETON, frozenset(range(len(KPT_NAMES)))
    edges = [(a, b, p) for a, b, p in SKELETON if a not in FACE_KPTS and b not in FACE_KPTS]
    return edges, frozenset(i for i in range(len(KPT_NAMES)) if i not in FACE_KPTS)


def draw_person(img, kpts, valid, *, width: int, height: int, edges, points,
                colors: dict, thickness: int, radius: int,
                bbox=None, bbox_color=(0, 255, 255), bbox_thickness: int = 2):
    """Draw one body. Coordinates arrive normalized to [0, 1].

    They are normalized against the frame the model was shown, but that scaling
    is uniform, so multiplying by this frame's width and height is exact rather
    than approximate — one inference serves any render size.
    """
    pts = [(int(round(x * width)), int(round(y * height))) for x, y in kpts]

    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(img,
                      (int(bx * width), int(by * height)),
                      (int((bx + bw) * width), int((by + bh) * height)),
                      bbox_color, bbox_thickness, cv2.LINE_AA)

    for a, b, part in edges:
        if valid[a] and valid[b]:
            cv2.line(img, pts[a], pts[b], colors[part], thickness, cv2.LINE_AA)

    for i in points:
        if valid[i]:
            cv2.circle(img, pts[i], radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

    return img
