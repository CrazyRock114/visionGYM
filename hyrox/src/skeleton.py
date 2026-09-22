"""COCO-17 骨骼模型与生物力学关键点几何计算库.

支持 17 个关键点的拓扑连线、关节角度计算（膝角、髋角、肘角、躯干倾角）、
质心追踪与高品质抗锯齿骨架绘制。
"""

from __future__ import annotations

import math
import cv2
import numpy as np

# COCO-17 关键点索引
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KPT_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

FACE_KPTS = frozenset({0, 1, 2, 3, 4})

# 骨骼连线与所属肢体 (a, b, side)
SKELETON = [
    (15, 13, "L"), (13, 11, "L"), (16, 14, "R"), (14, 12, "R"),
    (11, 12, "T"), (5, 11, "L"), (6, 12, "R"), (5, 6, "T"),
    (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"),
    (0, 1, "H"), (0, 2, "H"), (1, 3, "H"), (2, 4, "H"), (3, 5, "H"), (4, 6, "H"),
]

# BGR 配色
SIDE_COLORS = {
    "L": (255, 180, 0),    # 亮青蓝 (左侧)
    "R": (0, 140, 255),    # 活力橙 (右侧)
    "T": (80, 220, 80),    # 翠绿 (躯干)
    "H": (200, 80, 220),   # 浅紫红 (面部)
}


def angle_between_points(p1: tuple[float, float], p2: tuple[float, float],
                         p3: tuple[float, float]) -> float:
    """计算以 p2 为顶点的内角 (度数, 0° ~ 180°).

    p1 -> p2 -> p3, 例如 (hip, knee, ankle) 构成膝关节角。
    完全伸直约为 180°，大腿与小腿垂直屈膝约为 90°。
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0

    cosine = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cosine))


def inclination_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """计算两点向量相对于垂直地心引力方向的夹角 (度数, 0° = 正垂直直立)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]   # 图像坐标系 y 向下
    rad = math.atan2(abs(dx), max(1e-6, abs(dy)))
    return math.degrees(rad)


def torso_driving_angle(shoulder: tuple[float, float], hip: tuple[float, float],
                        ankle: tuple[float, float]) -> float:
    """计算雪橇推/拉时的前倾动力线角度 (相对于水平地面的夹角)."""
    dx = shoulder[0] - ankle[0]
    dy = ankle[1] - shoulder[1]   # 取正高度差
    rad = math.atan2(dy, max(1e-6, abs(dx)))
    return math.degrees(rad)


def extract_person_joints(person: dict) -> dict[str, tuple[float, float]]:
    """提取一个人的全部关节归一化坐标字典 (0~1)."""
    kpts = person.get("kpts_xy", [])
    joints = {}
    for name, i in KPT_INDEX.items():
        if i < len(kpts):
            pt = kpts[i]
            if pt != (0, 0) and not (math.isnan(pt[0]) or math.isnan(pt[1])):
                joints[name] = (float(pt[0]), float(pt[1]))
    return joints


def draw_skeleton(img: np.ndarray, person: dict, width: int, height: int, *,
                  draw_face: bool = False, thickness: int = 3, radius: int = 5,
                  highlight_color: tuple[int, int, int] | None = None) -> np.ndarray:
    """在视频帧上高品质绘制 COCO-17 骨架与关节点."""
    pts = [(int(round(x * width)), int(round(y * height)))
           for x, y in person.get("kpts_xy", [])]

    def ok(i: int) -> bool:
        return 0 <= i < len(pts) and pts[i] != (0, 0)

    # 绘制肢体连线
    for a, b, side in SKELETON:
        if not draw_face and (a in FACE_KPTS or b in FACE_KPTS):
            continue
        if ok(a) and ok(b):
            color = highlight_color if highlight_color else SIDE_COLORS[side]
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)

    # 绘制关节点核心与发光外圈
    for i in range(len(pts)):
        if not draw_face and i in FACE_KPTS:
            continue
        if ok(i):
            cv2.circle(img, pts[i], radius + 2, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], max(1, radius - 2),
                       highlight_color if highlight_color else (47, 128, 237),
                       -1, cv2.LINE_AA)

    return img
