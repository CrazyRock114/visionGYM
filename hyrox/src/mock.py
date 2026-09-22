"""Hyrox 8 项运动生物力学姿态模拟生成器 (用于离线与快速演示验证).

能够合成符合人体解剖学与生物力学的真实连续 COCO-17 骨架轨迹，
并支持故意注入动作违规事件（如：第 2 次墙球深蹲浅、第 2 次波比单脚起跳等），
以全面测试裁判规则系统与侧边数据看板。
"""

from __future__ import annotations

import math
import numpy as np


def generate_mock_poses(station_id: str, n_frames: int = 210, fps: float = 30.0) -> list[dict]:
    """生成指定站点的真实合成姿态时序帧数据."""
    frames = []

    # 基础人体尺寸比例 (相对于画面高度)
    head_y_base = 0.22
    shoulder_y_base = 0.32
    hip_y_base = 0.55
    knee_y_base = 0.72
    ankle_y_base = 0.88
    center_x = 0.50

    for f in range(n_frames):
        t = f / fps
        kpts = [(0.0, 0.0)] * 17

        if station_id == "8_wall_balls":
            # 墙球周期约 2.0 秒：站立 -> 下蹲 -> 触底 -> 站立上推抛球
            cycle_t = (t % 2.0) / 2.0
            rep_idx = int(t / 2.0)
            # 制造动作节奏：前半周期下蹲，后半周期推起
            dip = math.sin(cycle_t * math.pi) if cycle_t < 1.0 else 0.0

            # 故意将第 2 次动作设为深蹲不足 (No-Rep)
            is_shallow = (rep_idx == 1)
            if is_shallow:
                dip *= 0.40  # 浅蹲

            # 运动学生物力学形变：下蹲时膝前移，髋后移下沉
            hx = center_x + dip * 0.03
            hy = hip_y_base + dip * 0.23
            kx = center_x - 0.08 - dip * 0.08
            ky = knee_y_base + dip * 0.05
            ax = center_x - 0.08
            ay = ankle_y_base

            # 手臂随下蹲在胸前，起身推举过头顶
            push = math.sin((cycle_t - 0.5) * math.pi) if cycle_t >= 0.5 else 0.0
            wy = (shoulder_y_base + 0.08) - max(0.0, push) * 0.28

            kpts[0] = (center_x, head_y_base + dip * 0.20)
            kpts[5] = (center_x - 0.06, shoulder_y_base + dip * 0.20)
            kpts[6] = (center_x + 0.06, shoulder_y_base + dip * 0.20)
            kpts[7] = (center_x - 0.12, shoulder_y_base + dip * 0.20 + 0.10)
            kpts[8] = (center_x + 0.12, shoulder_y_base + dip * 0.20 + 0.10)
            kpts[9] = (center_x - 0.06, wy)
            kpts[10] = (center_x + 0.06, wy)
            kpts[11] = (hx - 0.04, hy)
            kpts[12] = (hx + 0.04, hy)
            kpts[13] = (kx, ky)
            kpts[14] = (kx + 0.08, ky)
            kpts[15] = (ax, ay)
            kpts[16] = (ax + 0.08, ay)

        elif station_id == "4_burpees":
            # 波比周期约 2.2 秒：站立 -> 俯卧贴地 -> 撑起跳跃
            cycle_t = (t % 2.2) / 2.2
            rep_idx = int(t / 2.2)
            shift_x = rep_idx * 0.08

            if cycle_t < 0.25:  # 下潜
                p = cycle_t / 0.25
                sy = shoulder_y_base + p * 0.50
                hy = hip_y_base + p * 0.26
                ay = ankle_y_base
                kx = center_x + shift_x
            elif cycle_t < 0.50:  # 胸贴地
                sy = 0.81
                hy = 0.82
                ay = 0.84
                kx = center_x + shift_x
            elif cycle_t < 0.75:  # 起跳滞空
                p = (cycle_t - 0.50) / 0.25
                sy = 0.40 - math.sin(p * math.pi) * 0.16
                hy = 0.55 - math.sin(p * math.pi) * 0.16
                ay = ankle_y_base - math.sin(p * math.pi) * 0.16
                kx = center_x + shift_x + p * 0.07
            else:  # 站立平稳
                sy = shoulder_y_base
                hy = hip_y_base
                ay = ankle_y_base
                kx = center_x + shift_x + 0.07

            kpts[0] = (kx, sy - 0.08)
            kpts[5] = (kx - 0.07, sy)
            kpts[6] = (kx + 0.07, sy)
            kpts[9] = (kx - 0.08, ay - 0.02)
            kpts[10] = (kx + 0.08, ay - 0.02)
            kpts[11] = (kx - 0.06, hy)
            kpts[12] = (kx + 0.06, hy)
            kpts[13] = (kx - 0.07, (hy + ay) / 2)
            kpts[14] = (kx + 0.07, (hy + ay) / 2)
            kpts[15] = (kx - 0.06, ay)
            kpts[16] = (kx + 0.06, ay)

        elif station_id == "7_lunges":
            # 箭步蹲周期约 1.8 秒：左步 -> 起身 -> 右步 -> 起身
            cycle_t = (t % 1.8) / 1.8
            rep_idx = int(t / 1.8)
            dip = math.sin(cycle_t * math.pi)
            left_leads = (rep_idx % 2 == 0)

            # 故意在第 2 次动作制造后膝未触地 No-Rep
            if rep_idx == 1:
                dip *= 0.60

            hy = hip_y_base + dip * 0.16
            l_kx = (center_x - 0.08) + (dip * 0.08 if left_leads else -dip * 0.03)
            r_kx = (center_x + 0.08) + (-dip * 0.03 if left_leads else dip * 0.08)
            l_ky = knee_y_base + (dip * 0.04 if left_leads else dip * 0.16)
            r_ky = knee_y_base + (dip * 0.16 if left_leads else dip * 0.04)

            kpts[0] = (center_x, head_y_base + dip * 0.16)
            kpts[5] = (center_x - 0.08, shoulder_y_base + dip * 0.16)
            kpts[6] = (center_x + 0.08, shoulder_y_base + dip * 0.16)
            kpts[11] = (center_x - 0.06, hy)
            kpts[12] = (center_x + 0.06, hy)
            kpts[13] = (l_kx, l_ky)
            kpts[14] = (r_kx, r_ky)
            kpts[15] = (center_x - 0.08, ankle_y_base)
            kpts[16] = (center_x + 0.08, ankle_y_base)

        elif station_id == "1_skierg":
            # 滑雪机周期约 1.3 秒 (约 46 SPM)
            cycle_t = (t % 1.3) / 1.3
            pull = (1.0 - math.cos(cycle_t * 2.0 * math.pi)) / 2.0
            wy = 0.22 + pull * 0.48
            hy = hip_y_base + pull * 0.08
            sy = shoulder_y_base + pull * 0.12

            kpts[0] = (center_x, head_y_base + pull * 0.12)
            kpts[5] = (center_x - 0.08, sy)
            kpts[6] = (center_x + 0.08, sy)
            kpts[9] = (center_x - 0.06, wy)
            kpts[10] = (center_x + 0.06, wy)
            kpts[11] = (center_x - 0.06, hy)
            kpts[12] = (center_x + 0.06, hy)
            kpts[13] = (center_x - 0.07, knee_y_base + pull * 0.04)
            kpts[14] = (center_x + 0.07, knee_y_base + pull * 0.04)
            kpts[15] = (center_x - 0.07, ankle_y_base)
            kpts[16] = (center_x + 0.07, ankle_y_base)

        elif station_id == "5_rowing":
            # 划船机周期约 1.8 秒 (约 33 SPM)
            cycle_t = (t % 1.8) / 1.8
            drive = (1.0 - math.cos(cycle_t * 2.0 * math.pi)) / 2.0
            kx = center_x + (drive - 0.5) * 0.15
            wx = center_x + (drive - 0.5) * 0.20
            # 抓水时屈膝，驱动时伸腿
            ky = knee_y_base - (1.0 - drive) * 0.12

            kpts[0] = (kx, shoulder_y_base - 0.10)
            kpts[5] = (kx - 0.05, shoulder_y_base)
            kpts[6] = (kx + 0.05, shoulder_y_base)
            kpts[9] = (wx, shoulder_y_base + 0.10)
            kpts[10] = (wx, shoulder_y_base + 0.10)
            kpts[11] = (kx - 0.05, hip_y_base + 0.10)
            kpts[12] = (kx + 0.05, hip_y_base + 0.10)
            kpts[13] = (kx - 0.05, ky)
            kpts[14] = (kx + 0.05, ky)
            kpts[15] = (center_x + 0.15, ankle_y_base)
            kpts[16] = (center_x + 0.15, ankle_y_base)

        elif station_id == "6_farmers_carry":
            # 农夫行走：身体直立微幅侧倾摆动与步进
            sway = math.sin(t * 4.0) * 0.015
            step = math.sin(t * 8.0) * 0.03
            kpts[0] = (center_x + sway, head_y_base)
            kpts[5] = (center_x - 0.09 + sway, shoulder_y_base)
            kpts[6] = (center_x + 0.09 + sway, shoulder_y_base)
            kpts[9] = (center_x - 0.11 + sway, hip_y_base + 0.10)
            kpts[10] = (center_x + 0.11 + sway, hip_y_base + 0.10)
            kpts[11] = (center_x - 0.06, hip_y_base)
            kpts[12] = (center_x + 0.06, hip_y_base)
            kpts[13] = (center_x - 0.07, knee_y_base + step)
            kpts[14] = (center_x + 0.07, knee_y_base - step)
            kpts[15] = (center_x - 0.07, ankle_y_base + step)
            kpts[16] = (center_x + 0.07, ankle_y_base - step)

        elif station_id in ("2_sled_push", "3_sled_pull"):
            # 雪橇推/拉：身体 45° 动力倾角与高频交替步
            step = math.sin(t * 10.0) * 0.04
            sx = center_x - 0.12 if station_id == "2_sled_push" else center_x + 0.08
            kpts[0] = (sx - 0.05, shoulder_y_base - 0.05)
            kpts[5] = (sx, shoulder_y_base)
            kpts[6] = (sx + 0.04, shoulder_y_base)
            kpts[9] = (sx + 0.12, shoulder_y_base + 0.05)
            kpts[10] = (sx + 0.12, shoulder_y_base + 0.05)
            kpts[11] = (center_x - 0.04, hip_y_base)
            kpts[12] = (center_x + 0.04, hip_y_base)
            kpts[13] = (center_x - 0.06, knee_y_base + step)
            kpts[14] = (center_x + 0.06, knee_y_base - step)
            kpts[15] = (center_x - 0.08, ankle_y_base + step)
            kpts[16] = (center_x + 0.08, ankle_y_base - step)

        else:
            # 基础跑步步态 (Hyrox Run)
            step = math.sin(t * 11.0) * 0.06
            kpts[0] = (center_x, head_y_base)
            kpts[5] = (center_x - 0.08, shoulder_y_base)
            kpts[6] = (center_x + 0.08, shoulder_y_base)
            kpts[11] = (center_x - 0.06, hip_y_base)
            kpts[12] = (center_x + 0.06, hip_y_base)
            kpts[13] = (center_x - 0.07, knee_y_base + step)
            kpts[14] = (center_x + 0.07, knee_y_base - step)
            kpts[15] = (center_x - 0.07, ankle_y_base + step)
            kpts[16] = (center_x + 0.07, ankle_y_base - step)

        person = {
            "track_id": 0,
            "bbox_xywh": [center_x - 0.18, head_y_base - 0.05, 0.36, 0.78],
            "kpts_xy": kpts,
        }
        frames.append({
            "index": f,
            "frame_id": f,
            "persons": [person]
        })

    return frames
