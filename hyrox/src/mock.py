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
            # 滑雪机冲程周期约 1.3 秒：过头高位引手 -> 腹肌收缩与深度屈髋下压 -> 髋后移两手摆至大腿后 -> 伸展起立复位
            cycle_t = (t % 1.3) / 1.3
            rep_idx = int(t / 1.3)
            # 驱动相 (0~0.55) 与回位恢复相 (0.55~1.0)
            if cycle_t < 0.55:
                p = math.sin((cycle_t / 0.55) * (math.pi / 2.0))
            else:
                p = math.cos(((cycle_t - 0.55) / 0.45) * (math.pi / 2.0))

            # 故意在第 2 次冲程注入动作过浅 (手臂未举过头顶、未屈髋)
            is_shallow = (rep_idx == 1)
            pull = p if not is_shallow else p * 0.45

            # 站立高位时双手高过头顶 (wy ≈ 0.12)，下拉到底部超过髋部 (wy ≈ 0.65)
            wy = 0.14 + pull * 0.52
            # 屈髋下压：下压时髋部后推 (hx 后移)，躯干前倾下压
            hx = center_x - pull * 0.07
            hy = hip_y_base + pull * 0.08
            sx = center_x - pull * 0.03
            sy = shoulder_y_base + pull * 0.14
            head_y = head_y_base + pull * 0.14

            # 膝关节微屈吸收冲力
            ky = knee_y_base + pull * 0.05
            kx = center_x - 0.02 - pull * 0.04

            kpts[0] = (sx, head_y)
            kpts[5] = (sx - 0.07, sy)
            kpts[6] = (sx + 0.07, sy)
            # 肘部关节
            kpts[7] = (sx - 0.10, sy + 0.08)
            kpts[8] = (sx + 0.10, sy + 0.08)
            # 双腕持握滑雪机手柄
            kpts[9] = (sx - 0.06, wy)
            kpts[10] = (sx + 0.06, wy)
            kpts[11] = (hx - 0.05, hy)
            kpts[12] = (hx + 0.05, hy)
            kpts[13] = (kx - 0.04, ky)
            kpts[14] = (kx + 0.04, ky)
            kpts[15] = (center_x - 0.06, ankle_y_base)
            kpts[16] = (center_x + 0.06, ankle_y_base)

        elif station_id == "2_sled_push":
            # 雪橇推：身体呈 45° 动力推进线，双手紧推前方把手，双腿高频活塞式蹬地
            step_phase = (t * 2.5) % 1.0  # 步频约 75 SPM
            leg_t = math.sin(step_phase * 2.0 * math.pi)

            # 前倾动力线：肩 (0.50, 0.40) -> 髋 (0.36, 0.55) -> 支撑踝 (0.24, 0.88)，夹角约 45°
            sx = center_x - 0.02
            sy = shoulder_y_base + 0.08
            hx = center_x - 0.16
            hy = hip_y_base + 0.02

            # 雪橇把手在前方 (0.62)
            sled_handle_x = center_x + 0.12
            sled_handle_y = sy + 0.04

            # 双腿交替高抬腿推进与蹬地
            l_kx = hx + 0.10 + leg_t * 0.08
            l_ky = knee_y_base - max(0.0, leg_t) * 0.12
            l_ax = hx + 0.06 + leg_t * 0.08
            l_ay = ankle_y_base - max(0.0, leg_t) * 0.06

            r_kx = hx + 0.10 - leg_t * 0.08
            r_ky = knee_y_base - max(0.0, -leg_t) * 0.12
            r_ax = hx + 0.06 - leg_t * 0.08
            r_ay = ankle_y_base - max(0.0, -leg_t) * 0.06

            kpts[0] = (sx + 0.04, sy - 0.07)
            kpts[5] = (sx - 0.04, sy)
            kpts[6] = (sx + 0.04, sy)
            kpts[7] = (sx + 0.06, sy + 0.02)
            kpts[8] = (sx + 0.06, sy + 0.02)
            # 手臂笔直抵紧雪橇杆
            kpts[9] = (sled_handle_x, sled_handle_y)
            kpts[10] = (sled_handle_x, sled_handle_y)
            kpts[11] = (hx - 0.04, hy)
            kpts[12] = (hx + 0.04, hy)
            kpts[13] = (l_kx, l_ky)
            kpts[14] = (r_kx, r_ky)
            kpts[15] = (l_ax, l_ay)
            kpts[16] = (r_ax, r_ay)

        elif station_id == "3_sled_pull":
            # 雪橇拉：低重心抗阻后倾坐姿，双手交替向后倒拉绳索，碎步后退
            step_phase = (t * 2.0) % 1.0  # 约 60 SPM
            arm_cycle = math.sin(t * 3.5)

            # 身体低重心后坐抗阻 (后仰角对抗绳索张力)
            hx = center_x - 0.14
            hy = hip_y_base + 0.09
            sx = center_x - 0.18
            sy = shoulder_y_base + 0.12

            # 双手交替倒手抓绳拉向腰肋
            # 左臂前伸抓绳 -> 后拉
            l_wx = center_x + 0.08 + arm_cycle * 0.12
            l_wy = sy + 0.02 + arm_cycle * 0.04
            # 右臂对相后拉 -> 前伸
            r_wx = center_x + 0.08 - arm_cycle * 0.12
            r_wy = sy + 0.02 - arm_cycle * 0.04

            # 屈膝低重心支撑步
            foot_shift = math.sin(step_phase * 2.0 * math.pi) * 0.04
            kpts[0] = (sx - 0.03, sy - 0.08)
            kpts[5] = (sx - 0.05, sy)
            kpts[6] = (sx + 0.05, sy)
            kpts[7] = ((sx + l_wx) / 2.0, sy + 0.06)
            kpts[8] = ((sx + r_wx) / 2.0, sy + 0.06)
            kpts[9] = (l_wx, l_wy)
            kpts[10] = (r_wx, r_wy)
            kpts[11] = (hx - 0.05, hy)
            kpts[12] = (hx + 0.05, hy)
            kpts[13] = (hx + 0.08, knee_y_base + 0.04)
            kpts[14] = (hx + 0.14, knee_y_base + 0.04)
            kpts[15] = (hx + 0.04 + foot_shift, ankle_y_base)
            kpts[16] = (hx + 0.18 - foot_shift, ankle_y_base)

        elif station_id == "5_rowing":
            # 划船机：真实水平坐姿滑轨冲程 (抓水 Catch -> 蹬腿伸髋 Drive -> 12°出水后仰 Finish -> 恢复 Recovery)
            cycle_t = (t % 1.8) / 1.8
            # 滑轨水平滑行动力学
            if cycle_t < 0.45:
                # 驱动蹬腿向后滑行
                drive = math.sin((cycle_t / 0.45) * (math.pi / 2.0))
            else:
                # 回桨向前滑行复位
                drive = math.cos(((cycle_t - 0.45) / 0.55) * (math.pi / 2.0))

            # 坐姿高度 (y ≈ 0.68 水平滑轨)
            seat_y = 0.68
            # 滑座水平位置：抓水前移到 0.48，出水后移到 0.28
            seat_x = center_x + 0.02 - drive * 0.18
            # 固定脚踏板在右前方
            foot_x = center_x + 0.22
            foot_y = 0.72

            # 躯干角度：抓水向前倾斜 15° (sx 在 seat_x 前)，出水向后仰 12° (sx 在 seat_x 后)
            torso_tilt = (1.0 - drive) * 0.08 - drive * 0.07
            sx = seat_x + torso_tilt
            sy = seat_y - 0.20

            # 屈膝角：抓水时紧凑屈膝 (膝关节抬高)，出水时双腿蹬直 (膝关节落平)
            kx = (seat_x + foot_x) / 2.0
            ky = seat_y - (1.0 - drive) * 0.15

            # 手腕拉动把手：抓水时向前伸至脚踏前 (0.28)，出水时拉回胸腹下沿 (0.10)
            handle_x = seat_x + 0.08 + (1.0 - drive) * 0.16
            handle_y = sy + 0.08

            kpts[0] = (sx + torso_tilt * 0.3, sy - 0.08)
            kpts[5] = (sx - 0.04, sy)
            kpts[6] = (sx + 0.04, sy)
            kpts[7] = ((sx + handle_x) / 2.0, sy + 0.06)
            kpts[8] = ((sx + handle_x) / 2.0, sy + 0.06)
            kpts[9] = (handle_x, handle_y)
            kpts[10] = (handle_x, handle_y)
            kpts[11] = (seat_x - 0.04, seat_y)
            kpts[12] = (seat_x + 0.04, seat_y)
            kpts[13] = (kx, ky)
            kpts[14] = (kx, ky)
            kpts[15] = (foot_x, foot_y)
            kpts[16] = (foot_x, foot_y)

        elif station_id == "6_farmers_carry":
            # 农夫行走：双手体侧持重下垂 (2x24kg/16kg)，躯干直立大步快走，微幅自然侧倾摆动
            walk_cycle = t * 3.2  # 约 95 步/分
            stride = math.sin(walk_cycle * 2.0 * math.pi)
            rep_idx = int(t / 2.5)

            # 故意在后程引入严重侧倾疲劳 (No-Rep 警示)
            is_fatigued = (rep_idx == 1)
            sway_amp = 0.045 if is_fatigued else 0.012
            sway = math.sin(walk_cycle * math.pi) * sway_amp
            shoulder_drop = 0.03 if is_fatigued else 0.0

            # 直立躯干
            sx = center_x + sway
            sy = shoulder_y_base
            hx = center_x + sway * 0.5
            hy = hip_y_base

            # 双手在身体两侧自然悬挂持重
            l_wx = sx - 0.12
            r_wx = sx + 0.12
            l_wy = hy + 0.10 + shoulder_drop
            r_wy = hy + 0.10 - shoulder_drop

            # 双腿交替行进步态
            l_foot_x = center_x - 0.06 + stride * 0.09
            r_foot_x = center_x + 0.06 - stride * 0.09
            l_lift = max(0.0, stride) * 0.05
            r_lift = max(0.0, -stride) * 0.05

            kpts[0] = (sx, head_y_base)
            kpts[5] = (sx - 0.08, sy + shoulder_drop)
            kpts[6] = (sx + 0.08, sy - shoulder_drop)
            kpts[7] = (l_wx, sy + 0.12)
            kpts[8] = (r_wx, sy + 0.12)
            kpts[9] = (l_wx, l_wy)
            kpts[10] = (r_wx, r_wy)
            kpts[11] = (hx - 0.05, hy)
            kpts[12] = (hx + 0.05, hy)
            kpts[13] = (l_foot_x, knee_y_base - l_lift)
            kpts[14] = (r_foot_x, knee_y_base - r_lift)
            kpts[15] = (l_foot_x, ankle_y_base - l_lift)
            kpts[16] = (r_foot_x, ankle_y_base - r_lift)

        else:
            # 8x1km 跑步区间：真实跑步步态，包含前倾驱动角、高抬膝摆腿、双脚离地滞空相与对向摆臂
            run_cycle = t * 2.9  # 步频约 174 SPM
            leg_t = math.sin(run_cycle * 2.0 * math.pi)

            # 跑步前倾角约 6°~8°
            sx = center_x + 0.03
            sy = shoulder_y_base - 0.01
            hx = center_x
            hy = hip_y_base - 0.01

            # 滞空相垂直浮动
            flight_bob = abs(math.cos(run_cycle * 2.0 * math.pi)) * 0.03

            # 左右对向大摆臂
            l_wx = sx - leg_t * 0.12
            l_wy = sy + 0.08 + leg_t * 0.06
            r_wx = sx + leg_t * 0.12
            r_wy = sy + 0.08 - leg_t * 0.06

            # 活塞式跑步摆腿
            l_kx = hx + 0.05 + leg_t * 0.11
            l_ky = knee_y_base - flight_bob - max(0.0, leg_t) * 0.12
            l_ax = hx + leg_t * 0.12
            l_ay = ankle_y_base - flight_bob - max(0.0, leg_t) * 0.08

            r_kx = hx + 0.05 - leg_t * 0.11
            r_ky = knee_y_base - flight_bob - max(0.0, -leg_t) * 0.12
            r_ax = hx - leg_t * 0.12
            r_ay = ankle_y_base - flight_bob - max(0.0, -leg_t) * 0.08

            kpts[0] = (sx + 0.02, head_y_base - flight_bob)
            kpts[5] = (sx - 0.07, sy - flight_bob)
            kpts[6] = (sx + 0.07, sy - flight_bob)
            kpts[7] = ((sx + l_wx) / 2.0, sy + 0.08)
            kpts[8] = ((sx + r_wx) / 2.0, sy + 0.08)
            kpts[9] = (l_wx, l_wy)
            kpts[10] = (r_wx, r_wy)
            kpts[11] = (hx - 0.05, hy - flight_bob)
            kpts[12] = (hx + 0.05, hy - flight_bob)
            kpts[13] = (l_kx, l_ky)
            kpts[14] = (r_kx, r_ky)
            kpts[15] = (l_ax, l_ay)
            kpts[16] = (r_ax, r_ay)

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
