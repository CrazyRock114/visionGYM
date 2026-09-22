"""Station 4: Burpee Broad Jumps (波比跳远) 动作动力学与裁判规则分析.

Hyrox 裁判核心规则：
1. 胸部贴地判定 (Chest to Deck)：伏地时胸部必须完全贴及地面或手掌平齐平面。
2. 双脚同步起跳判定 (Two-Foot Takeoff)：必须双脚同时起跳向前跃出，严禁单脚单腿跨步起跳。
3. 统计跳远跃出距离、俯卧下潜耗时、起跳滞空耗时与动作周期。
"""

from __future__ import annotations

import numpy as np
from ..rules import RepResult, StationAnalysis, ViolationType
from ..skeleton import extract_person_joints


def analyze_burpees(frames: list[dict], fps: float, n_frames: int, *,
                    cfg) -> StationAnalysis:
    """对视频序列进行波比跳远合规检测与动作周期分析."""
    analysis = StationAnalysis(
        station_id="4_burpees",
        station_name="Station 4: Burpee Broad Jumps (波比跳远)",
        fps=fps,
        total_frames=n_frames,
        duration=n_frames / fps if fps else 0.0,
        primary_metric_name="有效跳跃次数",
        primary_metric_unit="次",
    )

    if not frames:
        return analysis

    shoulder_y_series = []
    hip_y_series = []
    ankle_y_series = []
    ankle_x_series = []
    left_ankle_y = []
    right_ankle_y = []

    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            shoulder_y_series.append(0.3)
            hip_y_series.append(0.5)
            ankle_y_series.append(0.85)
            ankle_x_series.append(0.5)
            left_ankle_y.append(0.85)
            right_ankle_y.append(0.85)
            continue

        joints = extract_person_joints(persons[0])

        sy = (joints.get("left_shoulder", (0, 0.3))[1] + joints.get("right_shoulder", (0, 0.3))[1]) / 2.0
        hy = (joints.get("left_hip", (0, 0.5))[1] + joints.get("right_hip", (0, 0.5))[1]) / 2.0

        lay = joints.get("left_ankle", (0, 0.85))[1]
        ray = joints.get("right_ankle", (0, 0.85))[1]
        ay = (lay + ray) / 2.0

        lax = joints.get("left_ankle", (0.5, 0))[0]
        rax = joints.get("right_ankle", (0.5, 0))[0]
        ax = (lax + rax) / 2.0

        shoulder_y_series.append(sy)
        hip_y_series.append(hy)
        ankle_y_series.append(ay)
        ankle_x_series.append(ax)
        left_ankle_y.append(lay)
        right_ankle_y.append(ray)

    shoulder_y = np.array(shoulder_y_series)
    hip_y = np.array(hip_y_series)
    ankle_y = np.array(ankle_y_series)
    ankle_x = np.array(ankle_x_series)

    # 状态机:
    # 0: STANDING/PREPARE, 1: DROP_TO_CHEST, 2: CHEST_ON_GROUND, 3: PUSHUP_TO_SQUAT, 4: TAKEOFF_JUMP
    state = 0
    start_frame = 0
    chest_touch_frame = 0
    takeoff_frame = 0
    takeoff_x = 0.0
    min_chest_ground_gap = 1.0

    reps = []
    rep_num = 1

    for f in range(len(shoulder_y)):
        sy = shoulder_y[f]
        hy = hip_y[f]
        ay = ankle_y[f]
        # 胸地面垂直间隙 (越小越接近贴地，y向下增加)
        # 站立时 ay 约 0.85，sy 约 0.3，差距 0.55
        # 贴地时 ay 约 0.85，sy 约 0.80，差距 < 0.10
        gap = abs(ay - sy)

        if state == 0:  # 站立准备
            if sy > 0.45 and gap < 0.35:  # 开始俯卧
                state = 1
                start_frame = f
                min_chest_ground_gap = gap
                chest_touch_frame = f

        elif state == 1:  # 俯卧下潜中
            if gap < min_chest_ground_gap:
                min_chest_ground_gap = gap
                chest_touch_frame = f
            if gap <= cfg.BURPEE_CHEST_FLOOR_TOLERANCE or (sy >= ay - 0.08):
                # 判定达到胸部贴地阶段
                state = 2

        elif state == 2:  # 胸部贴地撑起
            if sy < ay - 0.20:  # 撑起并双腿向前收回
                state = 3

        elif state == 3:  # 收腿准备向前跳跃
            # 检测踝部离地/向前加速度或向上离地
            # 突然发生水平跃进
            if f > 2 and abs(ankle_x[f] - ankle_x[f - 2]) > 0.02:
                state = 4
                takeoff_frame = f
                takeoff_x = ankle_x[f]

        elif state == 4:  # 空中跳跃至落地
            # 检测落地：位移停止增长且双脚再度站稳
            if f - takeoff_frame >= int(0.2 * fps) and (f == len(shoulder_y) - 1 or abs(ankle_x[f] - ankle_x[f - 1]) < 0.005):
                end_frame = f
                landing_x = ankle_x[f]
                jump_distance = abs(landing_x - takeoff_x)
                duration = (end_frame - start_frame) / fps if fps else 0.0

                # 裁判规则核查：
                # 1. 胸部是否贴地达标
                chest_touch_passed = min_chest_ground_gap <= (cfg.BURPEE_CHEST_FLOOR_TOLERANCE + 0.03)

                # 2. 双脚起跳同步性检测 (检查起跳时刻左右脚踝离地时差)
                # 若左右脚 y 变化差异显著，判定为单脚跨步跳
                sync_passed = True
                if takeoff_frame > 2:
                    left_diff = abs(left_ankle_y[takeoff_frame] - left_ankle_y[takeoff_frame - 2])
                    right_diff = abs(right_ankle_y[takeoff_frame] - right_ankle_y[takeoff_frame - 2])
                    if abs(left_diff - right_diff) > 0.06:
                        sync_passed = False

                is_valid = chest_touch_passed and sync_passed
                reason = ViolationType.NONE
                if not chest_touch_passed:
                    reason = ViolationType.CHEST_NOT_TOUCHED
                elif not sync_passed:
                    reason = ViolationType.STEP_TAKEOFF

                rep = RepResult(
                    number=rep_num,
                    is_valid=is_valid,
                    reason=reason,
                    start_time=start_frame / fps,
                    peak_time=takeoff_frame / fps,
                    end_time=end_frame / fps,
                    duration=duration,
                    start_frame=start_frame,
                    peak_frame=takeoff_frame,
                    end_frame=end_frame,
                    metrics={
                        "jump_dist_norm": jump_distance,
                        "chest_gap": min_chest_ground_gap,
                        "pushup_duration": (takeoff_frame - start_frame) / fps,
                        "flight_duration": (end_frame - takeoff_frame) / fps,
                    }
                )
                reps.append(rep)
                rep_num += 1
                state = 0  # 重置

    analysis.reps = reps
    analysis.finalize()
    if reps:
        analysis.avg_cadence = (len(reps) / (analysis.duration / 60.0)) if analysis.duration > 0 else 0.0
    return analysis
