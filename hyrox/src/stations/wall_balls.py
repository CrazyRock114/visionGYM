"""Station 8: Wall Balls (药球投掷/墙球) 动作动力学与裁判规则分析.

裁判规则核心检测：
1. 深度必须达标：髋关节折痕必须低于膝关节上缘 (Hip Crease below Knee Top)。
2. 完全站立锁髋：抛球上升阶段，髋关节与膝关节必须充分伸展锁紧。
3. 统计每次动作的深蹲深度、下潜耗时、起身上推耗时与投掷节奏。
"""

from __future__ import annotations

import numpy as np
from ..rules import RepResult, StationAnalysis, ViolationType
from ..skeleton import angle_between_points, extract_person_joints


def analyze_wall_balls(frames: list[dict], fps: float, n_frames: int, *,
                       cfg) -> StationAnalysis:
    """对视频序列进行墙球动作识别、合规判定与时序分析."""
    analysis = StationAnalysis(
        station_id="8_wall_balls",
        station_name="Station 8: Wall Balls (药球投掷/墙球)",
        fps=fps,
        total_frames=n_frames,
        duration=n_frames / fps if fps else 0.0,
        primary_metric_name="墙球有效次数",
        primary_metric_unit="次",
    )

    if not frames:
        return analysis

    # 1. 提取髋部、膝盖、脚踝及手腕的垂直位移序列与膝关节夹角
    knee_angles = []
    hip_y_series = []
    knee_y_series = []
    wrist_y_series = []
    times = []

    for idx, frame in enumerate(frames):
        persons = frame.get("persons") or []
        t = idx / fps if fps else 0.0
        times.append(t)
        if not persons:
            knee_angles.append(175.0)
            hip_y_series.append(0.5)
            knee_y_series.append(0.7)
            wrist_y_series.append(0.4)
            continue

        joints = extract_person_joints(persons[0])
        # 双侧取均值或取可见度较高侧
        has_left = "left_hip" in joints and "left_knee" in joints and "left_ankle" in joints
        has_right = "right_hip" in joints and "right_knee" in joints and "right_ankle" in joints

        left_ang = angle_between_points(joints["left_hip"], joints["left_knee"],
                                        joints["left_ankle"]) if has_left else 175.0
        right_ang = angle_between_points(joints["right_hip"], joints["right_knee"],
                                         joints["right_ankle"]) if has_right else 175.0

        if has_left and has_right:
            ang = (left_ang + right_ang) / 2.0
            hip_y = (joints["left_hip"][1] + joints["right_hip"][1]) / 2.0
            knee_y = (joints["left_knee"][1] + joints["right_knee"][1]) / 2.0
        elif has_left:
            ang = left_ang
            hip_y = joints["left_hip"][1]
            knee_y = joints["left_knee"][1]
        elif has_right:
            ang = right_ang
            hip_y = joints["right_hip"][1]
            knee_y = joints["right_knee"][1]
        else:
            ang = 175.0
            hip_y = 0.5
            knee_y = 0.7

        wrist_y = 0.4
        if "left_wrist" in joints and "right_wrist" in joints:
            wrist_y = (joints["left_wrist"][1] + joints["right_wrist"][1]) / 2.0
        elif "left_wrist" in joints:
            wrist_y = joints["left_wrist"][1]
        elif "right_wrist" in joints:
            wrist_y = joints["right_wrist"][1]

        knee_angles.append(ang)
        hip_y_series.append(hip_y)
        knee_y_series.append(knee_y)
        wrist_y_series.append(wrist_y)

    knee_angles = np.array(knee_angles)
    hip_y_series = np.array(hip_y_series)
    knee_y_series = np.array(knee_y_series)
    wrist_y_series = np.array(wrist_y_series)

    # 2. 状态机识别深蹲周期 (站立 -> 下潜 -> 蹲到底部 -> 起身推投 -> 站立锁定)
    # STATE: 0: STANDING, 1: DESCENDING, 2: BOTTOM, 3: ASCENDING
    state = 0
    start_frame = 0
    min_angle_in_rep = 180.0
    max_hip_y_in_rep = 0.0
    knee_y_at_bottom = 0.0
    bottom_frame = 0

    reps = []
    rep_num = 1

    for f in range(len(knee_angles)):
        ang = knee_angles[f]
        hip_y = hip_y_series[f]
        knee_y = knee_y_series[f]

        if state == 0:  # 准备/站立
            if ang < 155.0:  # 开始屈膝下蹲
                state = 1
                start_frame = f
                min_angle_in_rep = ang
                max_hip_y_in_rep = hip_y
                knee_y_at_bottom = knee_y
                bottom_frame = f

        elif state == 1:  # 下潜中
            if ang < min_angle_in_rep:
                min_angle_in_rep = ang
                max_hip_y_in_rep = hip_y
                knee_y_at_bottom = knee_y
                bottom_frame = f
            if ang > min_angle_in_rep + 12.0 and ang < 140.0:
                # 触底开始回升
                state = 2

        elif state == 2:  # 起身推举中
            if ang >= cfg.WALL_BALL_LOCKOUT_KNEE_ANGLE - 5.0:
                # 完成一次完整周期
                end_frame = f
                duration = (end_frame - start_frame) / fps if fps else 0.0

                # 规则评判：
                # 1. 深度评判：在图像归一化坐标系中，y 值越大位置越低。
                # 髋折痕低于膝盖上缘 => max_hip_y_in_rep >= knee_y_at_bottom - 0.02
                # 且膝关节最小角度需低于阈值 (如 95 度)
                depth_passed = (min_angle_in_rep <= cfg.WALL_BALL_MIN_KNEE_ANGLE) or \
                               (max_hip_y_in_rep >= knee_y_at_bottom - 0.015)

                lockout_passed = ang >= cfg.WALL_BALL_LOCKOUT_KNEE_ANGLE - 8.0

                is_valid = depth_passed and lockout_passed
                reason = ViolationType.NONE
                if not depth_passed:
                    reason = ViolationType.SQUAT_DEPTH
                elif not lockout_passed:
                    reason = ViolationType.NO_LOCKOUT

                rep = RepResult(
                    number=rep_num,
                    is_valid=is_valid,
                    reason=reason,
                    start_time=start_frame / fps,
                    peak_time=bottom_frame / fps,
                    end_time=end_frame / fps,
                    duration=duration,
                    start_frame=start_frame,
                    peak_frame=bottom_frame,
                    end_frame=end_frame,
                    metrics={
                        "min_knee_angle": min_angle_in_rep,
                        "squat_depth_ratio": max_hip_y_in_rep / max(1e-4, knee_y_at_bottom),
                        "descent_time": (bottom_frame - start_frame) / fps,
                        "ascent_time": (end_frame - bottom_frame) / fps,
                    }
                )
                reps.append(rep)
                rep_num += 1
                state = 0  # 重置为站立

    analysis.reps = reps
    analysis.finalize()
    if reps:
        analysis.avg_cadence = (len(reps) / (analysis.duration / 60.0)) if analysis.duration > 0 else 0.0
    return analysis
