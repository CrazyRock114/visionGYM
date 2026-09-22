"""Station 7: Sandbag Lunges (沙袋弓步蹲/箭步蹲) 动作动力学与裁判规则分析.

Hyrox 官方竞赛裁判规则：
1. 后膝必须触地 (Rear Knee Touchdown)：每一步下潜弓步，后侧膝盖必须轻触地面。
2. 起身完全直立锁紧 (Full Lockout)：在进入下一步之前，必须双脚靠拢并完全站立伸展双腿与髋部。
3. 左右腿交替出步与步态对称性分析。
"""

from __future__ import annotations

import numpy as np
from ..rules import RepResult, StationAnalysis, ViolationType
from ..skeleton import angle_between_points, extract_person_joints


def analyze_lunges(frames: list[dict], fps: float, n_frames: int, *,
                   cfg) -> StationAnalysis:
    """对视频序列进行沙袋弓步蹲动作识别、触地合规判罚与对称性分析."""
    analysis = StationAnalysis(
        station_id="7_lunges",
        station_name="Station 7: Sandbag Lunges (沙袋箭步蹲)",
        fps=fps,
        total_frames=n_frames,
        duration=n_frames / fps if fps else 0.0,
        primary_metric_name="有效弓步步数",
        primary_metric_unit="步",
    )

    if not frames:
        return analysis

    left_knee_ang = []
    right_knee_ang = []
    left_knee_y = []
    right_knee_y = []
    hip_y = []

    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            left_knee_ang.append(175.0)
            right_knee_ang.append(175.0)
            left_knee_y.append(0.7)
            right_knee_y.append(0.7)
            hip_y.append(0.5)
            continue

        joints = extract_person_joints(persons[0])

        has_l = "left_hip" in joints and "left_knee" in joints and "left_ankle" in joints
        has_r = "right_hip" in joints and "right_knee" in joints and "right_ankle" in joints

        lang = angle_between_points(joints["left_hip"], joints["left_knee"],
                                    joints["left_ankle"]) if has_l else 175.0
        rang = angle_between_points(joints["right_hip"], joints["right_knee"],
                                    joints["right_ankle"]) if has_r else 175.0

        lky = joints["left_knee"][1] if "left_knee" in joints else 0.7
        rky = joints["right_knee"][1] if "right_knee" in joints else 0.7
        hy = (joints.get("left_hip", (0, 0.5))[1] + joints.get("right_hip", (0, 0.5))[1]) / 2.0

        left_knee_ang.append(lang)
        right_knee_ang.append(rang)
        left_knee_y.append(lky)
        right_knee_y.append(rky)
        hip_y.append(hy)

    left_knee_ang = np.array(left_knee_ang)
    right_knee_ang = np.array(right_knee_ang)
    left_knee_y = np.array(left_knee_y)
    right_knee_y = np.array(right_knee_y)
    hip_y = np.array(hip_y)

    # 状态机：检测前后腿交替下潜
    state = 0  # 0: STANDING, 1: LUNGING_DOWN, 2: ASCENDING
    start_frame = 0
    bottom_frame = 0
    leading_leg = "left"
    max_rear_ky = 0.0

    reps = []
    rep_num = 1

    for f in range(len(hip_y)):
        l_ang = left_knee_ang[f]
        r_ang = right_knee_ang[f]
        hy = hip_y[f]

        if state == 0:
            if l_ang < 155.0 or r_ang < 155.0 or hy > 0.58:
                state = 1
                start_frame = f
                if l_ang < r_ang:
                    leading_leg = "left"
                else:
                    leading_leg = "right"
                rear_ky = right_knee_y[f] if leading_leg == "left" else left_knee_y[f]
                max_rear_ky = rear_ky
                bottom_frame = f

        elif state == 1:
            rear_ky = right_knee_y[f] if leading_leg == "left" else left_knee_y[f]
            if rear_ky > max_rear_ky:
                max_rear_ky = rear_ky
                bottom_frame = f
            # 下沉到底部后开始回升
            if rear_ky < max_rear_ky - 0.035:
                state = 2

        elif state == 2:
            duration = (f - start_frame) / fps if fps else 0.0
            # 站立伸展锁紧 (双膝均回弹 > 150° 且动作时长 > 0.6s)
            if duration >= 0.6 and l_ang >= cfg.LUNGE_LOCKOUT_ANGLE - 12.0 and r_ang >= cfg.LUNGE_LOCKOUT_ANGLE - 12.0:
                end_frame = f

                # 规则核查：
                # 1. 后膝是否触地:
                knee_touched = max_rear_ky >= 0.84  # 后膝触地深度判定 (y 达到 0.84 以上)

                # 2. 站立锁髋完全伸直:
                lockout_passed = ((l_ang + r_ang) / 2.0 >= 155.0 and min(l_ang, r_ang) >= 148.0)

                is_valid = knee_touched and lockout_passed
                reason = ViolationType.NONE
                if not knee_touched:
                    reason = ViolationType.REAR_KNEE_NOT_TOUCHED
                elif not lockout_passed:
                    reason = ViolationType.INCOMPLETE_EXTENSION

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
                        "leading_leg": leading_leg,
                        "front_knee_angle": left_knee_ang[bottom_frame] if leading_leg == "left" else right_knee_ang[bottom_frame],
                        "rear_knee_depth": max_rear_ky,
                        "lockout_angle": (l_ang + r_ang) / 2.0,
                    }
                )
                reps.append(rep)
                rep_num += 1
                state = 0

    analysis.reps = reps
    analysis.finalize()
    if reps:
        analysis.avg_cadence = (len(reps) / (analysis.duration / 60.0)) if analysis.duration > 0 else 0.0
        left_steps = sum(1 for r in reps if r.metrics.get("leading_leg") == "left")
        right_steps = sum(1 for r in reps if r.metrics.get("leading_leg") == "right")
        analysis.extra_data["left_right_ratio"] = f"左腿 {left_steps} 步 / 右腿 {right_steps} 步"
    return analysis
