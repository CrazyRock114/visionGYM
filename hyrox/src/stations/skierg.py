"""Station 1: SkiErg (滑雪机 1000m) 冲程时序与三屈三伸动作动力学分析.

检测与分析指标：
1. 冲程时序与冲程频次 (SPM - Strokes Per Minute)。
2. 高位引手到低位拉桨行程幅度 (Amplitude)，防止短浅浅拉。
3. 屈髋发力角 (Hip Hinge Angle) 与身体展开还原度。
4. 拉桨耗时 (Drive Time) 与 回桨耗时 (Recovery Time) 比例。
"""

from __future__ import annotations

import numpy as np
from ..rules import RepResult, StationAnalysis, ViolationType
from ..skeleton import angle_between_points, extract_person_joints


def analyze_skierg(frames: list[dict], fps: float, n_frames: int, *,
                   cfg) -> StationAnalysis:
    """对视频序列进行滑雪机冲程循环识别、桨频与发力动力学分析."""
    analysis = StationAnalysis(
        station_id="1_skierg",
        station_name="Station 1: 1000m SkiErg (滑雪机)",
        fps=fps,
        total_frames=n_frames,
        duration=n_frames / fps if fps else 0.0,
        primary_metric_name="划桨冲程数",
        primary_metric_unit="桨",
    )

    if not frames:
        return analysis

    wrist_y_series = []
    hip_angle_series = []

    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            wrist_y_series.append(0.5)
            hip_angle_series.append(170.0)
            continue

        joints = extract_person_joints(persons[0])

        wy = (joints.get("left_wrist", (0, 0.5))[1] + joints.get("right_wrist", (0, 0.5))[1]) / 2.0
        wrist_y_series.append(wy)

        # 髋关节角 (肩-髋-膝)
        has_l = "left_shoulder" in joints and "left_hip" in joints and "left_knee" in joints
        has_r = "right_shoulder" in joints and "right_hip" in joints and "right_knee" in joints
        lang = angle_between_points(joints["left_shoulder"], joints["left_hip"], joints["left_knee"]) if has_l else 170.0
        rang = angle_between_points(joints["right_shoulder"], joints["right_hip"], joints["right_knee"]) if has_r else 170.0
        hip_angle_series.append((lang + rang) / 2.0)

    wrist_y = np.array(wrist_y_series)
    hip_angles = np.array(hip_angle_series)

    # 状态机：高举引手 (y小) -> 下拉 (y变大) -> 底部 (y最大) -> 回弹 (y变小)
    state = 0  # 0: TOP_REACH, 1: DRIVING_DOWN, 2: RECOVERING
    start_frame = 0
    bottom_frame = 0
    top_y = 1.0
    bottom_y = 0.0

    reps = []
    rep_num = 1

    for f in range(len(wrist_y)):
        wy = wrist_y[f]

        if state == 0:
            if wy < top_y:
                top_y = wy
            # 手腕开始明显下拉
            if wy > top_y + 0.12:
                state = 1
                start_frame = f
                bottom_y = wy
                bottom_frame = f

        elif state == 1:
            if wy > bottom_y:
                bottom_y = wy
                bottom_frame = f
            # 开始回升
            if wy < bottom_y - 0.10:
                state = 2

        elif state == 2:
            # 重新回到高位
            if wy <= top_y + 0.08:
                end_frame = f
                amplitude = bottom_y - top_y
                duration = (end_frame - start_frame) / fps if fps else 0.0
                drive_time = (bottom_frame - start_frame) / fps
                recovery_time = (end_frame - bottom_frame) / fps

                # 规则核查：冲程幅度不能过于短浅
                is_valid = amplitude >= cfg.SKIERG_MIN_AMPLITUDE
                reason = ViolationType.NONE if is_valid else "警示: 冲程拉桨幅度不足"

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
                        "amplitude": amplitude,
                        "drive_time": drive_time,
                        "recovery_time": recovery_time,
                        "ratio": recovery_time / max(0.01, drive_time),
                        "min_hip_angle": hip_angles[bottom_frame] if bottom_frame < len(hip_angles) else 120.0,
                    }
                )
                reps.append(rep)
                rep_num += 1
                top_y = wy
                state = 0

    analysis.reps = reps
    analysis.finalize()
    if reps:
        analysis.avg_cadence = (len(reps) / (analysis.duration / 60.0)) if analysis.duration > 0 else 0.0
        avg_amp = np.mean([r.metrics.get("amplitude", 0) for r in reps])
        analysis.extra_data["avg_amplitude"] = f"{avg_amp:.2f} 画幅高度"
    return analysis
