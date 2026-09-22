"""Station 5: Rowing (划船机 1000m) 动作动力学与四阶段发力时序分析.

检测与分析指标：
1. 划船机四阶段精确划分：抓水 (Catch) -> 驱动 (Drive) -> 出水 (Finish) -> 恢复 (Recovery)。
2. 划船桨频 (SPM - Strokes Per Minute)。
3. 出水后仰角 (Layback Angle，理想约 10°-15°)，防止过度仰卧或含胸驼背。
4. 发力冲程时序与回桨节奏对称性。
"""

from __future__ import annotations

import numpy as np
from ..rules import RepResult, StationAnalysis, ViolationType
from ..skeleton import angle_between_points, extract_person_joints, inclination_angle


def analyze_rowing(frames: list[dict], fps: float, n_frames: int, *,
                   cfg) -> StationAnalysis:
    """对视频序列进行划船机冲程时序划分与运动学动力学分析."""
    analysis = StationAnalysis(
        station_id="5_rowing",
        station_name="Station 5: 1000m Rowing (划船机)",
        fps=fps,
        total_frames=n_frames,
        duration=n_frames / fps if fps else 0.0,
        primary_metric_name="划船冲程数",
        primary_metric_unit="桨",
    )

    if not frames:
        return analysis

    knee_angles = []
    wrist_x_series = []
    torso_layback_series = []

    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            knee_angles.append(160.0)
            wrist_x_series.append(0.5)
            torso_layback_series.append(10.0)
            continue

        joints = extract_person_joints(persons[0])

        # 膝角
        has_l = "left_hip" in joints and "left_knee" in joints and "left_ankle" in joints
        has_r = "right_hip" in joints and "right_knee" in joints and "right_ankle" in joints
        lang = angle_between_points(joints["left_hip"], joints["left_knee"], joints["left_ankle"]) if has_l else 160.0
        rang = angle_between_points(joints["right_hip"], joints["right_knee"], joints["right_ankle"]) if has_r else 160.0
        knee_angles.append((lang + rang) / 2.0)

        # 手腕水平位置 (前后拉动)
        wx = (joints.get("left_wrist", (0.5, 0))[0] + joints.get("right_wrist", (0.5, 0))[0]) / 2.0
        wrist_x_series.append(wx)

        # 躯干后仰角 (垂直线夹角)
        sh = joints.get("left_shoulder", joints.get("right_shoulder", (0.5, 0.3)))
        hp = joints.get("left_hip", joints.get("right_hip", (0.5, 0.6)))
        layback = inclination_angle(hp, sh)
        torso_layback_series.append(layback)

    knee_angles = np.array(knee_angles)
    wrist_x = np.array(wrist_x_series)
    laybacks = np.array(torso_layback_series)

    # 状态机：Catch (最大屈膝, 膝角小) -> Drive (伸腿伸髋拉臂) -> Finish (完全伸展, 出水后仰) -> Recovery (收桨)
    state = 0
    start_frame = 0
    finish_frame = 0
    reps = []
    rep_num = 1

    for f in range(len(knee_angles)):
        kang = knee_angles[f]

        if state == 0:  # 抓水位 (Catch, 屈膝)
            if kang < 80.0:
                state = 1
                start_frame = f

        elif state == 1:  # 驱动阶段 (Drive, 蹬腿后拉)
            if kang > 150.0:  # 腿已伸展接近出水位
                state = 2
                finish_frame = f

        elif state == 2:  # 回桨阶段 (Recovery)
            if kang < 85.0:  # 再次回到抓水准备位
                end_frame = f
                duration = (end_frame - start_frame) / fps if fps else 0.0
                drive_time = (finish_frame - start_frame) / fps
                recovery_time = (end_frame - finish_frame) / fps

                layback_at_finish = laybacks[finish_frame] if finish_frame < len(laybacks) else 12.0
                # 规则核查：后仰角过大或未充分后仰
                layback_ok = 5.0 <= layback_at_finish <= 25.0
                reason = ViolationType.NONE if layback_ok else "提示: 出水后仰角度偏离建议区间 (10°-15°)"

                rep = RepResult(
                    number=rep_num,
                    is_valid=True,
                    reason=reason,
                    start_time=start_frame / fps,
                    peak_time=finish_frame / fps,
                    end_time=end_frame / fps,
                    duration=duration,
                    start_frame=start_frame,
                    peak_frame=finish_frame,
                    end_frame=end_frame,
                    metrics={
                        "drive_time": drive_time,
                        "recovery_time": recovery_time,
                        "ratio": recovery_time / max(0.01, drive_time),
                        "layback_deg": layback_at_finish,
                    }
                )
                reps.append(rep)
                rep_num += 1
                state = 0

    analysis.reps = reps
    analysis.finalize()
    if reps:
        analysis.avg_cadence = (len(reps) / (analysis.duration / 60.0)) if analysis.duration > 0 else 0.0
        avg_lay = np.mean([r.metrics.get("layback_deg", 12.0) for r in reps])
        analysis.extra_data["avg_layback"] = f"{avg_lay:.1f}°"
    return analysis
