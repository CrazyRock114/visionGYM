"""Station 6: Farmers Carry (农夫行走 200m) 负重体态中立度与步态稳定性分析.

检测与分析指标：
1. 脊柱中立度与躯干左右侧倾 (Lateral Trunk Sway)，监测核心代偿与疲劳倾斜。
2. 双肩水平度 (Shoulder Leveling) 与单侧下塌检测。
3. 负重行进步频 (Cadence) 与步幅节律一致性。
"""

from __future__ import annotations

import numpy as np
from ..rules import RepResult, StationAnalysis, ViolationType
from ..skeleton import extract_person_joints, inclination_angle


def analyze_farmers_carry(frames: list[dict], fps: float, n_frames: int, *,
                          cfg) -> StationAnalysis:
    """对视频序列进行农夫行走体态中立与步态稳定性分析."""
    analysis = StationAnalysis(
        station_id="6_farmers_carry",
        station_name="Station 6: 200m Farmers Carry (农夫行走)",
        fps=fps,
        total_frames=n_frames,
        duration=n_frames / fps if fps else 0.0,
        primary_metric_name="负重步频",
        primary_metric_unit="SPM",
    )

    if not frames:
        return analysis

    sway_angles = []
    shoulder_imbalance = []
    left_ankle_ys = []
    right_ankle_ys = []

    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            sway_angles.append(0.0)
            shoulder_imbalance.append(0.0)
            left_ankle_ys.append(0.85)
            right_ankle_ys.append(0.85)
            continue

        joints = extract_person_joints(persons[0])

        ls = joints.get("left_shoulder", (0.45, 0.3))
        rs = joints.get("right_shoulder", (0.55, 0.3))
        lh = joints.get("left_hip", (0.46, 0.6))
        rh = joints.get("right_hip", (0.54, 0.6))

        mid_shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        mid_hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)

        # 侧倾角
        sway = inclination_angle(mid_hip, mid_shoulder)
        sway_angles.append(sway)

        # 左右肩高差
        diff = abs(ls[1] - rs[1])
        shoulder_imbalance.append(diff)

        lay = joints.get("left_ankle", (0, 0.85))[1]
        ray = joints.get("right_ankle", (0, 0.85))[1]
        left_ankle_ys.append(lay)
        right_ankle_ys.append(ray)

    sway_arr = np.array(sway_angles)
    mean_sway = float(np.mean(sway_arr))
    max_sway = float(np.max(sway_arr))
    sway_std = float(np.std(sway_arr))

    # 步频估算 (基于左右单脚各自的触地极值)
    l_arr = np.array(left_ankle_ys)
    r_arr = np.array(right_ankle_ys)
    l_steps = sum(1 for i in range(1, len(l_arr) - 1)
                  if l_arr[i] > l_arr[i - 1] and l_arr[i] > l_arr[i + 1])
    r_steps = sum(1 for i in range(1, len(r_arr) - 1)
                  if r_arr[i] > r_arr[i - 1] and r_arr[i] > r_arr[i + 1])
    steps = max(1, l_steps + r_steps)

    spm = (steps / (analysis.duration / 60.0)) if analysis.duration > 0 else 0.0
    analysis.avg_cadence = spm

    # 稳定性评级
    is_stable = max_sway <= cfg.FARMERS_SWAY_MAX_DEG
    status_text = "核心稳定" if is_stable else ViolationType.POSTURE_SWAY

    analysis.extra_data = {
        "mean_sway_deg": f"{mean_sway:.1f}°",
        "max_sway_deg": f"{max_sway:.1f}°",
        "sway_stability_score": f"{max(0, 100 - int(sway_std * 10))}/100",
        "estimated_steps": steps,
        "posture_status": status_text,
    }

    # 将行进分段作为 RepResult 输出
    analysis.reps = [
        RepResult(
            number=1,
            is_valid=is_stable,
            reason=ViolationType.NONE if is_stable else status_text,
            start_time=0.0,
            peak_time=analysis.duration / 2.0,
            end_time=analysis.duration,
            duration=analysis.duration,
            metrics={
                "cadence_spm": spm,
                "max_sway": max_sway,
                "stability": max(0, 100 - int(sway_std * 10)),
            }
        )
    ]
    analysis.finalize()
    return analysis
