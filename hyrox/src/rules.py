"""Hyrox 官方竞赛裁判与动作合规规则引擎.

严格遵循 Hyrox 官方竞赛规则手册（Hyrox Rulebook），实时判定：
- 有效动作 (Valid Rep)
- 无效动作与违规判定 (No-Rep)
- 违规原因分类记录（如：深蹲深度不足、起跳未双脚同步、后膝未触地、未完全站立锁髋）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


class ViolationType:
    """常见 Hyrox 竞赛违规类型."""
    NONE = "合规有效"
    # Wall Balls
    SQUAT_DEPTH = "No Rep: 深蹲深度不足 (屈髋未低于膝关节)"
    NO_LOCKOUT = "No Rep: 起身未完全锁髋/锁膝"
    BALL_LOW = "No Rep: 药球投掷未达标靶高度"
    # Burpee Broad Jumps
    CHEST_NOT_TOUCHED = "No Rep: 胸部未完全贴地"
    STEP_TAKEOFF = "No Rep: 非双脚同时起跳 (严禁单脚跨步)"
    STEP_LANDING = "No Rep: 非双脚同时落地"
    JUMP_TOO_SHORT = "No Rep: 跳跃距离不足"
    # Sandbag Lunges
    REAR_KNEE_NOT_TOUCHED = "No Rep: 后膝未触碰地面"
    INCOMPLETE_EXTENSION = "No Rep: 出步前双腿未完全伸展直立"
    WRONG_LEG_SEQUENCE = "No Rep: 未交替腿部出步"
    # Sled / General
    IRREGULAR_PACE = "警示: 推进节奏失速停顿"
    POSTURE_SWAY = "警示: 躯干核心严重侧倾代偿"


@dataclass
class RepResult:
    """单次动作详情与合规判罚记录."""
    number: int
    is_valid: bool
    reason: str = ViolationType.NONE
    start_time: float = 0.0
    peak_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    start_frame: int = 0
    peak_frame: int = 0
    end_frame: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        clean_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, (np.floating, float)):
                clean_metrics[k] = round(float(v), 2)
            elif isinstance(v, (np.integer, int)):
                clean_metrics[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                clean_metrics[k] = bool(v)
            else:
                clean_metrics[k] = str(v)

        return {
            "rep": int(self.number),
            "valid": bool(self.is_valid),
            "status": "有效 (Valid)" if self.is_valid else f"无效 ({self.reason})",
            "reason": str(self.reason),
            "duration": round(float(self.duration), 3),
            "start_time": round(float(self.start_time), 3),
            "peak_time": round(float(self.peak_time), 3),
            "end_time": round(float(self.end_time), 3),
            "metrics": clean_metrics,
        }


@dataclass
class StationAnalysis:
    """单项 Hyrox 站点的全流程分析报告."""
    station_id: str
    station_name: str
    fps: float
    total_frames: int
    duration: float
    reps: list[RepResult] = field(default_factory=list)
    cadence_series: list[float] = field(default_factory=list)
    primary_metric_name: str = "动作次数"
    primary_metric_unit: str = "次"
    avg_cadence: float = 0.0
    valid_reps_count: int = 0
    no_reps_count: int = 0
    validity_rate: float = 0.0
    warnings: list[str] = field(default_factory=list)
    extra_data: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """汇总统计数据."""
        self.valid_reps_count = sum(1 for r in self.reps if r.is_valid)
        self.no_reps_count = sum(1 for r in self.reps if not r.is_valid)
        total = len(self.reps)
        self.validity_rate = (self.valid_reps_count / total * 100.0) if total > 0 else 0.0

    def summary(self) -> dict:
        self.finalize()
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "duration_seconds": round(self.duration, 2),
            "total_reps": len(self.reps),
            "valid_reps": self.valid_reps_count,
            "no_reps": self.no_reps_count,
            "validity_rate_pct": round(self.validity_rate, 1),
            "avg_cadence": round(self.avg_cadence, 1),
            "reps_detail": [r.as_dict() for r in self.reps],
            "extra": self.extra_data,
        }
