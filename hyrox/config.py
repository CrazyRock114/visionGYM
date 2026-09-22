"""Hyrox 8 项专项运动与跑步区间配置参数与裁判规则标准.

包含 Hyrox 官方竞赛裁判标准（深蹲深度、后膝触地、双脚起跳、锁髋锁膝）、
关键点动力学分析阈值、中文界面与侧边数据面板（Side Panel）视觉规范。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"

# ── 运行与模型配置 ─────────────────────────────────────────────────────────────
MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0
EVERY_FRAME = True
VIDEO_FPS = 2.0
PRECISION = 4

# 离线演示 / 模拟测试开关：若无 API Key，可自动使用运动学模拟生成姿态测试动作
MOCK_FALLBACK = True

# 视频转码与导出配置
INFERENCE_HEIGHT = 1080
EXPORT_HEIGHT = 1080
CONVERT_CRF = 23
OUTPUT_CRF = 20
FORCE_RECONVERT = False
REUSE_POSES = True
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

# ── 视觉与渲染配置 ─────────────────────────────────────────────────────────────
DRAW_FACE = False
DRAW_BBOX = True
DRAW_SKELETON = True
DRAW_TRAIL = True
TRAIL_LENGTH = 15
LINE_THICKNESS = 3
POINT_RADIUS = 5
BBOX_COLOR = (255, 160, 40)       # BGR: 活力橙蓝
VALID_REP_COLOR = (80, 220, 80)    # BGR: 合规绿色
NO_REP_COLOR = (40, 40, 240)       # BGR: 违规红色
WARN_COLOR = (40, 200, 255)        # BGR: 警示黄色

# ── 中文字体配置 ───────────────────────────────────────────────────────────────
# 优先匹配 macOS、Linux 与 Windows 系统中文字体
PANEL_FONT = "auto"
PANEL_FONT_INDEX = None
CHINESE_FONT_CANDIDATES = (
    ("/System/Library/Fonts/PingFang.ttc", 0, "PingFang SC"),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0, "STHeiti Medium"),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0, "STHeiti Light"),
    ("/Library/Fonts/Arial Unicode.ttf", 0, "Arial Unicode"),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 0, "Songti SC"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0, "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0, "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0, "WenQuanYi Micro Hei"),
    ("C:/Windows/Fonts/msyh.ttc", 0, "Microsoft YaHei"),
    ("C:/Windows/Fonts/simhei.ttf", 0, "SimHei"),
)

# 侧边面板基础尺寸
SIDE_PANEL_WIDTH = 720
SIDE_PANEL_SCALE = 0.50            # 画幅右侧占总宽度的比例
PANEL_BG = "#101318"
PANEL_FG = "#f3f5f8"
PANEL_MUTED = "#88909b"
PANEL_ACCENT = "#2f80ed"
PANEL_SUCCESS = "#27ae60"
PANEL_DANGER = "#eb5757"
PANEL_WARNING = "#f2994a"

# ── Hyrox 8 项站点与比赛规则参数 ───────────────────────────────────────────────

STATIONS = {
    "1_skierg": "Station 1: 1000m SkiErg 滑雪机",
    "2_sled_push": "Station 2: 50m Sled Push 雪橇推",
    "3_sled_pull": "Station 3: 50m Sled Pull 雪橇拉",
    "4_burpees": "Station 4: 80m Burpee Broad Jumps 波比跳远",
    "5_rowing": "Station 5: 1000m Rowing 划船机",
    "6_farmers_carry": "Station 6: 200m Farmers Carry 农夫行走",
    "7_lunges": "Station 7: 100m Sandbag Lunges 沙袋箭步蹲",
    "8_wall_balls": "Station 8: 75/100 Wall Balls 药球掷高",
    "hyrox_run": "Hyrox Running 8x1km 跑步区间分析",
}

# 1. Wall Balls (墙球) 规则与阈值
WALL_BALL_SQUAT_DEPTH_RATIO = 1.0     # 髋关节折痕低于膝关节上缘 (y_hip >= y_knee)
WALL_BALL_MIN_KNEE_ANGLE = 95.0       # 有效深蹲膝关节最大允许角 (度，需小于此值)
WALL_BALL_LOCKOUT_KNEE_ANGLE = 160.0  # 站立完全锁膝角度 (度)
WALL_BALL_LOCKOUT_HIP_ANGLE = 160.0   # 站立完全锁髋角度 (度)
WALL_BALL_BALL_THROW_GATE = True      # 检查药球出手与抛升

# 2. Burpee Broad Jumps (波比跳远) 规则与阈值
BURPEE_CHEST_FLOOR_TOLERANCE = 0.08   # 胸腔至支撑地面的归一化垂直高度差
BURPEE_TAKEOFF_SYNC_TOLERANCE_S = 0.08# 双脚同步起跳时间差容差 (秒，超额判单脚违规)
BURPEE_MIN_JUMP_DIST_RATIO = 0.35     # 每次跳远最小有效位移比例 (相对于身长)

# 3. Sandbag Lunges (沙袋箭步蹲) 规则与阈值
LUNGE_REAR_KNEE_FLOOR_TOL = 0.06      # 后膝距地面触地判定容差
LUNGE_LOCKOUT_ANGLE = 162.0           # 站立起身完全锁紧角度
LUNGE_FRONT_KNEE_FLEXION = 100.0      # 前腿有效弓步膝关节角度

# 4. SkiErg (滑雪机) 规则与阈值
SKIERG_MIN_AMPLITUDE = 0.25           # 冲程最小垂直幅度 (相对于身高)
SKIERG_POWER_HINGE_ANGLE = 115.0      # 底部屈髋发力角参考

# 5. Rowing (划船机) 规则与阈值
ROWING_LAYBACK_TARGET_DEG = 12.0      # 出水后仰理想角度 (10°-15°)
ROWING_CATCH_KNEE_ANGLE = 65.0        # 抓水最大屈膝角度

# 6. Farmers Carry (农夫行走) 规则与阈值
FARMERS_SWAY_MAX_DEG = 8.0            # 躯干侧倾警示角度
FARMERS_POSTURE_TOLERANCE = 0.05      # 左右肩下垂不对称度上限

# 7. Sled Push & Pull (雪橇推拉) 规则与阈值
SLED_PUSH_OPTIMAL_ANGLE = (38.0, 52.0)# 理想推进力线倾角区间 (度)
SLED_CADENCE_STALL_S = 1.5            # 停顿失速判定期 (秒)
