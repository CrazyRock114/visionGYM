# Vision-Demos 中文汉化与 HYROX 体能赛事优化套件

本项目基于 [jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos) 进行全面中文汉化、跨平台中文字体自适应渲染支持，并全新扩展构建了针对 **HYROX 全球体能耐力锦标赛全套 8 项功能性专项运动 + 跑步区间** 的专业级计算机视觉裁判与动作动力学分析套件。

> 🌐 **在线交互控制台 (Live Vercel Demo)**: [https://vision-gym-seven.vercel.app](https://vision-gym-seven.vercel.app)  
> 📦 **GitHub 官方开源仓库**: [https://github.com/CrazyRock114/visionGYM](https://github.com/CrazyRock114/visionGYM)

---

## 目录

- [核心功能矩阵](#核心功能矩阵)
- [HYROX 8 项体能专项裁判与分析系统](#hyrox-8-项体能专项裁判与分析系统)
- [原版 4 大视觉演示汉化与升级](#原版-4-大视觉演示汉化与升级)
- [中文字体与多平台渲染优化](#中文字体与多平台渲染优化)
- [环境安装与快速启动](#环境安装与快速启动)
- [命令行运行指南](#命令行运行指南)
- [开源协议](#开源协议)

---

## 核心功能矩阵

| 项目分类 | 模块目录 | 核心功能 | 核心算法与视觉模型 |
|---|---|---|---|
| **HYROX 专项 (全新)** | **[`hyrox/`](hyrox/)** | **针对 Hyrox 8 站功能性动作 + 跑步区间的全自动规则执裁 (No-Rep)、动作计数、周期分解与体能衰减评估** | COCO-17 骨架动力学模型、官方 Rulebook 裁判状态机、双脚离地时间戳同步、侧倾角检测 |
| **经典体能演示** | **[`chin_ups/`](chin_ups/)** | 智能引体向上次数统计，精准划分拉起与下落耗时，生成完整中文分析面板与图表 | `vitpose-plus-large` 姿态估计、垂直速度极值分段 |
| **跑步生物力学** | **[`running/`](running/)** | 跑步步频 (SPM) 测算、每步触地瞬间 (Foot Strike) 检测、触地瞬间膝关节屈曲形态平均与中文字体看板 | `vitpose-plus-large` 姿态估计、脚踝垂直加速度零交叉 |
| **舞蹈时空同步** | **[`dance_sync/`](dance_sync/)** | 对比多位舞者同一编舞动作的时空一致性，自动计算骨架相似度得分与关键帧对比 | `vitpose-plus-large` 骨骼关键点、动态时间规整 (DTW) |
| **攀岩抱石分析** | **[`rock_climbing/`](rock_climbing/)** | 自动分割攀岩抱石支点，识别攀爬者触碰岩点顺序，多轮尝试路线对比及中文叠加 | `sam3.1` (分割) + `vitpose-plus-large` (四肢追踪) |

---

## HYROX 8 项体能专项裁判与分析系统

在 Hyrox 比赛中，选手需要在 8 个高强度功能性运动站与 8 组 1km 跑步之间连续交替转换。传统人工裁判难以精准肉眼捕捉微小角度缺陷。本套件严格依据 **Hyrox Official Rulebook**，针对每个专项动作研发了专属的运动动力学检测算法：

```
Hyrox 竞赛全程流水线:
[1km Run] ➔ [1. SkiErg 滑雪机] ➔ [1km Run] ➔ [2. Sled Push 雪橇推] ➔ 
[1km Run] ➔ [3. Sled Pull 雪橇拉] ➔ [1km Run] ➔ [4. Burpee Broad Jumps 波比跳远] ➔ 
[1km Run] ➔ [5. Rowing 划船机] ➔ [1km Run] ➔ [6. Farmers Carry 农夫行走] ➔ 
[1km Run] ➔ [7. Sandbag Lunges 沙袋箭步蹲] ➔ [1km Run] ➔ [8. Wall Balls 药球掷高]
```

### 1. 裁判规则与算法实现对照

| 赛事工作站 | 官方规则核心要求 (Hyrox Rules) | 计算机视觉判定算法 (CV Logic) | 违规判定 (No-Rep) 条件 |
| :--- | :--- | :--- | :--- |
| **Station 1: SkiErg<br>(1000m 滑雪机)** | 动作展开充分，双手高位过头引手到底部完整冲程。 | 双腕垂直位移曲线、髋-膝-踝三屈三伸屈伸角时序分析。 | 冲程幅度过浅、未过头预备即下压。 |
| **Station 2: Sled Push<br>(50m 雪橇推)** | 维持推进动力线，持续向前高频蹬踏推进。 | 肩-髋-踝躯干驱动前倾角（基准 40°-50°）、下肢高频蹬踏节奏。 | 躯干过直（角 > 65°）、长时间卡顿失速停滞。 |
| **Station 3: Sled Pull<br>(50m 雪橇拉)** | 低重心抗阻拔河式站姿，向后引拉拽动绳索。 | 身体质心高度追踪、倒步步频与双臂后拉冲程节奏。 | 重心漂移失稳、后拉动作不连贯。 |
| **Station 4: Burpee Broad Jumps<br>(80m 波比跳远)** | **① 胸部完全贴地** (Chest to Deck)；<br>**② 必须双脚同时起跳**，禁止单腿跨步。 | 1. 躯干胸腔与地面最低垂直距离；<br>2. 左右脚起跳离地时间差 ($\Delta t$)；<br>3. 滞空位移估算。 | 1. 胸部未完全触底；<br>2. 双脚离地时差 $\Delta t > 80\text{ms}$（判 **单腿跨步违规**）。 |
| **Station 5: Rowing<br>(1000m 划船机)** | 标准发力顺序：蹬腿->伸髋->拉臂；出水躯干后仰 10°-15°。 | 膝关节屈伸角、双腕水平行程、出水躯干后仰角（Layback Angle）。 | 提早拉臂偷力、出水过度后仰或弓背。 |
| **Station 6: Farmers Carry<br>(200m 农夫行走)** | 双手持重行进，核心收紧，脊柱保持中立直立。 | 左右肩水平夹角、躯干侧倾角（Lateral Sway）标准差。 | 侧倾摇摆幅度过大、单侧溜肩代偿严重。 |
| **Station 7: Sandbag Lunges<br>(100m 沙袋箭步蹲)** | **① 后膝必须触及地面**；<br>**② 迈入下一步前必须双腿完全站立直立锁膝锁髋**。 | 1. 后膝最低点离地垂直距离；<br>2. 站立相双膝屈伸角检测；<br>3. 左右腿交替步态追踪。 | 1. 后膝未触地偷深度；<br>2. 站立相膝关节未伸展直立（仍屈曲）。 |
| **Station 8: Wall Balls<br>(75/100 药球掷高)** | **① 屈髋折痕必须低于膝盖上缘**（深度不足判 No-Rep）；<br>**② 起身完全锁髋上推，击中目标标靶**。 | 1. 膝关节屈曲内角（$\theta < 95^\circ$）与髋膝垂直相对位置（$y_{\text{hip}} \ge y_{\text{knee}}$）；<br>2. 站立直立锁定判定。 | 1. 深蹲幅度不足（大腿未破平行）；<br>2. 站立未伸展完全。 |
| **跑步区间<br>(8 x 1km Running)** | 8 组高强度力量站之间的 1km 衔接跑步。 | 双侧脚踝离地垂直轨迹、触地时间（GCT）、步频衰减率。 | 步频大幅跳水（疲劳衰竭）、左右步态不对称严重。 |

### 2. 竞赛级暗黑数据看板与分析产物

每次运行 Hyrox 分析后，系统会在 `hyrox/data/output/` 自动生成全套数据资产：
- **`*_annotated.mp4`**：包含骨骼热力高亮、实时违规警示横幅（`[警示] NO-REP`）、动作次数计数器、实时角度与右侧专业暗黑风格数据侧面板（Side Panel）的高清合成视频。
- **`analysis.png`**：包含单次动作耗时瀑布图（绿色代表有效合规 Rep，红色代表违规 No-Rep）、动作节奏演变趋势与疲劳拟合回归线。
- **`summary.txt`**：文字版裁判执裁报告，包含达标率、平均周期、违规扣分项详单与改进建议。
- **`analysis.json`**：标准化结构化 JSON 数据，方便接入下游运动员训练系统与赛事大屏。

---

## 原版 4 大视觉演示汉化与升级

1. **`chin_ups/` 引体向上分析**：
   - 终端输出与进度条汉化。
   - 右侧实时数据看板（动作次数、平均拉起耗时、平均下落耗时、节奏比）全面中文化。
   - 最终产出图表（`analysis.png`）的 Matplotlib 坐标轴、图例与统计框支持中文字体渲染。
2. **`running/` 跑步姿态分析**：
   - 跑步步频 (SPM)、每步触地耗时、左右脚对称性文字与图表全汉化。
   - 触地瞬间膝关节平均屈曲形态图（Average Knee Angle at Contact）全面支持中文排版与角度标注。
3. **`dance_sync/` 与 `rock_climbing/`**：
   - 统一升级字体加载引擎，自动适配操作系统中文字体，彻底消除方块字（Tofu glyphs）与乱码。

---

## 中文字体与多平台渲染优化

系统内建了多平台字体自适应探测器（`find_system_chinese_font()`），自动搜索并优先加载高品质中文字体：
- **macOS**: `STHeiti Medium.ttc`（华文黑体）、`PingFang.ttc`（苹方）、`Arial Unicode.ttf`
- **Linux**: `Noto Sans CJK SC`、`WenQuanYi Zen Hei`（文泉驿正黑）、`Source Han Sans CN`
- **Windows**: `msyh.ttc`（微软雅黑）、`simhei.ttf`（黑体）

在 Matplotlib 图像生成模块中，预先注入了 `plt.rcParams['font.sans-serif']` 与 `plt.rcParams['axes.unicode_minus'] = False`，确保正负号与中文汉字同时无损绘制。

---

## 环境安装与快速启动

### 1. 克隆代码与配置 Python 环境

推荐使用 [`uv`](https://github.com/astral-sh/uv) 极速创建虚拟环境（或使用常规 `python3 -m venv`）：

```bash
# 1. 切换到项目根目录
cd /Users/crazyrock/Antigravity/visionGYM

# 2. 创建并激活虚拟环境 (Python 3.10+)
uv venv
source .venv/bin/activate

# 3. 安装依赖项
uv pip install -r chin_ups/requirements.txt
uv pip install -r running/requirements.txt
```

### 2. API Key 说明（可选）

- **完全离线模式 / 模拟合成演示**：**无需任何 API 密钥**！内置高保真人体生物力学与动作违规合成引擎，随时直接启动演示与测试。
- **真实视频分析模式**：如需使用云端高精度姿态模型分析自定义 MP4 视频，请在项目根目录创建 `.env` 文件：
  ```bash
  VLMRUN_API_KEY=your_vlmrun_api_key_here
  ```

---

## 命令行运行指南

### 1. 运行 HYROX 专项分析

#### (1) 运行单项模拟演示（以第 8 站 Wall Balls 墙球为例）
```bash
python hyrox/main.py --station 8_wall_balls
```

#### (2) 分析用户录制的真实比赛视频
```bash
python hyrox/main.py --station 8_wall_balls --video /path/to/your_video.mp4
```

#### (3) 支持的所有运动站代号 (`--station`)
- `1_skierg`: 第 1 站 滑雪机 (1000m SkiErg)
- `2_sled_push`: 第 2 站 雪橇推 (50m Sled Push)
- `3_sled_pull`: 第 3 站 雪橇拉 (50m Sled Pull)
- `4_burpees`: 第 4 站 波比跳远 (80m Burpee Broad Jumps)
- `5_rowing`: 第 5 站 划船机 (1000m Rowing)
- `6_farmers_carry`: 第 6 站 农夫行走 (200m Farmers Carry)
- `7_lunges`: 第 7 站 沙袋箭步蹲 (100m Sandbag Lunges)
- `8_wall_balls`: 第 8 站 药球掷高 (Wall Balls)
- `hyrox_run`: 跑步区间 (8 x 1km Running)

#### (4) 一键全自动综合测试（9 站全量自动化执裁）
```bash
python hyrox/main.py --test-all
```

### 2. 运行原版汉化演示

#### (1) 引体向上汉化版 (`chin_ups`)
```bash
# 模拟模式（无 API Key 即可体验完整渲染效果）
python chin_ups/main.py --mock

# 真实视频模式
python chin_ups/main.py --video chin_ups/data/input/chin_up_01.mp4
```

#### (2) 跑步步频与触地形态汉化版 (`running`)
```bash
# 模拟模式
python running/main.py --mock

# 真实视频模式
python running/main.py --video running/data/input/runner_01.mp4
```

---

## 开源协议

本项目原版核心基于 [Apache-2.0 许可证](LICENSE) 发布。本中文汉化与 HYROX 运动优化套件扩展保留原有许可证声明。
