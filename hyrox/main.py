"""Hyrox 8 项专项运动与跑步区间视觉裁判与动作分析主入口.

使用方式：
    # 运行指定站点的动作分析与裁判检测 (以墙球为例)
    python main.py --station 8_wall_balls

    # 运行真实视频
    python main.py --station 8_wall_balls --video /path/to/wall_balls.mp4

    # 运行全部 8 个站点与跑步区间的全套自动化测试套件
    python main.py --test-all
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src.mock import generate_mock_poses
from src.render import render_hyrox_video
from src.report import generate_hyrox_reports
from src.rules import StationAnalysis
from src.stations import (
    analyze_burpees,
    analyze_farmers_carry,
    analyze_hyrox_run,
    analyze_lunges,
    analyze_rowing,
    analyze_skierg,
    analyze_sled,
    analyze_wall_balls,
)

console = Console()


def rule_step(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def get_station_analyzer(station_id: str):
    """根据站点 ID 获取对应专项动力学分析器."""
    analyzers = {
        "1_skierg": analyze_skierg,
        "2_sled_push": lambda f, fps, n, cfg: analyze_sled(f, fps, n, is_push=True, cfg=cfg),
        "3_sled_pull": lambda f, fps, n, cfg: analyze_sled(f, fps, n, is_push=False, cfg=cfg),
        "4_burpees": analyze_burpees,
        "5_rowing": analyze_rowing,
        "6_farmers_carry": analyze_farmers_carry,
        "7_lunges": analyze_lunges,
        "8_wall_balls": analyze_wall_balls,
        "hyrox_run": analyze_hyrox_run,
    }
    return analyzers.get(station_id, analyze_wall_balls)


def synthesize_mock_data(station_id: str, n_frames: int, fps: float):
    """回退到生物力学运动学模拟器以生成姿态时序帧."""
    return generate_mock_poses(station_id, n_frames=n_frames, fps=fps)


def extract_vlmrun_video_poses(
    video_path: Path,
    station_id: str,
    fps: float,
    n_frames: int,
    *,
    cfg,
) -> list[dict]:
    """通过 VLMRun 云端网关调用 ViTPose 提取真实视频人体骨骼姿态."""
    import base64
    import hashlib
    import os

    # 1. 确保项目与缓存路径就绪
    for d in (cfg.PROJECT_DIR, cfg.DATA_DIR, cfg.INPUT_DIR, cfg.CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 2. 视频特征哈希与缓存检索
    cache_meta = {
        "video": video_path.name,
        "size": video_path.stat().st_size,
        "model": cfg.MODEL,
        "inference_height": cfg.INFERENCE_HEIGHT,
        "video_fps": cfg.VIDEO_FPS,
        "precision": cfg.PRECISION,
        "convert_crf": cfg.CONVERT_CRF,
        "output_crf": cfg.OUTPUT_CRF,
    }
    digest = hashlib.sha256(json.dumps(cache_meta, sort_keys=True).encode()).hexdigest()[:12]
    cache_file = cfg.CACHE_DIR / f"poses_{digest}.json"

    if cfg.REUSE_POSES and not cfg.FORCE_RECONVERT and cache_file.exists():
        console.print(f"  [green]命中 VLMRun 姿态缓存:[/] {cache_file.name}")
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 3. 检查 API Key 凭据
    api_key = os.environ.get("VLMRUN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if cfg.MOCK_FALLBACK:
            console.print("  [yellow]未检测到 VLMRUN_API_KEY，根据 MOCK_FALLBACK 配置自动回退到动力学对照基准...[/]")
            return synthesize_mock_data(station_id, n_frames=n_frames, fps=fps)
        raise RuntimeError("Missing VLMRUN_API_KEY for VLMRun inference.")

    # 4. 执行云端 VLMRun / OpenAI API 请求
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=cfg.GATEWAY_BASE_URL,
            api_key=api_key,
            timeout=cfg.REQUEST_TIMEOUT,
        )

        video_bytes = video_path.read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode("ascii")

        extra_body = {
            "method": "pose",
            "video_fps": fps if cfg.EVERY_FRAME else cfg.VIDEO_FPS,
            "precision": cfg.PRECISION,
            "height": cfg.INFERENCE_HEIGHT,
        }

        console.print(f"  正在请求 VLMRun 云端推理网关 ({cfg.MODEL})...")
        response = client.chat.completions.create(
            model=cfg.MODEL,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                }],
            }],
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        payload = json.loads(response.choices[0].message.content)
        content = payload.get("content", {})
        frames = content.get("frames", [])

        if not frames and cfg.MOCK_FALLBACK:
            console.print("  [yellow]VLMRun 返回为空，启用对照数据...[/]")
            return synthesize_mock_data(station_id, n_frames=n_frames, fps=fps)

        if cfg.REUSE_POSES and frames:
            cache_file.write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")

        return frames
    except Exception as e:
        console.print(f"  [yellow]VLMRun 网关连接异常 ({e})[/]")
        if cfg.MOCK_FALLBACK:
            console.print("  [cyan]根据 MOCK_FALLBACK 策略自动切换至动力学基准流...[/]")
            return synthesize_mock_data(station_id, n_frames=n_frames, fps=fps)
        raise


def run_station(station_id: str, video_file: Path | None = None,
                use_mock: bool = True) -> StationAnalysis:
    """运行单项站点的完整分析、视频渲染与报表输出流程."""
    t0 = time.perf_counter()
    station_name = cfg.STATIONS.get(station_id, station_id)
    console.print()
    console.rule(f"[bold green]HYROX 专项视觉分析: {station_name}[/]", align="center")

    # 1. 检查输入源与姿态数据获取方式
    rule_step(1, "输入源与姿态流检测")
    has_video = video_file and video_file.is_file()
    if has_video:
        console.print(f"  输入视频: [green]{video_file}[/]")
    else:
        console.print(f"  [yellow]未提供实体视频文件[/]，自动启用 [bold cyan]Hyrox 生物力学合成姿态模拟器[/]")

    # 2. 获取姿态时序帧
    rule_step(2, "人体关键点骨骼时序提取")
    fps = 30.0
    n_frames = 210

    if has_video:
        import cv2
        cap = cv2.VideoCapture(str(video_file))
        if cap.isOpened():
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps and detected_fps > 0:
                fps = round(detected_fps, 2)
            detected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if detected_frames and detected_frames > 0:
                n_frames = min(detected_frames, 900)  # 上限保护 30 秒
            cap.release()
            console.print(f"  视频解析成功: [bold]{n_frames}[/] 帧 (@{fps} fps)")

    if has_video and not use_mock:
        console.print(f"  启动 VLMRun 云端姿态推理 (OpenAI client.chat.completions / 模型: [bold]{cfg.MODEL}[/])...")
        frames = extract_vlmrun_video_poses(
            video_file, station_id, fps=fps, n_frames=n_frames, cfg=cfg
        )
    else:
        console.print(f"  已生成 [bold]{n_frames}[/] 帧连续生物力学动力学轨迹帧 (@{fps} fps)")
        frames = synthesize_mock_data(station_id, n_frames=n_frames, fps=fps)

    # 3. 运行 Hyrox 专项动作识别与裁判规则核查
    rule_step(3, "Hyrox 竞赛裁判与合规规则检测")
    analyzer = get_station_analyzer(station_id)
    analysis = analyzer(frames, fps, len(frames), cfg=cfg)

    # 终端输出判罚汇总表格
    table = Table(box=None, pad_edge=False)
    table.add_column("动作序号", justify="center", style="dim")
    table.add_column("判罚结果", justify="left")
    table.add_column("动作耗时", justify="right")
    table.add_column("起始时刻", justify="right")
    table.add_column("峰值/触底", justify="right")
    table.add_column("关键动力学度量", justify="left")

    for r in analysis.reps:
        status_str = "[bold green]✓ 有效 (Valid)[/]" if r.is_valid \
            else f"[bold red]✗ 违规 ({r.reason})[/]"
        metric_str = " · ".join(
            f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in r.metrics.items()
        )
        table.add_row(
            str(r.number),
            status_str,
            f"{r.duration:.2f}s",
            f"{r.start_time:.2f}s",
            f"{r.peak_time:.2f}s",
            metric_str,
        )

    title_summary = (
        f"[bold]累计动作: {len(analysis.reps)} 次[/] | "
        f"[bold green]有效: {analysis.valid_reps_count}[/] | "
        f"[bold red]违规 (No-Rep): {analysis.no_reps_count}[/] | "
        f"通过率: [bold cyan]{analysis.validity_rate:.1f}%[/]"
    )
    console.print(Panel(table, title=title_summary, title_align="left", expand=False))

    # 4. 视频合成与侧边数据看板 (Side Panel) 渲染
    rule_step(4, "合成视频与侧边数据看板 (Side Panel) 编码")
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    out_dir = cfg.OUTPUT_DIR / f"{station_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = out_dir / f"{station_id}_annotated.mp4"
    render_stats = render_hyrox_video(
        video_file if has_video else None,
        frames,
        analysis,
        out_video_path,
        cfg=cfg,
        console=console
    )
    console.print(f"  生成合成视频 -> [dim]{out_video_path}[/] "
                  f"({render_stats['width']}x{render_stats['height']}, "
                  f"{render_stats['frames_written']} 帧)")

    # 5. 生成专业评估报表 (JSON, TXT, PNG)
    rule_step(5, "生成赛事评估与生物力学分析报告")
    reports = generate_hyrox_reports(
        analysis,
        out_dir,
        source_name=video_file.name if has_video else "运动学模拟输入"
    )
    console.print(f"  JSON 详细数据 -> [dim]{reports['json']}[/]")
    console.print(f"  裁判文本报告   -> [dim]{reports['txt']}[/]")
    console.print(f"  时序趋势图表   -> [dim]{reports['png']}[/]")

    elapsed = time.perf_counter() - t0
    console.print(f"\n[bold green]✓ {station_name} 分析渲染完成，总耗时 {elapsed:.2f} 秒！[/]\n")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Hyrox 8 项专项运动与跑步视觉裁判分析系统")
    parser.add_argument("--station", type=str, default="8_wall_balls",
                        choices=list(cfg.STATIONS.keys()) + ["all"],
                        help="选择分析的 Hyrox 运动站点 (默认: 8_wall_balls)")
    parser.add_argument("--video", type=str, default=None,
                        help="输入的源视频路径 (.mp4, .mov)")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="强制启用离线模拟测试模式")
    parser.add_argument("--test-all", action="store_true", default=False,
                        help="一键运行全部 8 个站点与跑步区间的自动化全套测试")

    args = parser.parse_args()

    video_path = Path(args.video).expanduser() if args.video else None

    if args.test_all or args.station == "all":
        console.rule("[bold cyan]开始执行 HYROX 全套 8 项运动 + 跑步区间自动化评测[/]", align="center")
        results = {}
        for st_id in cfg.STATIONS.keys():
            res = run_station(st_id, video_file=None, use_mock=True)
            results[st_id] = res
        console.print(f"[bold green]✔ 全部 {len(results)} 个 Hyrox 运动站点测试完毕！[/]")
    else:
        run_station(args.station, video_file=video_path, use_mock=args.mock or not video_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
