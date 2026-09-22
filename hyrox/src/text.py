"""中文字体解析与无损抗锯齿绘制引擎.

原生支持 macOS (苹方/华文黑体)、Linux (思源黑体/文泉驿) 和 Windows (微软雅黑/黑体) 的
CJK 字体自适应探测与渲染，确保中文文本在控制台、视频图层及侧边看板上排版精准美观。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np

# 跨平台中文字体候选优先级列表 (路径, TTC字体索引, 名称)
CHINESE_FONT_CANDIDATES: tuple[tuple[str, int, str], ...] = (
    # macOS
    ("/System/Library/Fonts/PingFang.ttc", 0, "PingFang SC"),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0, "STHeiti Medium"),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0, "STHeiti Light"),
    ("/Library/Fonts/Arial Unicode.ttf", 0, "Arial Unicode"),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 0, "Songti SC"),
    # Linux
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0, "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0, "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0, "WenQuanYi Micro Hei"),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0, "WenQuanYi Zen Hei"),
    # Windows
    ("C:/Windows/Fonts/msyh.ttc", 0, "Microsoft YaHei"),
    ("C:/Windows/Fonts/simhei.ttf", 0, "SimHei"),
    # 英文备用
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1, "Helvetica Neue Bold"),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, "Arial Bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, "DejaVu Sans Bold"),
)


def resolve_font(spec: str = "auto", index: int | None = None) -> tuple[str, int, str] | None:
    """自动匹配系统可用中文字体."""
    try:
        from PIL import ImageFont  # noqa: F401
    except ImportError:
        return None

    if spec == "opencv":
        return None

    if spec != "auto":
        path = Path(spec).expanduser()
        if path.is_file():
            return (str(path), index or 0, path.stem)

    for path, idx, name in CHINESE_FONT_CANDIDATES:
        if Path(path).is_file():
            return (path, index if index is not None else idx, name)

    # 尝试由 matplotlib 内置字体兜底
    try:
        import matplotlib
        p = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if p.is_file():
            return (str(p), 0, "DejaVu Sans Bold")
    except Exception:
        pass

    return None


@lru_cache(maxsize=4096)
def _patch(text: str, font_path: str, font_index: int, size: int,
           color: tuple[int, int, int]) -> np.ndarray:
    """利用 Pillow 渲染单行文本为紧凑 RGBA 像素块并缓存."""
    from PIL import Image, ImageDraw, ImageFont

    face = ImageFont.truetype(font_path, size, index=font_index)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=face)
    w, h = max(1, right - left), max(1, bottom - top)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # color 为 (R, G, B)
    ImageDraw.Draw(img).text((-left, -top), text, font=face, fill=(*color, 255))
    return np.array(img)


def measure(text: str, font, size: int) -> tuple[int, int]:
    """计算文本像素宽高 (width, height)."""
    if font is None:
        import cv2
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
        return w, h
    patch = _patch(text, font[0], font[1], size, (255, 255, 255))
    return patch.shape[1], patch.shape[0]


def blend(img: np.ndarray, patch: np.ndarray, x: int, y: int, *,
          opacity: float = 1.0) -> np.ndarray:
    """将 RGBA 文本像素块以指定透明度混合到 BGR 图像底图上."""
    ph, pw = patch.shape[:2]
    fh, fw = img.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + pw), min(fh, y + ph)
    if x0 >= x1 or y0 >= y1:
        return img

    sub = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    alpha = (sub[..., 3:4].astype(np.float32) / 255.0) * opacity
    bgr = sub[..., 2::-1].astype(np.float32)   # RGBA -> BGR
    img[y0:y1, x0:x1] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return img


def draw(img: np.ndarray, text: str, font, *, size: int, xy: tuple[int, int],
         color: tuple[int, int, int] = (255, 255, 255), anchor: str = "lt",
         opacity: float = 1.0, rotate: int = 0) -> np.ndarray:
    """在图像指定锚点位置绘制抗锯齿中文文本."""
    if not text:
        return img

    if font is None:
        import cv2
        scale = max(0.4, size / 30.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = xy
        x -= {"l": 0, "c": tw // 2, "r": tw}[anchor[0]]
        y += {"t": th, "m": th // 2, "b": 0}[anchor[1]]
        # color BGR
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color[::-1], 2, cv2.LINE_AA)
        return img

    patch = _patch(text, font[0], font[1], size, color)
    if rotate:
        patch = np.rot90(patch, k=(rotate // 90) % 4)

    ph, pw = patch.shape[:2]
    x, y = xy
    x -= {"l": 0, "c": pw // 2, "r": pw}[anchor[0]]
    y -= {"t": 0, "m": ph // 2, "b": ph}[anchor[1]]
    return blend(img, patch, x, y, opacity=opacity)
