"""TrueType text rendering for video overlays.

OpenCV's ``putText`` only has the Hershey fonts: single-stroke vector shapes
from the 1960s, with no real weight and no kerning. Pillow can draw any
TrueType/OpenType face onto the same frames, which is the whole difference
between an overlay that looks generated and one that looks designed.

If Pillow is missing, or no usable face is found, callers fall back to Hershey
rather than failing the run.
"""

from __future__ import annotations

from pathlib import Path

# Ordered preference. Each entry is (path, face index within a .ttc, name).
# Indices are verified, not guessed: in HelveticaNeue.ttc face 1 is Bold, and in
# "Avenir Next.ttc" face 0 is Bold while face 1 is Bold *Italic*, and picking by
# eye is how an overlay ends up silently slanted.
FONT_CANDIDATES: tuple[tuple[str, int, str], ...] = (
    # macOS 中文字体
    ("/System/Library/Fonts/PingFang.ttc", 0, "PingFang SC"),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0, "STHeiti Medium"),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0, "STHeiti Light"),
    ("/Library/Fonts/Arial Unicode.ttf", 0, "Arial Unicode"),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 0, "Songti SC"),
    # Linux 中文字体
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0, "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0, "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0, "WenQuanYi Micro Hei"),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0, "WenQuanYi Zen Hei"),
    # Windows 中文字体
    ("C:/Windows/Fonts/msyh.ttc", 0, "Microsoft YaHei"),
    ("C:/Windows/Fonts/simhei.ttf", 0, "SimHei"),
    # 英文高质量备选
    ("/System/Library/Fonts/Avenir Next.ttc", 0, "Avenir Next Bold"),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1, "Helvetica Neue Bold"),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, "Arial Bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, "DejaVu Sans Bold"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0, "Liberation Sans Bold"),
)


def _matplotlib_dejavu() -> tuple[str, int, str] | None:
    """DejaVu Sans Bold ships inside matplotlib, the portable last resort."""
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return (str(path), 0, "DejaVu Sans Bold (bundled)") if path.is_file() else None


def resolve_font(spec: str = "auto", index: int | None = None) -> tuple[str, int, str] | None:
    """Pick a font face. Returns None to mean "use the OpenCV fallback".

    ``spec`` is "auto" (first available candidate), "opencv" (force Hershey), or
    an explicit path to a .ttf/.otf/.ttc.
    """
    try:
        from PIL import ImageFont  # noqa: F401
    except ImportError:
        return None

    if spec == "opencv":
        return None

    if spec != "auto":
        path = Path(spec).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"PANEL_FONT points at a missing file: {path}")
        return (str(path), index or 0, path.stem)

    for path, idx, name in FONT_CANDIDATES:
        if Path(path).is_file():
            return (path, index if index is not None else idx, name)
    return _matplotlib_dejavu()
