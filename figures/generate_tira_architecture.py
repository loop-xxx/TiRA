from pathlib import Path
from math import atan2, cos, sin

from PIL import Image, ImageDraw, ImageFont


FIG_W, FIG_H = 2400, 1400
X_MAX, Y_MAX = 12.0, 7.0
FONT_SCALE = 1.15

BLUE = "#2f7ebc"
ORANGE = "#f0ad24"
GREEN = "#24a46d"
PINK = "#c979a8"
TEXT = "#222222"
MUTED = "#666666"
GRID = "#bcc7d1"


def px(x):
    return int(round(x / X_MAX * FIG_W))


def py(y):
    return int(round(FIG_H - y / Y_MAX * FIG_H))


def color(hex_color, alpha=255):
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    if alpha < 255:
        rgb = [int(round(channel * alpha / 255 + 255 * (1 - alpha / 255))) for channel in rgb]
    return tuple(rgb) + (255,)


def font(size, bold=False):
    size = int(round(size * FONT_SCALE))
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_center(draw, xy, text, size=40, fill=TEXT, bold=False):
    fnt = font(size, bold)
    x, y = px(xy[0]), py(xy[1])
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=color(fill))


def draw_text(draw, xy, text, size=36, fill=TEXT, bold=False, anchor="mm"):
    fnt = font(size, bold)
    draw.text((px(xy[0]), py(xy[1])), text, font=fnt, fill=color(fill), anchor=anchor)


def draw_segments_center(draw, xy, segments, default_fill=TEXT):
    rendered = []
    total_width = 0
    max_height = 0
    for text, size, y_offset, *rest in segments:
        fill = rest[0] if len(rest) >= 1 else default_fill
        bold = rest[1] if len(rest) >= 2 else False
        fnt = font(size, bold)
        box = draw.textbbox((0, 0), text, font=fnt)
        width = box[2] - box[0]
        height = box[3] - box[1]
        rendered.append((text, fnt, width, height, y_offset, fill))
        total_width += width
        max_height = max(max_height, height + abs(y_offset))

    x = px(xy[0]) - total_width / 2
    center_y = py(xy[1])
    for text, fnt, width, height, y_offset, fill in rendered:
        draw.text((x, center_y - height / 2 + y_offset), text, font=fnt, fill=color(fill))
        x += width


def draw_label(draw, xy, label, size=36, fill=TEXT):
    if isinstance(label, list):
        draw_segments_center(draw, xy, label, default_fill=fill)
    else:
        draw_text(draw, xy, label, size=size, fill=fill)


def rect(draw, x, y, w, h, fill="white", outline=TEXT, width=4, alpha=255):
    box = [px(x), py(y + h), px(x + w), py(y)]
    if fill == "white":
        fill_rgba = color("#ffffff", alpha)
    elif fill is None:
        fill_rgba = None
    else:
        fill_rgba = color(fill, alpha)
    outline_rgba = None if outline is None else color(outline)
    draw.rectangle(box, fill=fill_rgba, outline=outline_rgba, width=width)


def line(draw, start, end, fill=TEXT, width=4):
    draw.line([px(start[0]), py(start[1]), px(end[0]), py(end[1])], fill=color(fill), width=width)


def arrow(draw, start, end, fill=TEXT, width=5):
    x1, y1 = px(start[0]), py(start[1])
    x2, y2 = px(end[0]), py(end[1])
    draw.line([x1, y1, x2, y2], fill=color(fill), width=width)
    ang = atan2(y2 - y1, x2 - x1)
    length = 28
    spread = 0.45
    p1 = (x2 - length * cos(ang - spread), y2 - length * sin(ang - spread))
    p2 = (x2 - length * cos(ang + spread), y2 - length * sin(ang + spread))
    draw.polygon([(x2, y2), p1, p2], fill=color(fill))


def grid(draw, x, y, w, h, rows, cols, fill=GRID, width=2):
    for i in range(1, cols):
        xi = x + w * i / cols
        line(draw, (xi, y), (xi, y + h), fill=fill, width=width)
    for j in range(1, rows):
        yj = y + h * j / rows
        line(draw, (x, yj), (x + w, yj), fill=fill, width=width)


def matrix(draw, x, y, w, h, rows, cols, fill, label, sublabel):
    rect(draw, x, y, w, h, fill=fill, outline=fill, width=4, alpha=36)
    grid(draw, x, y, w, h, rows, cols, width=2)
    draw_label(draw, (x + w / 2, y + h + 0.22), label, size=50)
    draw_label(draw, (x + w / 2, y - 0.18), sublabel, size=32, fill=MUTED)


def tira_block_generator(draw, x, y):
    bx, by, bw, bh = x, y + 0.18, 0.24, 1.20
    axx, ay, aw, ah = x + 1.00, y + 0.66, 1.20, 0.24
    cx, cy, cs = x + 2.90, y + 0.28, 1.00

    matrix(draw, bx, by, bw, bh, 5, 1, BLUE, [("b", 50, 0), ("k,m", 30, 16)], [("n", 32, 0), ("out", 22, 10), ("=d", 32, 0), ("out", 22, 10), ("/M", 32, 0)])
    draw_text(draw, (x + 0.62, y + 0.78), "x", size=64)
    matrix(draw, axx, ay, aw, ah, 1, 5, ORANGE, [("a", 50, 0), ("k,m", 30, 16), ("T", 28, -20)], [("n", 32, 0), ("in", 22, 10), ("=d", 32, 0), ("in", 22, 10), ("/M", 32, 0)])
    draw_text(draw, (x + 2.55, y + 0.78), "=", size=64)

    rect(draw, cx, cy, cs, cs, fill=GREEN, outline=GREEN, width=4, alpha=46)
    grid(draw, cx, cy, cs, cs, 5, 5, width=2)
    draw_segments_center(draw, (cx + cs / 2, cy + cs + 0.18), [("C", 48, 0), ("k,m", 28, 15)])
    return cx + cs / 2, cy + cs / 2


def delta_wk(draw, x, y, size, shift, highlight_row=1):
    rect(draw, x, y, size, size, fill="white", outline=TEXT, width=4)
    grid(draw, x, y, size, size, 4, 4, width=2)
    cell = size / 4
    pad = cell * 0.08
    highlight_center = None
    for m in range(4):
        col = (m + shift) % 4
        rx = x + col * cell + pad
        ry = y + (3 - m) * cell + pad
        block_color = GREEN if m == highlight_row else BLUE
        rect(draw, rx, ry, cell - 2 * pad, cell - 2 * pad, fill=block_color, outline=None, width=0, alpha=230)
        if m == highlight_row:
            rect(draw, rx, ry, cell - 2 * pad, cell - 2 * pad, fill=None, outline=GREEN, width=6)
            highlight_center = (x + col * cell + cell / 2, y + (3 - m) * cell + cell / 2)
    draw_segments_center(draw, (x + size / 2, y + size + 0.18), [("ΔW", 44, 0), ("k", 28, 14)])
    draw_segments_center(draw, (x + size / 2, y - 0.18), [("rank(ΔW", 32, 0, MUTED), ("k", 22, 10, MUTED), (")=M", 32, 0, MUTED)], default_fill=MUTED)
    return highlight_center


def shifted_group(draw, x, y, size, shift, fill, label):
    rect(draw, x, y, size, size, fill="white", outline=TEXT, width=3)
    grid(draw, x, y, size, size, 4, 4, width=2)
    cell = size / 4
    pad = cell * 0.10
    for m in range(4):
        col = (m + shift) % 4
        rect(
            draw,
            x + col * cell + pad,
            y + (3 - m) * cell + pad,
            cell - 2 * pad,
            cell - 2 * pad,
            fill=fill,
            outline=None,
            width=0,
            alpha=240,
        )
    draw_label(draw, (x + size / 2, y - 0.20), label, size=30, fill=MUTED)


def block_delta(draw, x, y, size):
    colors = [BLUE, ORANGE, GREEN, PINK]
    rect(draw, x, y, size, size, fill="white", outline=TEXT, width=5)
    grid(draw, x, y, size, size, 4, 4, width=2)
    cell = size / 4
    pad = cell * 0.06
    for m in range(4):
        for k, block_color in enumerate(colors):
            col = (m + k) % 4
            rect(
                draw,
                x + col * cell + pad,
                y + (3 - m) * cell + pad,
                cell - 2 * pad,
                cell - 2 * pad,
                fill=block_color,
                outline=None,
                width=0,
                alpha=235,
            )
    draw_segments_center(draw, (x + size / 2, y + size + 0.20), [("ΔW", 48, 0), ("TiRA", 30, 15)])


def build():
    img = Image.new("RGBA", (FIG_W, FIG_H), color("#ffffff", 255))
    draw = ImageDraw.Draw(img)

    draw_text(draw, (6.00, 6.45), "TiRA", size=84, bold=True)

    c_center = tira_block_generator(draw, 0.85, 4.02)
    wk_center = delta_wk(draw, 8.25, 4.22, 1.06, shift=1, highlight_row=1)
    arrow(draw, (c_center[0] + 0.50, c_center[1]), (wk_center[0] - 0.10, wk_center[1] + 0.02), fill=GREEN, width=5)
    draw_segments_center(
        draw,
        ((c_center[0] + wk_center[0]) / 2, c_center[1] - 0.38),
        [("place C", 28, 0, GREEN), ("k,m", 20, 10, GREEN), (" at block (m, (m+k) mod M)", 28, 0, GREEN)],
        default_fill=GREEN,
    )
    draw_text(draw, (6.00, 3.76), "k: group index (0 <= k < K); m: block-row index (0 <= m < M)", size=42, fill=MUTED)

    shifted_group(draw, 1.02, 1.88, 0.82, 0, BLUE, [("ΔW", 30, 0, MUTED), ("0", 20, 10, MUTED)])
    draw_text(draw, (1.97, 2.29), "+", size=52)
    shifted_group(draw, 2.12, 1.88, 0.82, 1, ORANGE, [("ΔW", 30, 0, MUTED), ("1", 20, 10, MUTED)])
    draw_text(draw, (3.07, 2.29), "+", size=52)
    shifted_group(draw, 3.22, 1.88, 0.82, 2, GREEN, [("ΔW", 30, 0, MUTED), ("2", 20, 10, MUTED)])
    draw_text(draw, (4.72, 2.29), "+ ... =>", size=46)
    block_delta(draw, 6.08, 1.55, 1.40)
    draw_segments_center(draw, (6.78, 1.22), [("rank(ΔW", 36, 0, MUTED), ("TiRA", 24, 12, MUTED), (") <= KM", 36, 0, MUTED)], default_fill=MUTED)

    return img


if __name__ == "__main__":
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    image = build()
    image.save(out_dir / "tira_architecture.png")
    image.convert("RGB").save(out_dir / "tira_architecture.pdf", "PDF", resolution=200.0)
