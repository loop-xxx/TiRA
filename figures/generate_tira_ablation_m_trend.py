from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent

MODELS = [
    {
        "name": "Qwen-2-0.5B",
        "hidden": 896,
        "lora": 38.59,
        "scores": {2: 39.20, 4: 38.82, 8: 36.54},
    },
    {
        "name": "Qwen-2-1.5B",
        "hidden": 1536,
        "lora": 68.08,
        "scores": {2: 68.16, 4: 68.39, 8: 68.84, 16: 68.01},
    },
    {
        "name": "Qwen-2.5-3B",
        "hidden": 2048,
        "lora": 74.30,
        "scores": {8: 74.53, 16: 75.13, 32: 75.44},
    },
]

K_BUDGET = 32
X_TICKS = [1000, 1500, 2000]
X_MIN, X_MAX = 800, 2100
Y_TICKS = [0, 10, 20, 30, 40, 50]
Y_MIN, Y_MAX = 0, 55


def font(size, bold=False):
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    names += ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def lerp(a, b, t):
    return int(round(a + (b - a) * t))


def mix(c1, c2, t):
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def gain_color(gain):
    neg = (216, 95, 39)
    zero = (247, 247, 247)
    pos = (26, 152, 80)
    if gain < 0:
        return mix(neg, zero, max(0.0, min(1.0, (gain + 2.1) / 2.1)))
    return mix(zero, pos, max(0.0, min(1.0, gain / 1.2)))


def text_center(draw, xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2 - 1), text, font=fnt, fill=fill)


def text_with_bg(draw, xy, text, fnt, fill, pad=8):
    x, y = xy
    box = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle(
        (box[0] - pad, box[1] - pad // 2, box[2] + pad, box[3] + pad // 2),
        fill="white",
    )
    draw.text((x, y), text, font=fnt, fill=fill)


def draw_dashed_line(draw, p1, p2, fill, width=4, dash=18, gap=12):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x1 + dx * pos, y1 + dy * pos), (x1 + dx * end, y1 + dy * end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def draw_arrow(draw, start, end, fill, width=4):
    draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 18
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=fill)


W, H = 1800, 980
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

plot_l, plot_t, plot_r, plot_b = 360, 95, 1430, 735
data_t, data_b = plot_t + 72, plot_b - 72
axis_color = (60, 60, 60)
grid_color = (229, 229, 229)
text_color = (32, 32, 32)
muted = (92, 92, 92)

f_label = font(44)
f_tick = font(36)
f_small = font(34)
f_mlabel = font(24)
f_score = font(26, True)
f_best = font(36, True)


def rank_ratio(model, m):
    return K_BUDGET * m / model["hidden"] * 100


def x_pos(hidden):
    return plot_l + (hidden - X_MIN) / (X_MAX - X_MIN) * (plot_r - plot_l)


def y_pos(ratio):
    return data_b - (ratio - Y_MIN) / (Y_MAX - Y_MIN) * (data_b - data_t)


for tick in X_TICKS:
    x = x_pos(tick)
    draw.line([(x, plot_t), (x, plot_b)], fill=grid_color, width=2)

for tick in Y_TICKS:
    y = y_pos(tick)
    draw.line([(plot_l, y), (plot_r, y)], fill=grid_color, width=2)
    text_center(draw, (plot_l - 45, y), str(tick), f_small, muted)

for model in MODELS:
    x = x_pos(model["hidden"])
    draw.line([(x, plot_t), (x, plot_b)], fill=grid_color, width=2)
    label = f"{model['name']}\n(hidden {model['hidden']})"
    lines = label.split("\n")
    text_center(draw, (x, plot_b + 48), lines[0], f_mlabel, text_color)
    text_center(draw, (x, plot_b + 77), lines[1], f_mlabel, muted)

draw.line([(plot_l, plot_b), (plot_r, plot_b)], fill=axis_color, width=3)
draw.line([(plot_l, plot_t), (plot_l, plot_b)], fill=axis_color, width=3)
draw.text((plot_l, plot_t - 48), "KM / hidden size (%)", font=f_small, fill=text_color)

best_points = []
for model in MODELS:
    x = x_pos(model["hidden"])
    best_m = max(model["scores"], key=lambda m: model["scores"][m])
    best_score = model["scores"][best_m]
    for m, score in model["scores"].items():
        gain = score - model["lora"]
        ratio = rank_ratio(model, m)
        y = y_pos(ratio)
        r = 34
        fill = gain_color(gain)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline="white", width=4)
        text_center(draw, (x, y), f"{score:.2f}", f_score, text_color)
        if m != best_m:
            if x > plot_r - 120:
                draw.text((x - 104, y - 14), f"M={m}", font=f_mlabel, fill=muted)
            else:
                draw.text((x + 44, y - 14), f"M={m}", font=f_mlabel, fill=muted)
    best_ratio = rank_ratio(model, best_m)
    bx, by = x, y_pos(best_ratio)
    best_points.append((bx, by, best_m, best_score, best_ratio))

for p1, p2 in zip(best_points, best_points[1:]):
    draw_dashed_line(draw, (p1[0], p1[1]), (p2[0], p2[1]), fill=(70, 70, 70), width=5)

for bx, by, best_m, best_score, best_ratio in best_points:
    r = 48
    draw.ellipse((bx - r, by - r, bx + r, by + r), outline=(20, 20, 20), width=6)
    lines = [f"{best_ratio:.1f}%", f"(M={best_m})"]
    widths = [draw.textbbox((0, 0), line, font=f_best)[2] for line in lines]
    tw = max(widths)
    if bx > plot_r - 220:
        x_text = bx - tw - 60
    else:
        x_text = bx + 58
    y_text = by - 44
    for idx, line in enumerate(lines):
        draw.text((x_text, y_text + idx * 40), line, font=f_best, fill=text_color)

text_center(draw, ((plot_l + plot_r) / 2, plot_b + 122), "Hidden size", f_label, text_color)

cb_l, cb_t, cb_w, cb_h = 1510, 235, 46, 390
for j in range(cb_h):
    t = 1 - j / (cb_h - 1)
    gain = -2.1 + t * (1.2 + 2.1)
    draw.line([(cb_l, cb_t + j), (cb_l + cb_w, cb_t + j)], fill=gain_color(gain), width=1)
draw.rectangle((cb_l, cb_t, cb_l + cb_w, cb_t + cb_h), outline=(80, 80, 80), width=2)
draw.text((cb_l - 38, cb_t - 38), "Gain over LoRA (pp)", font=f_small, fill=text_color)
for gain in [-2, -1, 0, 0.5, 1]:
    t = (gain + 2.1) / 3.3
    y = cb_t + cb_h - t * cb_h
    draw.line([(cb_l + cb_w, y), (cb_l + cb_w + 10, y)], fill=axis_color, width=2)
    label = f"{gain:+.1f}" if isinstance(gain, float) and not gain.is_integer() else f"{gain:+.0f}"
    draw.text((cb_l + cb_w + 18, y - 12), label, font=f_small, fill=text_color)

png_path = OUT_DIR / "tira_ablation_m_trend.png"
pdf_path = OUT_DIR / "tira_ablation_m_trend.pdf"
tmp_png = OUT_DIR / "_tira_ablation_m_trend_tmp.png"
tmp_pdf = OUT_DIR / "_tira_ablation_m_trend_tmp.pdf"
img.save(tmp_png)
img.save(tmp_pdf, "PDF", resolution=220)
tmp_png.replace(png_path)
tmp_pdf.replace(pdf_path)
