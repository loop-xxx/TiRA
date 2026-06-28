from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent

POINTS = [
    {"name": "Qwen-2-0.5B", "hidden": 896, "gain": 0.61, "m": 2},
    {"name": "Qwen-2-1.5B", "hidden": 1536, "gain": 0.76, "m": 8},
    {"name": "Qwen-2.5-3B", "hidden": 2048, "gain": 1.14, "m": 32},
    {"name": "Llama-3-8B", "hidden": 4096, "gain": 6.97, "m": 32},
]


def font(size, bold=False):
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    names += ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


W, H = 1800, 1120
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

plot_l, plot_t, plot_r, plot_b = 170, 80, 1640, 875
x_min, x_max = 750, 4250
y_min, y_max = 0, 8

axis_color = (45, 45, 45)
grid_color = (226, 226, 226)
line_color = (43, 94, 143)
text_color = (26, 26, 26)
muted = (82, 82, 82)

f_label = font(46)
f_tick = font(36)
f_small = font(30)
f_point = font(32, True)


def x_pos(hidden):
    return plot_l + (hidden - x_min) / (x_max - x_min) * (plot_r - plot_l)


def y_pos(gain):
    return plot_b - (gain - y_min) / (y_max - y_min) * (plot_b - plot_t)


def text_center(xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


for y_tick in range(0, 9):
    y = y_pos(y_tick)
    draw.line([(plot_l, y), (plot_r, y)], fill=grid_color, width=2)
    text_center((plot_l - 55, y), str(y_tick), f_tick, muted)

for point in POINTS:
    x = x_pos(point["hidden"])
    draw.line([(x, plot_t), (x, plot_b)], fill=grid_color, width=2)
    text_center((x, plot_b + 48), str(point["hidden"]), f_tick, text_color)

draw.line([(plot_l, plot_b), (plot_r, plot_b)], fill=axis_color, width=4)
draw.line([(plot_l, plot_t), (plot_l, plot_b)], fill=axis_color, width=4)

coords = [(x_pos(p["hidden"]), y_pos(p["gain"])) for p in POINTS]
draw.line(coords, fill=line_color, width=8)
for x, y in coords:
    r = 15
    draw.ellipse((x - r, y - r, x + r, y + r), fill=line_color, outline=line_color)

for point, (x, y) in zip(POINTS, coords):
    label = f"{point['name']}\nM={point['m']}\n+{point['gain']:.2f}"
    lines = label.split("\n")
    if point["hidden"] == 896:
        tx, ty = x + 25, y - 130
    elif point["hidden"] == 4096:
        tx, ty = x - 210, y - 105
    elif point["hidden"] == 2048:
        tx, ty = x - 50, y - 112
    else:
        tx, ty = x - 42, y - 145
    for i, line in enumerate(lines):
        draw.text((tx, ty + i * 34), line, font=f_small if i < 2 else f_point, fill=text_color)

text_center(((plot_l + plot_r) / 2, plot_b + 125), "Hidden size", f_label, text_color)
draw.text((34, 34), "Best TiRA gain over LoRA (pp)", font=f_label, fill=text_color)

png_path = OUT_DIR / "tira_best_gain_vs_scale.png"
pdf_path = OUT_DIR / "tira_best_gain_vs_scale.pdf"
tmp_png = OUT_DIR / "_tira_best_gain_vs_scale_tmp.png"
tmp_pdf = OUT_DIR / "_tira_best_gain_vs_scale_tmp.pdf"
img.save(tmp_png)
img.save(tmp_pdf, "PDF", resolution=220)
tmp_png.replace(png_path)
tmp_pdf.replace(pdf_path)
