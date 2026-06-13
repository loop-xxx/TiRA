import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


FIG_W, FIG_H = 16, 7

BLUE = "#2f7ebc"
ORANGE = "#f0ad24"
GREEN = "#24a46d"
PINK = "#c979a8"
TEXT = "#222222"
MUTED = "#666666"
GRID = "#bcc7d1"


def rect(ax, x, y, w, h, fc="white", ec=TEXT, lw=2, alpha=1.0):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, lw=2.2, color=TEXT, linestyle="-", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=lw,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
        )
    )


def grid(ax, x, y, w, h, rows, cols, color=GRID, lw=0.75):
    for i in range(1, cols):
        xi = x + w * i / cols
        ax.plot([xi, xi], [y, y + h], color=color, lw=lw)
    for j in range(1, rows):
        yj = y + h * j / rows
        ax.plot([x, x + w], [yj, yj], color=color, lw=lw)


def matrix(ax, x, y, w, h, rows, cols, color, label, sublabel, label_size=28, sublabel_size=14):
    rect(ax, x, y, w, h, fc=color, ec=color, lw=2.5, alpha=0.14)
    grid(ax, x, y, w, h, rows, cols)
    ax.text(x + w / 2, y + h + 0.16, label, ha="center", va="bottom", fontsize=label_size, color=TEXT)
    ax.text(x + w / 2, y - 0.18, sublabel, ha="center", va="top", fontsize=sublabel_size, color=MUTED)


def dense_delta(ax, x, y, cell, rows, cols, title):
    w, h = cols * cell, rows * cell
    rect(ax, x, y, w, h, fc=GREEN, ec=TEXT, lw=2.6, alpha=0.18)
    for i in range(rows):
        for j in range(cols):
            rect(
                ax,
                x + j * cell + 0.006,
                y + (rows - 1 - i) * cell + 0.006,
                cell - 0.012,
                cell - 0.012,
                fc=GREEN,
                ec="none",
                alpha=0.18,
            )
    grid(ax, x, y, w, h, rows, cols, color="#d7dde3", lw=0.5)
    ax.text(x + w / 2, y + h + 0.18, title, ha="center", va="bottom", fontsize=25, color=TEXT)
    ax.text(x + w / 2, y - 0.18, r"$\mathrm{rank}(\Delta W_{\mathrm{LoRA}})\leq r$", ha="center", va="top", fontsize=14, color=MUTED)


def block_delta(ax, x, y, size, highlight=None):
    colors = [BLUE, ORANGE, GREEN, PINK]
    rect(ax, x, y, size, size, fc="white", ec=TEXT, lw=2.6)
    cell = size / 4
    for i in range(1, 4):
        ax.plot([x + i * cell, x + i * cell], [y, y + size], color=GRID, lw=1.0)
        ax.plot([x, x + size], [y + i * cell, y + i * cell], color=GRID, lw=1.0)
    pad = cell * 0.06
    for m in range(4):
        for k, color in enumerate(colors):
            col = (m + k) % 4
            rx = x + col * cell + pad
            ry = y + (3 - m) * cell + pad
            rect(ax, rx, ry, cell - 2 * pad, cell - 2 * pad, fc=color, ec="none", alpha=0.9)
    highlight_center = None
    if highlight is not None:
        row, col = highlight
        hx = x + col * cell
        hy = y + (3 - row) * cell
        rect(ax, hx + pad * 0.45, hy + pad * 0.45, cell - pad * 0.9, cell - pad * 0.9, fc="none", ec=GREEN, lw=3.2)
        highlight_center = (hx + cell / 2, hy + cell / 2)
    ax.text(x + size / 2, y + size + 0.18, r"$\Delta W_{\mathrm{TiRA}}$", ha="center", va="bottom", fontsize=28, color=TEXT)
    return highlight_center


def delta_wk(ax, x, y, size, shift, highlight_row=1):
    rect(ax, x, y, size, size, fc="white", ec=TEXT, lw=2.3)
    grid(ax, x, y, size, size, 4, 4, color=GRID, lw=0.9)
    cell = size / 4
    pad = cell * 0.08
    highlight_center = None
    for m in range(4):
        col = (m + shift) % 4
        rx = x + col * cell
        ry = y + (3 - m) * cell
        is_highlight = m == highlight_row
        rect(
            ax,
            rx + pad,
            ry + pad,
            cell - 2 * pad,
            cell - 2 * pad,
            fc=GREEN if is_highlight else BLUE,
            ec=GREEN if is_highlight else "none",
            lw=2.4 if is_highlight else 0,
            alpha=0.92 if is_highlight else 0.70,
        )
        if is_highlight:
            highlight_center = (rx + cell / 2, ry + cell / 2)
    ax.text(x + size / 2, y + size + 0.14, r"$\Delta W_k$", ha="center", va="bottom", fontsize=22, color=TEXT)
    ax.text(x + size / 2, y - 0.12, r"$\mathrm{rank}(\Delta W_k)=M$", ha="center", va="top", fontsize=12.4, color=MUTED)
    return highlight_center


def tira_block_generator(ax, x, y):
    bx, by, bw, bh = x, y + 0.18, 0.24, 1.20
    axx, ay, aw, ah = x + 1.00, y + 0.66, 1.20, 0.24
    cx, cy, cs = x + 2.90, y + 0.28, 1.00

    matrix(ax, bx, by, bw, bh, 5, 1, BLUE, r"$b_{k,m}$", r"$n_{\mathrm{out}}=d_{\mathrm{out}}/M$", label_size=23, sublabel_size=12)
    ax.text(x + 0.62, y + 0.78, r"$\times$", ha="center", va="center", fontsize=34, color=TEXT)
    matrix(ax, axx, ay, aw, ah, 1, 5, ORANGE, r"$a_{k,m}^{\top}$", r"$n_{\mathrm{in}}=d_{\mathrm{in}}/M$", label_size=23, sublabel_size=12)
    ax.text(x + 2.55, y + 0.78, r"$=$", ha="center", va="center", fontsize=32, color=TEXT)

    rect(ax, cx, cy, cs, cs, fc=GREEN, ec=GREEN, lw=2.3, alpha=0.18)
    grid(ax, cx, cy, cs, cs, 5, 5, color=GRID, lw=0.7)
    ax.text(cx + cs / 2, cy + cs + 0.04, r"$C_{k,m}$", ha="center", va="bottom", fontsize=23, color=TEXT)
    return cx + cs / 2, cy + cs / 2


def shifted_group(ax, x, y, size, shift, color, label):
    rect(ax, x, y, size, size, fc="white", ec=TEXT, lw=2.0)
    grid(ax, x, y, size, size, 4, 4, color=GRID, lw=0.85)
    cell = size / 4
    pad = cell * 0.10
    for m in range(4):
        col = (m + shift) % 4
        rect(
            ax,
            x + col * cell + pad,
            y + (3 - m) * cell + pad,
            cell - 2 * pad,
            cell - 2 * pad,
            fc=color,
            ec="none",
            alpha=0.95,
        )
    ax.text(x + size / 2, y - 0.12, label, ha="center", va="top", fontsize=13, color=MUTED)


def build():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(3.05, 6.45, "LoRA", ha="center", va="center", fontsize=40, fontweight="bold", color=TEXT)
    ax.text(11.55, 6.45, "TiRA", ha="center", va="center", fontsize=40, fontweight="bold", color=TEXT)
    ax.text(11.55, 6.05, r"$k$: group index $(0\leq k<K)$; $m$: block-row index $(0\leq m<M)$", ha="center", va="center", fontsize=14, color=MUTED)
    ax.plot([6.95, 6.95], [0.62, 6.65], color=TEXT, lw=1.8)

    lora_cell = 0.24
    matrix(ax, 0.72, 3.35, 3 * lora_cell, 6 * lora_cell, 6, 3, BLUE, r"$B$", r"$d_{\mathrm{out}}\times r$")
    ax.text(1.65, 4.07, r"$\times$", ha="center", va="center", fontsize=36, color=TEXT)
    matrix(ax, 2.00, 3.71, 6 * lora_cell, 3 * lora_cell, 3, 6, ORANGE, r"$A$", r"$r\times d_{\mathrm{in}}$")
    ax.text(3.78, 4.07, r"$=$", ha="center", va="center", fontsize=36, color=TEXT)
    dense_delta(ax, 4.25, 3.35, 0.26, 6, 6, r"$\Delta W_{\mathrm{LoRA}}$")


    c_center = tira_block_generator(ax, 7.65, 4.02)
    wk_block_center = delta_wk(ax, 13.55, 4.22, 1.06, shift=1, highlight_row=1)
    arrow(
        ax,
        (c_center[0] + 0.50, c_center[1]),
        (wk_block_center[0] - 0.10, wk_block_center[1] + 0.02),
        lw=2.0,
        color=GREEN,
        linestyle="--",
        connectionstyle="arc3,rad=0.0",
    )
    ax.text(12.70, 5.82, r"place $C_{k,m}$ at block $(m,(m+k)\ \mathrm{mod}\ M)$", ha="center", va="center", fontsize=12, color=GREEN)

    shifted_group(ax, 7.95, 1.88, 0.82, 0, BLUE, r"$\Delta W_0$")
    ax.text(8.90, 2.29, r"$+$", ha="center", va="center", fontsize=26, color=TEXT)
    shifted_group(ax, 9.05, 1.88, 0.82, 1, ORANGE, r"$\Delta W_1$")
    ax.text(10.00, 2.29, r"$+$", ha="center", va="center", fontsize=26, color=TEXT)
    shifted_group(ax, 10.15, 1.88, 0.82, 2, GREEN, r"$\Delta W_2$")
    ax.text(11.62, 2.29, r"$+\ \cdots\ \Longrightarrow$", ha="center", va="center", fontsize=23, color=TEXT)
    block_delta(ax, 12.95, 1.55, 1.40)

    ax.text(13.65, 1.30, r"$\mathrm{rank}(\Delta W_{\mathrm{TiRA}})\leq KM$", ha="center", va="top", fontsize=14, color=MUTED)

    fig.tight_layout(pad=0)
    return fig


if __name__ == "__main__":
    fig = build()
    fig.savefig("figures/lora_tira_architecture.png", bbox_inches="tight")
    fig.savefig("figures/lora_tira_architecture_updated.pdf", bbox_inches="tight")
    try:
        fig.savefig("figures/lora_tira_architecture.pdf", bbox_inches="tight")
    except PermissionError:
        pass
