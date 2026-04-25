from __future__ import annotations

import html
import math
from pathlib import Path


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SERIES = {
    "src/batch": {
        1: 154.03,
        2: 270.65,
        4: 518.80,
        8: 958.35,
        16: 1914.46,
        32: 3608.94,
        64: 6237.99,
        128: 9109.06,
        256: 11428.26,
    },
    "src/small GEMV": {
        1: 197.98,
        2: 272.59,
        4: 460.22,
        8: 614.82,
        16: 802.99,
        32: 918.80,
        64: 1006.51,
        128: 1057.46,
        256: 1089.71,
    },
    "SGLang": {
        1: 165.16,
        2: 256.48,
        4: 379.13,
        8: 599.85,
        16: 953.28,
        32: 1585.99,
        64: 2833.83,
        128: 3945.98,
        256: 7094.12,
    },
    "vLLM": {
        1: 78.44,
        2: 185.52,
        4: 390.01,
        8: 585.65,
        16: 930.27,
        32: 1549.51,
        64: 2638.93,
        128: 3655.33,
        256: 6113.90,
    },
    "llama.cpp": {
        1: 137.01,
        2: 226.56,
        4: 367.57,
        8: 593.93,
        16: 849.41,
        32: 1285.17,
        64: 1838.71,
        128: 2389.42,
        256: 2051.89,
    },
}
COLORS = {
    "src/batch": "#d62728",
    "src/small GEMV": "#ff7f0e",
    "SGLang": "#1f77b4",
    "vLLM": "#2ca02c",
    "llama.cpp": "#9467bd",
}
MARKERS = {
    "src/batch": "circle",
    "src/small GEMV": "circle",
    "SGLang": "square",
    "vLLM": "triangle",
    "llama.cpp": "diamond",
}
LEGEND_LABELS = {
    "src/batch": "batch",
    "src/small GEMV": "small",
    "SGLang": "SGLang",
    "vLLM": "vLLM",
    "llama.cpp": "llama.cpp",
}


WIDTH = 760
HEIGHT = 640
LEFT = 72
RIGHT = 33
TOP = 100
BOTTOM = 72
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM


def x_pos(batch: int) -> float:
    return LEFT + math.log2(batch) / math.log2(max(BATCHES)) * PLOT_W


Y_MIN = 64.0
Y_MAX = 16384.0
Y_TICKS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def y_pos(tps: float) -> float:
    min_log = math.log2(Y_MIN)
    max_log = math.log2(Y_MAX)
    return TOP + (max_log - math.log2(tps)) / (max_log - min_log) * PLOT_H


def marker_svg(kind: str, x: float, y: float, color: str) -> str:
    if kind == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="{color}" />'
    if kind == "triangle":
        points = f"{x:.1f},{y - 6:.1f} {x - 6:.1f},{y + 5:.1f} {x + 6:.1f},{y + 5:.1f}"
        return f'<polygon points="{points}" fill="{color}" />'
    points = f"{x:.1f},{y - 6:.1f} {x - 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x + 6:.1f},{y:.1f}"
    return f'<polygon points="{points}" fill="{color}" />'


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def text(x: float, y: float, value: str, **attrs: str) -> str:
    attr_text = " ".join(f'{key.replace("_", "-")}="{html.escape(str(val))}"' for key, val in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attr_text}>{html.escape(value)}</text>'


def build_svg() -> str:
    elements: list[str] = []
    elements.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#fff" />')
    elements.append(text(WIDTH / 2, 28, "Batch Scaling", text_anchor="middle", font_size="20", font_weight="700", fill="#111"))
    elements.append(text(WIDTH / 2, 50, "DeepSeek-V2-Lite-Chat, A100 80GB, input 24 tokens, output 100 tokens", text_anchor="middle", font_size="14", fill="#444"))

    legend_y = 76
    legend_w = 620
    legend_left = LEFT + PLOT_W / 2 - legend_w / 2
    legend_gap = legend_w / len(SERIES)
    legend_x = legend_left + 18
    elements.append(f'<rect x="{legend_left:.1f}" y="{legend_y - 17}" width="{legend_w}" height="30" rx="4" ry="4" fill="#fff" stroke="#cccccc" stroke-width="1" opacity="0.95" />')
    for index, label in enumerate(SERIES):
        x = legend_x + index * legend_gap
        color = COLORS[label]
        elements.append(f'<line x1="{x:.1f}" y1="{legend_y:.1f}" x2="{x + 26:.1f}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round" />')
        elements.append(marker_svg(MARKERS[label], x + 13, legend_y, color))
        elements.append(text(x + 32, legend_y + 5, LEGEND_LABELS[label], font_size="14", fill="#111"))

    for batch in BATCHES:
        x = x_pos(batch)
        elements.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + PLOT_H}" stroke="#e8e8e8" stroke-width="1" />')
        elements.append(text(x, TOP + PLOT_H + 25, str(batch), text_anchor="middle", font_size="18", fill="#111"))

    for tick in Y_TICKS:
        y = y_pos(float(tick))
        elements.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1" />')
        label = f"{tick // 1024}k" if tick >= 1024 else str(tick)
        elements.append(text(LEFT - 12, y + 6, label, text_anchor="end", font_size="18", fill="#111"))

    elements.append(f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#222" stroke-width="1.2" />')
    elements.append(text(LEFT + PLOT_W / 2, TOP + PLOT_H + 54, "Batch size / parallel sequences", text_anchor="middle", font_size="20", fill="#111"))
    elements.append(f'<text x="24" y="{TOP + PLOT_H / 2:.1f}" text-anchor="middle" font-size="20" fill="#111" transform="rotate(-90 24 {TOP + PLOT_H / 2:.1f})">Throughput (tok/s, log scale)</text>')

    for label, data in SERIES.items():
        color = COLORS[label]
        points = [(x_pos(batch), y_pos(data[batch])) for batch in BATCHES if batch in data]
        elements.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />')
        for x, y in points:
            elements.append(marker_svg(MARKERS[label], x, y, color))

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" style="font-family: Times New Roman, DejaVu Serif, serif;">',
        "<title>Batch-size throughput scaling</title>",
        "<desc>Log-log line chart comparing src/run.py kernel families, SGLang, vLLM, and llama.cpp batched-bench throughput across batch sizes.</desc>",
        *elements,
        "</svg>",
        "",
    ])


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / "batch_scaling.svg"
    svg.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
