from __future__ import annotations

import html
import math
from pathlib import Path


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SERIES = {
    "src/sota": {
        1: 136.36,
        2: 183.00,
        4: 360.15,
        8: 600.38,
        16: 945.53,
        32: 1262.38,
        64: 2066.46,
        128: 2876.35,
        256: 3451.47,
    },
    "SGLang": {
        1: 110.26,
        2: 154.86,
        4: 213.02,
        8: 311.40,
        16: 482.08,
        32: 819.97,
        64: 1359.07,
        128: 1884.23,
        256: 3387.60,
    },
    "vLLM": {
        1: 67.13,
        2: 124.02,
        4: 208.98,
        8: 301.30,
        16: 461.50,
        32: 760.38,
        64: 1291.24,
        128: 1740.49,
        256: 2944.65,
    },
    "llama.cpp": {
        1: 99.92,
        2: 155.50,
        4: 233.66,
        8: 359.21,
        16: 518.28,
        32: 772.79,
        64: 1081.21,
        128: 1363.71,
        256: 1563.54,
    },
}
COLORS = {
    "src/sota": "#d62728",
    "SGLang": "#1f77b4",
    "vLLM": "#2ca02c",
    "llama.cpp": "#9467bd",
}
MARKERS = {
    "src/sota": "circle",
    "SGLang": "square",
    "vLLM": "triangle",
    "llama.cpp": "diamond",
}


WIDTH = 1080
HEIGHT = 640
LEFT = 92
RIGHT = 280
TOP = 72
BOTTOM = 82
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM


def x_pos(batch: int) -> float:
    return LEFT + math.log2(batch) / math.log2(max(BATCHES)) * PLOT_W


def y_pos(tps: float) -> float:
    min_log = math.log10(50.0)
    max_log = math.log10(7000.0)
    return TOP + (max_log - math.log10(tps)) / (max_log - min_log) * PLOT_H


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
    y_ticks = [50, 100, 200, 500, 1000, 2000, 5000]
    elements: list[str] = []
    elements.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white" />')
    elements.append(text(WIDTH / 2, 32, "Batch Scaling", text_anchor="middle", font_size="24", font_weight="700"))
    elements.append(text(WIDTH / 2, 56, "DeepSeek-V2-Lite-Chat, RTX A6000 GPU2, input 24 tokens, output/decode 100 tokens", text_anchor="middle", font_size="14", fill="#555"))

    for batch in BATCHES:
        x = x_pos(batch)
        elements.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + PLOT_H}" stroke="#e8e8e8" stroke-width="1" />')
        elements.append(text(x, TOP + PLOT_H + 28, str(batch), text_anchor="middle", font_size="13", fill="#333"))

    for tick in y_ticks:
        y = y_pos(tick)
        elements.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1" />')
        label = f"{tick // 1000}k" if tick >= 1000 else str(tick)
        elements.append(text(LEFT - 12, y + 4, label, text_anchor="end", font_size="13", fill="#333"))

    elements.append(f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#222" stroke-width="1.2" />')
    elements.append(text(LEFT + PLOT_W / 2, HEIGHT - 24, "Batch size / parallel sequences", text_anchor="middle", font_size="16", fill="#111"))
    elements.append(f'<text x="24" y="{TOP + PLOT_H / 2:.1f}" text-anchor="middle" font-size="16" fill="#111" transform="rotate(-90 24 {TOP + PLOT_H / 2:.1f})">Throughput (tok/s, log scale)</text>')

    for label, data in SERIES.items():
        color = COLORS[label]
        points = [(x_pos(batch), y_pos(data[batch])) for batch in BATCHES if batch in data]
        elements.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />')
        for x, y in points:
            elements.append(marker_svg(MARKERS[label], x, y, color))

    legend_x = LEFT + PLOT_W + 44
    legend_y = TOP + 28
    elements.append(text(legend_x, legend_y - 18, "Backend", font_size="16", font_weight="700", fill="#111"))
    for index, (label, data) in enumerate(SERIES.items()):
        y = legend_y + index * 54
        color = COLORS[label]
        elements.append(f'<line x1="{legend_x}" y1="{y:.1f}" x2="{legend_x + 34}" y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round" />')
        elements.append(marker_svg(MARKERS[label], legend_x + 17, y, color))
        last_batch = max(data)
        suffix = f" ({data[last_batch]:.0f} tok/s @ bsz {last_batch})"
        elements.append(text(legend_x + 46, y + 5, label + suffix, font_size="14", fill="#111"))

    note_y = HEIGHT - 34
    elements.append(text(LEFT + PLOT_W + 44, note_y, "all plotted backends capped at bsz 256", font_size="12", fill="#666"))

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        "<title>Batch-size throughput scaling</title>",
        "<desc>Log-log line chart comparing src/sota, SGLang, vLLM, and llama.cpp batched-bench throughput across batch sizes.</desc>",
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
