"""Recover DEFER-Dx's in-set deferral curve (training batches, 5-step moving mean) for the PI controller
inset of Figure 2, from the vector plot docs/figures/fig_training_dynamics.svg (panel b), which
scripts/make_figures.py drew from outputs/runs/cev/train_log.jsonl. The training log is not in git
(outputs/ is ignored), so the plotted coordinates are the published record of it.

The axis mapping is checked, not assumed: the panel's six gridlines must map to 0, 20, ..., 100 and its
horizontal reference line to the 30% cap; the DEFER-Dx line (colour #2a78d6) must span steps 1 to 150.
Usage: python3 paper/figures/src/extract_training_curve.py   (writes paper/figures/src/data/cev_inset_defer.dat)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SVG = ROOT / "docs/figures/fig_training_dynamics.svg"
OUT = Path(__file__).with_name("data") / "cev_inset_defer.dat"

svg = SVG.read_text(encoding="utf-8")
panel = svg[svg.index('id="axes_2"'):svg.index('id="axes_3"')]

# x: the first vertical gridline is step 0, gridlines every 20 steps; y: the axes box spans 0 to 100%
xticks = [float(x) for x in re.findall(r'<use xlink:href="#[^"]+" x="([\d.]+)" y="[\d.]+"', panel)][:8]
x0, px_per_step = xticks[0], (xticks[1] - xticks[0]) / 20
box = [float(v) for v in re.findall(r"[\d.]+", re.search(r'<g id="patch_5">\s*<path d="([^"]+)"', panel).group(1))]
y0, y100 = box[1], box[5]


def to_data(x, y):
    return (x - x0) / px_per_step, (y0 - y) / (y0 - y100) * 100


lines = re.findall(r'<path d="([^"]+)"[^>]*style="[^"]*stroke: (#[0-9a-f]{6})', panel)
hlines = sorted({round(to_data(0, float(re.findall(r"[\d.]+", d)[1]))[1], 3)
                for d, _ in lines if len(re.findall(r"L", d)) == 1 and re.findall(r"[\d.]+", d)[1] == re.findall(r"[\d.]+", d)[3]})
assert hlines == [0.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0], hlines

blue = [d for d, colour in lines if colour == "#2a78d6"]
assert len(blue) == 1, len(blue)
points = [to_data(float(a), float(b)) for a, b in re.findall(r"([\d.]+) ([\d.]+)", blue[0])]
steps = [round(s) for s, _ in points]
assert all(abs(s - round(s)) < 0.01 for s, _ in points) and steps[0] == 1 and steps[-1] == 150, steps[:3]

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("# step  inset_defer_pct  (DEFER-Dx, training batches, 5-step moving mean; from " + SVG.name + ")\n"
               + "".join(f"{s} {r:.2f}\n" for s, (_, r) in zip(steps, points)), encoding="utf-8")
print(f"{OUT.relative_to(ROOT)}: {len(points)} points, steps {steps[0]} to {steps[-1]}, "
      f"max {max(r for _, r in points):.1f}%, mean over steps 101 to 150 "
      f"{sum(r for s, (_, r) in zip(steps, points) if s > 100) / sum(1 for s in steps if s > 100):.1f}%")
