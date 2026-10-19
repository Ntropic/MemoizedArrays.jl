"""Generate the sampling and radix-slot SVG diagrams with Matplotlib."""
from pathlib import Path
from random import Random
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

ASSETS = Path(__file__).resolve().parent / "src" / "assets"
GRID = "#7c8590"
BLUE = "#3684b7"
CENTRAL = "#299b87"
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "memoizedarrays"})


def save(fig, name, title):
    path = ASSETS / name
    fig.savefig(path, transparent=True, metadata={"Date": None, "Title": title})
    plt.close(fig)
    # Keep every mark vector-based: no embedded raster images in the SVG.
    assert not list(ET.parse(path).iter("{http://www.w3.org/2000/svg}image"))
    print(path)


def sampling():
    rng = Random(2026)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.12)
    for ax, clustered in zip(axes, (False, True)):
        for x in range(5):
            for y in range(5):
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor=GRID, lw=0.7))
        ax.add_patch(Rectangle((2, 2), 1, 1, facecolor=CENTRAL, alpha=0.12, edgecolor="none"))
        ax.add_patch(Rectangle((2, 2), 1, 1, fill=False, edgecolor=CENTRAL, lw=1.8))
        points = [((2 if clustered else x) + rng.uniform(0.07, 0.93),
                   (2 if clustered else y) + rng.uniform(0.07, 0.93))
                  for x in range(5) for y in range(5) for _ in range(8)]
        ax.scatter(*zip(*points), s=8, color=BLUE, linewidths=0)
        ax.set(xlim=(-0.06, 5.06), ylim=(-0.06, 5.06), aspect="equal")
        ax.set_axis_off()
    save(fig, "tree-sampling.svg", "Evenly spread requests and requests confined to the central leaf")


def radix_slots():
    fig, ax = plt.subplots(figsize=(6, 7))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    width, tile = 2, 0.5
    def project(x, y, u, v):
        return x + 0.8 * (u - v), y + 0.4 * (u + v - width)

    def arrow(start, end, x, y):
        # Find where the descending path emerges from beneath the source plane.
        dx, dy = end[0] - start[0], end[1] - start[1]
        exits = []
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = sx / (0.8 * width), sy / (0.4 * width)
            rate = a * dx + b * dy
            if rate > 0:
                exits.append((1 - a * (start[0] - x) - b * (start[1] - y + 0.1)) / rate)
        t = min(exits)
        assert 0 < t < 1
        edge = start[0] + t * dx, start[1] + t * dy
        ax.annotate("", edge, start, zorder=4,
                    arrowprops={"arrowstyle": "-", "linestyle": ":", "color": BLUE,
                                "lw": 2, "shrinkA": 6, "shrinkB": 0})
        ax.annotate("", end, edge, zorder=4,
                    arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2,
                                "shrinkA": 0, "shrinkB": 6})

    rows = []
    x = 0
    for y, selected, label in ((7, (2, 1), "Branch · child slots"),
                                (4, (1, 2), "Branch · child slots"),
                                (1, (2, 1), "Leaf · entry slots")):
        rows.append((x, y, selected, label))
        x, _ = project(x, y, (selected[0] + 0.5) * tile, (selected[1] + 0.5) * tile)
    ax.text(rows[0][0], 8.3, "⋮", color=GRID, fontsize=23, ha="center", va="center")
    for x, y, (col, row), label in rows:
        ax.text(x - 1.9, y, label.replace(" · ", "\n"),
                color=GRID, ha="right", va="center")
        # A shallow front rim gives the tilted array a little depth.
        front = project(x, y, 0, 0)
        for u, v in ((width, 0), (0, width)):
            side = project(x, y, u, v)
            ax.add_patch(Polygon((front, side, (side[0], side[1] - 0.1),
                                  (front[0], front[1] - 0.1)),
                                 facecolor=GRID, edgecolor="none", alpha=0.18))
        for i in range(4):
            for j in range(4):
                corners = [project(x, y, (i + du) * tile, (j + dv) * tile)
                           for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1))]
                if (i, j) == (col, row):
                    ax.add_patch(Polygon(corners, facecolor=BLUE, alpha=0.2, edgecolor="none"))
                ax.add_patch(Polygon(corners, fill=False, edgecolor=GRID, lw=0.8))
        ax.plot(*project(x, y, (col + 0.5) * tile, (row + 0.5) * tile),
                "o", color=BLUE, ms=5, zorder=5)

    for (x, y, (col, row), _), (nx, ny, (ncol, nrow), _) in zip(rows, rows[1:]):
        # Expansion guides connect the selected slot to its child's whole array.
        center = project(x, y, (col + 0.5) * tile, (row + 0.5) * tile)
        assert abs(center[0] - nx) < 1e-12
        for u, v, nu, nv in ((col * tile, (row + 1) * tile, 0, width),
                              ((col + 1) * tile, row * tile, width, 0)):
            start, end = project(x, y, u, v), project(nx, ny, nu, nv)
            ax.plot((start[0], end[0]), (start[1], end[1]), color=GRID, lw=0.8,
                    linestyle=(0, (3, 4)), alpha=0.65, zorder=0)
        target = project(nx, ny, (ncol + 0.5) * tile, (nrow + 0.5) * tile)
        arrow(center, target, x, y)

    x, y, (col, row), _ = rows[-1]
    target = project(x, y, (col + 0.5) * tile, (row + 0.5) * tile)
    arrow(target, (target[0], -0.35), x, y)
    ax.text(target[0], -0.6, "Cached entry → value", ha="center", va="center", color=GRID)
    ax.set(xlim=(-3.8, 2.2), ylim=(-0.85, 8.65), aspect="equal")
    ax.set_axis_off()
    save(fig, "radix-slots.svg", "A radix lookup follows selected branch slots to a leaf, with further levels above")


if __name__ == "__main__":
    sampling()
    radix_slots()
