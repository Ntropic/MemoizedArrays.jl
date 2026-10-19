"""Generate transparent SVGs for expanding maps and shared LRU memberships."""
from matplotlib.patches import Rectangle
from plot_tree_sampling import plt, save, GRID, BLUE, CENTRAL


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, text, color=GRID, size=11, **options):
    ax.text(x, y, text, color=color, fontsize=size, ha="center", va="center", **options)


def arrow(ax, start, end, color=BLUE, dashed=False):
    ax.annotate("", end, start, arrowprops=dict(
        arrowstyle="->", color=color, lw=1.5, linestyle="--" if dashed else "-",
        shrinkA=4, shrinkB=4))


def tile(ax, x, y, width, height, text="", color=GRID, selected=False):
    if selected:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, alpha=0.12, edgecolor="none"))
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, lw=1.2))
    if text:
        label(ax, x + width / 2, y + height / 2, text, color)


def expanding_map():
    fig, ax = canvas(10, 3.4)
    fig.set_size_inches(7, 2.38)
    existing = {(0, 0), (1, 2), (2, 1)}
    tile_size = 0.5
    for stage, (x, columns, title) in enumerate((
            (0.25, 3, "1 · Request"),
            (3.55, 5, "2 · Allocate + copy"),
            (7.05, 5, "3 · Generate"))):
        label(ax, x + columns * tile_size / 2, 3.1, title, size=9)
        for column in range(columns):
            for row in range(3):
                tile(ax, x + column * tile_size, 0.8 + row * tile_size,
                     tile_size, tile_size, color=CENTRAL if column >= 3 else GRID,
                     selected=column >= 3)
        for column, row in existing:
            ax.plot(x + (column + 0.5) * tile_size, 0.8 + (row + 0.5) * tile_size,
                    "o", color=BLUE, markersize=4)
        target = (x + 4.5 * tile_size, 0.8 + 1.5 * tile_size)
        if stage == 0:
            ax.plot(*target, "x", color=CENTRAL, markersize=7, markeredgewidth=1.5)
            label(ax, target[0], 2.65, "cache[5, 2]", CENTRAL, size=9)
            arrow(ax, (target[0], 2.45), target, CENTRAL)
        elif stage == 2:
            ax.plot(*target, "o", color=CENTRAL, markersize=4)
        label(ax, x + columns * tile_size / 2, 0.5,
              "size: (3, 3)" if stage == 0 else "size: (5, 3)", size=9)
    arrow(ax, (2.85, 1.6), (3.4, 1.6))
    arrow(ax, (6.2, 1.6), (6.9, 1.6))
    label(ax, 5, 0.12, "increment = 2   ·   blue: cached   ·   green: new   ·   first coordinate →", size=8)
    save(fig, "expanding-array-map.svg", "Expansion preserves coordinates and existing values; only the requested new value is generated")


def lru_queues():
    fig, ax = canvas(9, 7.3)
    label(ax, 4.5, 6.95, "One map entry · multiple queue references", size=13)
    for row in range(2):
        for column in range(5):
            selected = (column, row) == (2, 0)
            tile(ax, 2.75 + column * 0.7, 5.45 + row * 0.5, 0.7, 0.5,
                 "A" if selected else "", BLUE if selected else GRID, selected)
    label(ax, 4.5, 6.65, "A: last-access hint = Q2", BLUE, size=10)
    for x, name, values in ((0.8, "Q1", ("A", "B", "C")),
                            (5.3, "Q2", ("D", "E", "A"))):
        label(ax, x + 1.4, 4.4, name, size=12)
        for i, value in enumerate(values):
            tile(ax, x + i * 0.95, 3.4, 0.9, 0.55, value,
                 BLUE if value == "A" else GRID, value == "A")
        label(ax, x + 1.4, 3.05, "oldest → newest", size=10)
        a_x = x + values.index("A") * 0.95 + 0.45
        target_x = 4.3 if name == "Q1" else 4.7
        ax.plot([a_x, a_x, target_x], [4.0, 4.85, 4.85], "--", color=BLUE, lw=1.5)
        arrow(ax, (target_x, 4.85), (target_x, 5.45), dashed=True)

    for x, title, detail, result, color in (
            (0.35, "Q1 reaches A", "Last access belongs to Q2", "Drop Q1 reference\nKeep A in the map", CENTRAL),
            (4.85, "Later, Q2 reaches A", "Still last accessed by Q2", "Remove A from the map\nReuse its entry", BLUE)):
        arrow(ax, (x + 1.9, 2.8), (x + 1.9, 2.45), color)
        tile(ax, x, 0.65, 3.8, 1.75, color=color, selected=True)
        label(ax, x + 1.9, 2.1, title, color, size=12)
        label(ax, x + 1.9, 1.7, detail, size=10)
        label(ax, x + 1.9, 1.15, result, color, size=11)
    label(ax, 4.5, 0.2, "Removal also requires the same live entry version and a successful exclusive claim.", size=10)
    save(fig, "lru-queues.svg", "Queues reference shared entries; last-access hints distinguish dropping a membership from removing a map entry")


if __name__ == "__main__":
    expanding_map()
    lru_queues()
