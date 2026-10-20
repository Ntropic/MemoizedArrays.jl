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
    fig, ax = canvas(6.6, 4.2)
    label(ax, 1.15, 3.95, "Map slots", size=11)
    for row in range(2):
        for col in range(3):
            name = (("D", "E", ""), ("B", "C", "A"))[row][col]
            tile(ax, 0.55 + col * 0.4, 3.05 + (1-row) * 0.4,
                 0.4, 0.4, name, BLUE if name == "A" else GRID, name == "A")
    tile(ax, 3.05, 2.95, 3.05, 0.95, color=BLUE, selected=True)
    label(ax, 4.575, 3.69, "Entry A: value = …", BLUE)
    label(ax, 4.575, 3.37, "last_access: Q₂, version v", CENTRAL)
    arrow(ax, (1.75, 3.25), (3.05, 3.25))
    for x, name, values in ((0.5, "Q₁", ("A", "B", "C")),
                            (4.05, "Q₂", ("D", "E", "A"))):
        label(ax, x + 0.8, 2.48, name + " references", size=11)
        for i, value in enumerate(values):
            tile(ax, x + i * 0.55, 1.9, 0.4, 0.4, value,
                 BLUE if value == "A" else GRID, value == "A")
        label(ax, x + 0.8, 1.68, "oldest → newest", size=10)
        a_x = x + values.index("A") * 0.55 + 0.2
        arrow(ax, (a_x, 2.3), (3.2 if name == "Q₁" else 5.9, 2.95), dashed=True)
    label(ax, 3.3, 1.18, "Access timeline · left to right", size=11)
    arrow(ax, (0.4, 0.78), (6.2, 0.78), GRID)
    for x, queue, entry in ((0.8, "Q₁", "A"), (1.8, "Q₁", "B"),
                            (2.8, "Q₁", "C"), (3.8, "Q₂", "D"),
                            (4.8, "Q₂", "E"), (5.8, "Q₂", "A")):
        ax.plot(x, 0.78, "o", color=BLUE if entry == "A" else GRID, markersize=4)
        label(ax, x, 0.48, queue + ": " + entry, size=10)
    label(ax, 3.3, 0.14, "Final access to A stamps Q₂; Q₁'s ordering stays unchanged.", size=10)
    save(fig, "lru-queues.svg", "Square map slots and small queue references share entry A; the timeline explains each queue's order and A's last-access hint")


def lru_eviction():
    fig, ax = canvas(6.6, 4.8)
    def box(y, text, color=BLUE):
        tile(ax, 0.1, y, 3.8, 0.55, text, color, True)
    box(4.1, "Q₁ selects its oldest reference")
    box(3.25, "Same live entry version?")
    box(2.4, "Current-version hint\nnames another queue?")
    box(1.55, "Exclusive claim succeeds?")
    box(0.55, "Remove map entry\nReuse or release it", CENTRAL)
    for top, bottom, text in ((4.1, 3.8, ""), (3.25, 2.95, "Yes"),
                              (2.4, 2.1, "No"), (1.55, 1.1, "Yes")):
        arrow(ax, (2, top), (2, bottom))
        label(ax, 2.35, (top + bottom) / 2, text, size=10)
    tile(ax, 4.65, 2.25, 1.85, 0.9, "Drop only Q₁'s\nreference\nKeep map as is", CENTRAL, True)
    for y, answer in ((3.525, "No"), (2.675, "Yes"), (1.825, "No")):
        ax.plot([3.9, 4.4, 4.4], [y, y, 2.7], color=CENTRAL, lw=1.3)
        label(ax, 4.15, y + 0.15, answer, CENTRAL, size=10)
    arrow(ax, (4.4, 2.7), (4.65, 2.7), CENTRAL)
    label(ax, 3.3, 0.2, "Reusing an entry changes its version; old references become stale.", size=10)
    save(fig, "lru-eviction.svg", "Compact eviction flow checks version, last-access hint and exclusive claim")


if __name__ == "__main__":
    expanding_map()
    lru_queues()
    lru_eviction()
