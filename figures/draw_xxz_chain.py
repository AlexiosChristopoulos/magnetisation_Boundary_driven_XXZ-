"""Draw the boundary-driven XXZ model shown in the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("xxz_chain_baths.png")
NAVY = "#17324d"
BLUE = "#277da8"
GOLD = "#da8b25"
LEFT_BLUE = "#2766a0"
RED = "#b84e52"

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})


def wiggly_line(x_start, x_end, baseline, amplitude=0.055, turns=4, **style):
    xs = np.linspace(x_start, x_end, 160)
    ys = baseline + amplitude * np.sin(np.linspace(0, 2 * np.pi * turns, 160))
    ax.plot(xs, ys, **style)

fig, ax = plt.subplots(figsize=(13.2, 5.55), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 5.55)
ax.axis("off")

ax.text(6.6, 5.13, "Boundary-driven XXZ chain", ha="center", va="center",
        color=NAVY, fontsize=20, weight="bold")
ax.text(6.6, 4.38,
        r"$H=\sum_{i=1}^{N-1}\left[J(X_iX_{i+1}+Y_iY_{i+1})+\Delta Z_iZ_{i+1}\right]$",
        ha="center", va="center", color=NAVY, fontsize=16)

site_x = [2.85, 4.55, 6.25, 7.95, 9.65]
y = 2.59
for xa, xb in zip(site_x[:-1], site_x[1:]):
    ax.plot([xa + 0.27, xb - 0.27], [y + 0.055, y + 0.055],
            color=BLUE, lw=5, solid_capstyle="round", zorder=1)
    wiggly_line(xa + 0.27, xb - 0.27, y - 0.105,
                color=GOLD, lw=3.2, zorder=1)

for n, x in enumerate(site_x, 1):
    ax.add_patch(Circle((x, y), 0.285, facecolor="white", edgecolor=NAVY,
                        linewidth=2.5, zorder=3))
    ax.text(x, y, str(n), ha="center", va="center",
            color=NAVY, fontsize=14, weight="bold", zorder=4)

for x, title, color in [(0.28, "LEFT BATH", LEFT_BLUE), (10.91, "RIGHT BATH", RED)]:
    ax.add_patch(FancyBboxPatch((x, 2.13), 1.98, 0.92,
                                boxstyle="round,pad=0.08,rounding_size=0.16",
                                facecolor=color, edgecolor=color, zorder=2))
    ax.text(x + 0.99, y, title, color="white", ha="center", va="center",
            fontsize=12, weight="bold", zorder=3)


def flow(start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                 linewidth=2.2, color=color,
                                 connectionstyle="arc3,rad=0", zorder=5))


# At the left boundary, pumping is stronger; at the right, loss is stronger.
flow((2.31, 2.91), (2.59, 2.79), LEFT_BLUE)
flow((2.59, 2.38), (2.31, 2.25), LEFT_BLUE)
flow((10.84, 2.91), (9.91, 2.79), RED)
flow((9.91, 2.38), (10.84, 2.25), RED)

ax.text(1.50, 3.49, r"pump $\gamma(1+\mu)$", color=LEFT_BLUE,
        ha="center", fontsize=12, weight="bold")
ax.text(1.50, 1.60, r"loss $\gamma(1-\mu)$", color=LEFT_BLUE,
        ha="center", fontsize=12, weight="bold")
ax.text(11.75, 3.49, r"pump $\gamma(1-\mu)$", color=RED,
        ha="center", fontsize=12, weight="bold")
ax.text(11.75, 1.60, r"loss $\gamma(1+\mu)$", color=RED,
        ha="center", fontsize=12, weight="bold")

ax.plot([3.17, 3.59], [1.25, 1.25], color=BLUE, lw=5, solid_capstyle="round")
ax.text(3.71, 1.25, r"exchange $J(XX+YY)$", va="center", color=NAVY, fontsize=11.5)
wiggly_line(7.0, 7.42, 1.25, amplitude=0.045, turns=2,
            color=GOLD, lw=3.2)
ax.text(7.54, 1.25, r"Ising coupling $\Delta ZZ$", va="center", color=NAVY, fontsize=11.5)
ax.annotate("positive spin current", xy=(7.10, 0.54), xytext=(5.40, 0.54),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.8),
            ha="center", va="center", color=NAVY, fontsize=10.5)

fig.savefig(OUT, dpi=180, bbox_inches="tight", pad_inches=0.20)
plt.close(fig)
print(OUT)
