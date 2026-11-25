import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})
INK, GRID = "#161616", "#8a8a8a"
FILL = {"build": "#E8E8E8", "cold": "#CFE3F3", "warm": "#FBE3C4", "cfg": "#F5F5F5"}
EDGE = {"build": "#555555", "cold": "#0072B2", "warm": "#D55E00", "cfg": "#555555"}

W, H = 3.45, 2.28                 # one IEEE column is about 3.5 in wide
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
bw, gap, x0, bh = 0.57, 0.12, 0.08, 0.44
xs = [x0 + i * (bw + gap) for i in range(5)]

def box(x, y, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.04", fc=FILL[kind], ec=EDGE[kind], lw=0.9))
    ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=6.9, color=INK, linespacing=1.05)

def arrows(y):
    for i in range(4):
        ax.add_patch(FancyArrowPatch((xs[i] + bw + 0.005, y + bh / 2), (xs[i + 1] - 0.005, y + bh / 2),
                                     arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK, shrinkA=0, shrinkB=0))

def bracket(xa, xb, y, label):
    ax.plot([xa, xb], [y, y], lw=0.7, color=GRID)
    for xx in (xa, xb):
        ax.plot([xx, xx], [y, y + 0.05], lw=0.7, color=GRID)
    ax.text((xa + xb) / 2, y - 0.11, label, ha="center", va="center", fontsize=6.4, style="italic", color=INK)

# lane (a)
ya = 1.40
ax.text(x0, 2.14, "(a) Stand-alone runs", fontsize=7.4, fontweight="bold", va="center", color=INK)
for x, t, k in zip(xs, ["Build\nindex", "Empty\ncache", "Open\nindex", "Cold\npass", "Warm\npass"], ["build", "build", "cfg", "cold", "warm"]):
    box(x, ya, t, k)
arrows(ya)
bracket(xs[1], xs[4] + bw, ya - 0.06, "for each budget $B$")

# lane (b)
yb = 0.22
ax.text(x0, 0.97, "(b) VIBE runs", fontsize=7.4, fontweight="bold", va="center", color=INK)
for x, t, k in zip(xs, ["Build\nindex", "Set\nsetting", "5\nruns", "Fastest\nrun", "Recall"], ["build", "cfg", "warm", "build", "build"]):
    box(x, yb + 0.12, t, k)
arrows(yb + 0.12)
bracket(xs[1], xs[4] + bw, yb + 0.06, "for each query setting")

fig.savefig("fig_protocol.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_protocol_small.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
