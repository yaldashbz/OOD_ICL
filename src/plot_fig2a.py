import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path

#Reproduces Fig 2A (test MSE in 5-degree bands vs training cap half-angle phi) and Fig 13 (NSR vs phi)
#from the losses_angle*_tasks*.npy files written by train_linreg.train (test_cone_falloff, last position only).
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, default="../runs/fig2a/")
parser.add_argument("--log2_tasks", type=int, default=11)
parser.add_argument("--noise_std", type=float, default=0.)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
parser.add_argument("--nsr_threshold", type=float, default=0.5)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

PHIS = np.arange(15, 181, 15)
BAND_STARTS = np.arange(0, 180, 5)          #test_cone_falloff bands [delta, delta+5)
PLOT_DELTAS = np.arange(10, 176, 15)        #bands shown in the paper's Fig 2A legend
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEED_COLORS = ["#2a78d6", "#eb6834", "#1f9e89", "#7c4dbd"]
run_dir = Path(args.run_dir)

def load_losses(seed):
    losses = np.full((len(PHIS), len(BAND_STARTS)), np.nan)
    for i, phi in enumerate(PHIS):
        f = run_dir / "seed{0}_noise{1}_tasks{2}_angle{3}".format(seed, args.noise_std, args.log2_tasks, phi) \
            / "losses_angle{0}_tasks{1}.npy".format(phi, args.log2_tasks)
        if f.exists():
            losses[i] = np.load(f)
    return losses

losses = {s: load_losses(s) for s in args.seeds}
nsr = {s: np.nanstd(l, axis=1) / np.nanmean(l, axis=1) for s, l in losses.items()}

#Table: in-distribution (delta in [0,5)), OOD (delta in [175,180)), NSR
for s in args.seeds:
    print("seed {0}".format(s))
    print("  phi | MSE delta=0  | MSE delta=175 | NSR    | phase")
    for i, phi in enumerate(PHIS):
        if np.isnan(losses[s][i]).all():
            print("  {0:3d} | (missing)".format(phi))
            continue
        phase = "general" if nsr[s][i] < args.nsr_threshold else "specialized"
        print("  {0:3d} | {1:.3e}    | {2:.3e}     | {3:.3f}  | {4}".format(
            phi, losses[s][i, 0], losses[s][i, -1], nsr[s][i], phase))

cmap = LinearSegmentedColormap.from_list("blue_ordinal", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
norm = BoundaryNorm(np.append(PLOT_DELTAS, 190) - 7.5, cmap.N)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
n = len(args.seeds)
fig = plt.figure(figsize=(4.6 * (n + 1) + 1.2, 4.2), facecolor=SURFACE)
gs = fig.add_gridspec(1, n + 3, width_ratios=[1] * n + [0.035, 0.12, 0.9], wspace=0.28)
axes = [fig.add_subplot(gs[0, 0])]
axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in range(1, n)] + [fig.add_subplot(gs[0, n + 2])]
cax = fig.add_subplot(gs[0, n])

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([15, 45, 90, 135, 180])
    ax.set_xlim(8, 187)
    ax.set_xlabel(r"training cap half-angle $\phi$ (degrees)")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

for ax, s in zip(axes[:-1], args.seeds):
    style(ax)
    ax.axvline(120, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(121.5, 0.97, "paper: ~120°", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="top")
    for d in PLOT_DELTAS:
        ax.plot(PHIS, losses[s][:, d // 5], color=cmap(norm(d)), lw=1.5, marker="o", ms=3.5)
    for d in (PLOT_DELTAS[0], PLOT_DELTAS[-1]):
        ax.annotate(r"$\delta$={0}°".format(d), (PHIS[0], losses[s][0, d // 5]), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color=MUTED)
    if ax is axes[0]:
        ax.set_ylabel("test MSE (final position)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(r"Fig 2A, seed {0}  (N=$2^{{{1}}}$ tasks, $\sigma^2$={2:g})".format(s, args.log2_tasks, args.noise_std**2),
                 fontsize=10, loc="left")
ymin = min(np.nanmin(l) for l in losses.values())
ymax = max(np.nanmax(l) for l in losses.values())
for ax in axes[:-1]:
    ax.set_ylim(ymin / 1.6, ymax * 1.6)
    ax.set_xlim(-12, 187)

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=PLOT_DELTAS)
cbar.set_label(r"test band start $\delta$ (degrees, width 5°)")
cbar.outline.set_visible(False)
cbar.ax.minorticks_off()

ax = axes[-1]
style(ax)
for k, s in enumerate(args.seeds):
    ax.plot(PHIS, nsr[s], color=SEED_COLORS[k], lw=1.5, marker="o", ms=3.5, label="seed {0}".format(s))
ax.axhline(args.nsr_threshold, color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax.text(12, args.nsr_threshold * 1.12, "threshold {0:g}".format(args.nsr_threshold), color=MUTED, fontsize=8)
nsr_all = np.concatenate([v[~np.isnan(v)] for v in nsr.values()] + [[args.nsr_threshold]])
ax.set_ylim(nsr_all.min() / 2, nsr_all.max() * 2)
ax.set_ylabel(r"NSR = std$_\delta$ / mean$_\delta$ of test MSE")
ax.set_title("Fig 13: transition point", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left")

out = args.out or str(run_dir / "fig2a_tasks{0}_noise{1}.png".format(args.log2_tasks, args.noise_std))
fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
print("saved", out)
