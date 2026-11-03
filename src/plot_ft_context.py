import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

#Plots run_ft_context.py: risk vs number of in-context demonstrations k after zero-context and variable-context
#fine-tuning, on the fine-tuned task and on a different task at the same delta, plus risk vs fine-tuning step.
#Checkpoints (lr, step) are selected on the validation tasks by zero-context risk (the FT-only criterion of
#icl_after_ft_*.png) unless --select all_k (mean log risk over all k); everything shown is on test tasks.
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../runs/adaptation/ft_context_phi150_delta90.npz")
parser.add_argument("--select", choices=["k0", "all_k"], default="k0")
parser.add_argument("--traj_lr", type=float, default=1e-3, help="learning rate of the risk-vs-step panels")
parser.add_argument("--traj_k", type=int, default=20)
args = parser.parse_args()

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"pretrained": "#2a78d6", "zero": "#eb6834", "variable": "#1baf7a"}
LABEL = {"pretrained": "pretrained (ICL only)", "zero": "after zero-context FT", "variable": "after variable-context FT"}
LS = {"pretrained": "-", "zero": (0, (4, 2)), "variable": (0, (6, 2, 1, 2))}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

D = np.load(args.data)
variants, ns, lrs, ckpts = list(D['variants']), D['ns'], D['lrs'], D['ckpts']
val, test = D['member_split'] == "val", D['member_split'] == "test"
R_own, R_other = D['R_own'], D['R_other']
K = R_own.shape[-1]
k_axis = np.arange(K)
tag = "{0}_{1}".format(Path(args.data).stem, args.select)

def select(v, i):
    """(lr, step) indices minimizing the validation criterion for variant v and set size index i."""
    R = R_own[v, i][:, :, val]
    score = R[..., 0].mean(-1) if args.select == "k0" else np.log(R).mean((-1, -2))
    return np.unravel_index(np.argmin(score), score.shape)

def mean_se(R):
    return R.mean(0), R.std(0, ddof=1) / np.sqrt(R.shape[0])

def band(ax, x, R, key):
    mu, se = mean_se(R)
    ax.plot(x, mu, color=COLOR[key], ls=LS[key], lw=1.6, label=LABEL[key])
    ax.fill_between(x, np.maximum(mu - 2 * se, mu * 1e-3), mu + 2 * se, color=COLOR[key], alpha=0.15, linewidth=0)

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6)

choice = {(v, i): select(v, i) for v in range(len(variants)) for i in range(len(ns))}
fig, axes = plt.subplots(3, len(ns), figsize=(4.6 * len(ns) + 2.2, 9.0), facecolor=SURFACE, squeeze=False,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.18})
for i, n in enumerate(ns):
    for r, (R, what) in enumerate(((R_own, "the fine-tuned task"), (R_other, "a different task at the same δ"))):
        ax = axes[r, i]
        style(ax)
        band(ax, k_axis, R[0, i, 0, 0][test], "pretrained")
        notes = []
        for v, variant in enumerate(variants):
            l, c = choice[(v, i)]
            band(ax, k_axis, R[v, i, l, c][test], variant)
            notes.append("{0}: lr {1:.0e}, {2} steps".format(variant, lrs[l], ckpts[c]))
        if variants[-1] == "variable" and n < K:
            ax.axvspan(n - 0.5, K, color=GRID, alpha=0.5, lw=0)
            ax.text(n + 0.5, ax.get_ylim()[1] if r else 1.0, "beyond FT\nprompt length", fontsize=7, color=MUTED, va="top")
        ax.set_xlabel("in-context demonstrations k")
        ax.set_ylabel("excess test risk")
        ax.set_title("n={0}: demonstrations of {1}\n".format(n, what) + "; ".join(notes), fontsize=8.5, loc="left")
    ax = axes[2, i]
    style(ax)
    j = int(np.argmin(np.abs(np.log(lrs / args.traj_lr))))
    x = np.maximum(ckpts, 1)
    for v, variant in enumerate(variants):
        for R, k, ls, name in ((R_own, 0, "-", "own task, k=0"), (R_own, args.traj_k, (0, (4, 2)), "own task, k={0}".format(args.traj_k)),
                               (R_other, args.traj_k, ":", "other task, k={0}".format(args.traj_k))):
            mu, _ = mean_se(R[v, i, j][:, test, k].T)
            ax.plot(x, mu, color=COLOR[variant], ls=ls, lw=1.5, marker="o", ms=2.5, label="{0} FT: {1}".format(variant, name))
    ax.set_xscale("log")
    ax.set_xlabel("fine-tuning steps (step 0 drawn at 1)")
    ax.set_ylabel("excess test risk")
    ax.set_title("n={0}: risk during fine-tuning at lr {1:.0e}".format(n, lrs[j]), fontsize=8.5, loc="left")
axes[0, -1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
axes[2, -1].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
crit = "zero-context risk (FT-only criterion)" if args.select == "k0" else "mean log risk over all k"
fig.suptitle(r"Zero-context vs variable-context fine-tuning, pretrained $\phi$=150°, target $\delta$={0:.0f}°, $\sigma^2$=0.  "
             "Checkpoints selected on validation tasks by {1}; bands ±2 SE over {2} test tasks.".format(float(D['delta']), crit, test.sum()),
             fontsize=9.5, x=0.02, ha="left", y=0.995)
out_png = Path(args.data).with_name("{0}.png".format(tag))
fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")

ks = [k for k in (0, 1, 2, 5, 10, 20, 49) if k < K]
lines = ["{0}  select={1}  test tasks={2}".format(Path(args.data).name, args.select, test.sum()),
         "mean excess risk at k demonstrations: " + " ".join("{0:>9}".format("k={0}".format(k)) for k in ks)]
for i, n in enumerate(ns):
    for R, what in ((R_own, "own  "), (R_other, "other")):
        lines.append("n={0:4d} {1} pretrained          ".format(n, what) + " ".join("{0:9.2e}".format(R[0, i, 0, 0][test, k].mean()) for k in ks))
        for v, variant in enumerate(variants):
            l, c = choice[(v, i)]
            lines.append("n={0:4d} {1} {2:8s} lr{3:.0e}@{4:<4d}".format(n, what, variant, lrs[l], ckpts[c]) +
                         " ".join("{0:9.2e}".format(R[v, i, l, c][test, k].mean()) for k in ks))
text = "\n".join(lines)
print(text)
out_png.with_suffix(".txt").write_text(text + "\n")
print("saved", out_png)
