import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, ListedColormap, BoundaryNorm
from pathlib import Path

#Analysis of run_adaptation.py outputs: gap Delta(delta, n, k) = R_ICL(delta, k) - R_FT(delta, n)
#(Delta > 0: FT wins, Delta < 0: ICL wins), the crossover data size n*(delta), and the three-way winner
#among ICL, FT and FT+ICL. FT lr / step are chosen on the validation tasks; everything shown is on test tasks.
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, default="../runs/fig2a/")
parser.add_argument("--phis", type=int, nargs="+", default=[30, 150])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
parser.add_argument("--noise_var", type=float, default=0.)
parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
parser.add_argument("--select", choices=["per_delta", "global"], default="per_delta",
                    help="choose FT (lr, step) per (n, delta), or one recipe per n shared across delta")
parser.add_argument("--out_dir", type=str, default="../runs/adaptation/")
args = parser.parse_args()

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ICL_C, FT_C, FTICL_C = "#2a78d6", "#eb6834", "#1baf7a"
PHI_STYLE = {30: dict(color=INK, ls="-", marker="o"), 150: dict(color=MUTED, ls=(0, (4, 2)), marker="s")}
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
tag = "noise{0}_{1}".format(args.noise_var, args.select)

def select_ft(D, R):
    """(lr, step) index per (n, task), chosen to minimize the mean risk R (N, L, C, E) of the validation tasks at the
    same delta. Step 0 (no fine-tuning) is a candidate."""
    R = np.nan_to_num(R, nan=np.inf)
    deltas, split = D['member_delta'], D['member_split']
    choice = np.zeros((R.shape[0], R.shape[-1], 2), dtype=int)
    for i in range(R.shape[0]):
        per_delta = {d: R[i][:, :, (split == "val") & (deltas == d)].mean(-1) for d in np.unique(deltas)}
        if args.select == "global":
            score = np.mean([np.log(np.maximum(s, 1e-30)) for s in per_delta.values()], axis=0)
        for d, s in per_delta.items():
            l, c = np.unravel_index(np.argmin(score if args.select == "global" else s), s.shape)
            choice[i, deltas == d] = (l, c)
    return choice

def load(phi):
    """Test-task risks pooled over seeds: R_icl (E, K); R_ft (N, E); R_ft_icl_ftckpt (N, E, K), the FT-only checkpoint
    evaluated with demonstrations; R_ft_icl (N, E, len(ks)), FT+ICL with its own (lr, step) selected for each k;
    plus delta / seed per task and the selected FT-only lr / step. When run_baselines.py outputs exist: R_ols (N, E)
    and ridge toward 0 / toward the pole, R_ridge / R_ridge_pole (N, E), with lambda selected like the FT recipe."""
    out = {k: [] for k in ("R_icl", "R_ft", "R_ft_icl_ftckpt", "R_ft_icl", "delta", "seed", "lr", "step",
                           "R_ols", "R_ridge", "R_ridge_pole")}
    for s in args.seeds:
        f = Path(args.run_dir) / "seed{0}_noise0.0_tasks11_angle{1}".format(s, phi) / "adaptation_noise{0}.npz".format(args.noise_var)
        D = dict(np.load(f))
        test = D['member_split'] == "test"
        n_idx = np.arange(len(D['ns']))[:, None]
        e_idx = np.arange(len(test))[None, :]
        choice = select_ft(D, D['R_ft'])
        l, c = choice[..., 0], choice[..., 1]
        out["R_icl"].append(D['R_icl'][test])
        out["R_ft"].append(D['R_ft'][n_idx, l, c, e_idx][:, test])
        out["R_ft_icl_ftckpt"].append(D['R_ft_icl'][n_idx, l, c, e_idx][:, test])
        out["lr"].append(D['lrs'][l][:, test])
        out["step"].append(D['ckpts'][c][:, test])
        own = []
        for k in args.ks:
            ck = select_ft(D, D['R_ft_icl'][..., k])
            own.append(D['R_ft_icl'][n_idx, ck[..., 0], ck[..., 1], e_idx, k][:, test])
        out["R_ft_icl"].append(np.stack(own, -1))
        out["delta"].append(D['member_delta'][test])
        out["seed"].append(np.full(test.sum(), s))
        fb = f.with_name("baselines_noise{0}.npz".format(args.noise_var))
        if fb.exists():
            B = np.load(fb)
            out["R_ols"].append(B['R_ridge'][:, list(B['lambdas']).index(0)][:, test])
            for key in ("R_ridge", "R_ridge_pole"):
                ch = select_ft(D, B[key][:, :, None, :])
                out[key].append(B[key][n_idx, ch[..., 0], e_idx][:, test])
        ns, dim = D['ns'], D['W'].shape[1]
    assert len(out["R_ols"]) in (0, len(args.seeds)), "baselines exist for only some seeds"
    task_axis = {"R_icl": 0, "R_ft": 1, "R_ft_icl_ftckpt": 1, "R_ft_icl": 1, "lr": 1, "step": 1, "delta": 0, "seed": 0,
                 "R_ols": 1, "R_ridge": 1, "R_ridge_pole": 1}
    res = {k: np.concatenate(v, axis=task_axis[k]) for k, v in out.items() if v}
    res["ns"], res["dim"] = ns, dim
    return res

def gap_stats(res, k, seed=None):
    """Mean and standard error over test tasks of Delta = R_ICL(k) - R_FT(n), on the (n, delta) grid."""
    deltas = np.unique(res["delta"])
    mean = np.zeros((len(res["ns"]), len(deltas)))
    se = np.zeros_like(mean)
    for j, d in enumerate(deltas):
        m = (res["delta"] == d) & (res["seed"] == seed if seed is not None else True)
        g = res["R_icl"][m, k][None, :] - res["R_ft"][:, m]
        mean[:, j] = g.mean(1)
        se[:, j] = g.std(1, ddof=1) / np.sqrt(m.sum())
    return deltas, mean, se

def crossover(ns, gap):
    """log2 of the smallest n at which FT starts to win (first sign change of Delta, interpolated in log2 n).
    Returns (value, status) with status 'cross', 'below' (FT wins already at the smallest n) or 'above' (never)."""
    x = np.log2(ns)
    if gap[0] > 0:
        return x[0], "below"
    for i in range(len(ns) - 1):
        if gap[i] <= 0 < gap[i + 1]:
            return x[i] + (0 - gap[i]) / (gap[i + 1] - gap[i]) * (x[i + 1] - x[i]), "cross"
    return x[-1], "above"

data = {phi: load(phi) for phi in args.phis}
ns, dim = data[args.phis[0]]["ns"], data[args.phis[0]]["dim"]
deltas = np.unique(data[args.phis[0]]["delta"])
k_max = data[args.phis[0]]["R_icl"].shape[1] - 1
has_baselines = all("R_ols" in data[phi] for phi in args.phis)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

def style(ax, log2n_y=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel(r"target angle $\delta$ from pretraining pole (degrees)")
    if log2n_y:
        ax.set_yticks(np.log2(ns)[::2])
        ax.set_yticklabels([str(n) for n in ns[::2]])
        ax.set_ylabel("fine-tuning examples $n$")

def cell_edges(c):
    step = np.diff(c).min()
    return np.concatenate([c - step / 2, [c[-1] + step / 2]])

# ---------- Figure 1: Delta heatmaps + crossover n*(delta) ----------
gaps = {(phi, k): gap_stats(data[phi], k) for phi in args.phis for k in args.ks}
vmax = max(np.abs(g[1]).max() for g in gaps.values())
div = LinearSegmentedColormap.from_list("icl_ft", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f6b89c", "#eb6834", "#8a3412"])
norm = SymLogNorm(linthresh=1e-3, linscale=0.6, vmin=-vmax, vmax=vmax, base=10)
xe, ye = cell_edges(deltas.astype(float)), cell_edges(np.log2(ns).astype(float))

#heatmaps in the first len(phis) columns, crossover panel last; a thin bottom row holds the shared colorbar
height = 3.3 * len(args.ks) + 0.5
fig = plt.figure(figsize=(4.0 * (len(args.phis) + 1) + 0.6, height), facecolor=SURFACE)
gs = fig.add_gridspec(len(args.ks) + 1, len(args.phis) + 1, height_ratios=[1] * len(args.ks) + [0.05],
                      hspace=0.55, wspace=0.32, top=1 - 0.55 / height, bottom=0.04)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(len(args.phis) + 1)] for r in range(len(args.ks))])
summary = {}
for r, k in enumerate(args.ks):
    for cidx, phi in enumerate(args.phis):
        ax = axes[r, cidx]
        _, mean, se = gaps[(phi, k)]
        mesh = ax.pcolormesh(xe, ye, mean, cmap=div, norm=norm, shading="flat", edgecolors=SURFACE, linewidth=0.6)
        X, Y = np.meshgrid(deltas, np.log2(ns))
        if mean.min() < 0 < mean.max():
            ax.contour(X, Y, mean, levels=[0], colors=INK, linewidths=1.4)
        weak = np.abs(mean) < 2 * se
        ax.scatter(X[weak], Y[weak], s=4, color=MUTED, linewidths=0)
        if phi < 180:
            ax.axvline(phi, color=INK, lw=0.9, ls=(0, (3, 2)))
            right = phi < 90
            ax.text(phi + (2 if right else -2), ye[-1] - 0.15, r"cap edge $\phi$", fontsize=7.5, color=INK, va="top",
                    ha="left" if right else "right")
        style(ax)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.set_title(r"$\Delta$ = R$_{{ICL}}$(k={0}) $-$ R$_{{FT}}$(n),  pretrained $\phi$={1}°".format(k, phi), fontsize=9, loc="left")
    ax = axes[r, -1]
    style(ax)
    for phi in args.phis:
        _, mean, _ = gaps[(phi, k)]
        pts = [crossover(ns, mean[:, j]) for j in range(len(deltas))]
        summary[(phi, k)] = pts
        y = np.array([p[0] for p in pts])
        st = PHI_STYLE.get(phi, dict(color=INK, ls="-", marker="o"))
        ax.plot(deltas, y, color=st["color"], ls=st["ls"], lw=1.5, label=r"$\phi$={0}°".format(phi))
        for status, mk in (("cross", st["marker"]), ("above", "^"), ("below", "v")):
            m = np.array([p[1] == status for p in pts])
            ax.scatter(deltas[m], y[m], marker=mk, s=22, color=st["color"], zorder=3, edgecolors=SURFACE, linewidths=0.8)
        for s in args.seeds:
            _, ms, _ = gap_stats(data[phi], k, seed=s)
            ax.plot(deltas, [crossover(ns, ms[:, j])[0] for j in range(len(deltas))], color=st["color"], ls=st["ls"], lw=0.6, alpha=0.45)
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xlim(-5, 185)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_ylabel(r"crossover $n^*$ (FT wins for $n > n^*$)")
    ax.set_title("crossover, k={0}  (▲: ICL wins at all n, ▼: FT wins at all n)".format(k), fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
cb = fig.colorbar(mesh, cax=fig.add_subplot(gs[-1, :len(args.phis)]), orientation="horizontal")
cb_ticks = [t for t in (1e-3, 1e-2, 1e-1, 1) if t <= vmax]
cb.set_ticks([-t for t in cb_ticks[::-1]] + [0] + cb_ticks)
cb.set_ticklabels(["{0:g}".format(-t) for t in cb_ticks[::-1]] + ["0"] + ["{0:g}".format(t) for t in cb_ticks])
cb.minorticks_off()
cb.set_label(r"$\Delta$   (< 0: ICL wins   |   > 0: FT wins)")
cb.outline.set_visible(False)
fig.suptitle(r"ICL vs fine-tuning,  label noise $\sigma^2$={0:g}.  Black line: $\Delta$=0.  Gray dots: |mean $\Delta$| < 2 SE.  "
             "FT (lr, steps) selected {1}.".format(args.noise_var, "per (n, δ) on validation tasks" if args.select == "per_delta"
                                                   else "per n on validation tasks"), fontsize=9.5, x=0.02, ha="left", y=0.995)
fig.savefig(out_dir / "gap_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 2: risk curves vs delta ----------
k_shades = ["#86b6ef", "#2a78d6", "#104281"]
n_show = [n for n in (4, 32, 256, 1024) if n in ns]
n_shades = ["#f6b89c", "#f08a5b", "#eb6834", "#8a3412"][:len(n_show)]
fig, axes = plt.subplots(1, len(args.phis), figsize=(5.0 * len(args.phis), 3.8), facecolor=SURFACE, sharey=True)
for ax, phi in zip(np.atleast_1d(axes), args.phis):
    res = data[phi]
    style(ax, log2n_y=False)
    ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6)
    for k, c in zip(args.ks, k_shades):
        ax.plot(deltas, [res["R_icl"][res["delta"] == d, k].mean() for d in deltas], color=c, lw=1.6, marker="o", ms=3,
                label="ICL, k={0}".format(k))
    for n, c in zip(n_show, n_shades):
        i = list(ns).index(n)
        ax.plot(deltas, [res["R_ft"][i, res["delta"] == d].mean() for d in deltas], color=c, lw=1.6, marker="s", ms=3,
                ls=(0, (4, 2)), label="FT, n={0}".format(n))
    ax.plot(deltas, [res["R_icl"][res["delta"] == d, 0].mean() for d in deltas],
            color=MUTED, lw=1, ls=":", label="no adaptation (k=0, n=0)")
    if phi < 180:
        ax.axvline(phi, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.set_title(r"pretrained $\phi$={0}°".format(phi), fontsize=9.5, loc="left")
np.atleast_1d(axes)[0].set_ylabel("excess test risk")
np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.suptitle(r"Risk vs target angle, $\sigma^2$={0:g}".format(args.noise_var), fontsize=9.5, x=0.02, ha="left")
fig.savefig(out_dir / "risks_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 3: three-way winner (ICL / FT / FT+ICL) ----------
height = 3.3 * len(args.ks) + 0.6
fig, axes = plt.subplots(len(args.ks), len(args.phis), figsize=(4.0 * len(args.phis) + 1.2, height), facecolor=SURFACE,
                         squeeze=False, gridspec_kw={"hspace": 0.45, "wspace": 0.25, "top": 1 - 0.75 / height})
win_cmap = ListedColormap([ICL_C, FT_C, FTICL_C])
win_summary = {}
for r, k in enumerate(args.ks):
    for cidx, phi in enumerate(args.phis):
        res = data[phi]
        ax = axes[r, cidx]
        win = np.zeros((len(ns), len(deltas)))
        weak = np.zeros_like(win, dtype=bool)
        for j, d in enumerate(deltas):
            m = res["delta"] == d
            R3 = np.stack([np.broadcast_to(res["R_icl"][m, k], (len(ns), m.sum())), res["R_ft"][:, m], res["R_ft_icl"][:, m, r]])
            mean = R3.mean(-1)
            win[:, j] = mean.argmin(0)
            #is the winner significantly better than the runner-up? (paired over tasks)
            order = np.argsort(mean, axis=0)
            for i in range(len(ns)):
                diff = R3[order[1, i], i] - R3[order[0, i], i]
                weak[i, j] = diff.mean() < 2 * diff.std(ddof=1) / np.sqrt(m.sum())
        win_summary[(phi, k)] = win
        ax.pcolormesh(xe, ye, win, cmap=win_cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), edgecolors=SURFACE, linewidth=0.6)
        X, Y = np.meshgrid(deltas, np.log2(ns))
        ax.scatter(X[weak], Y[weak], s=5, color=SURFACE, linewidths=0)
        if phi < 180:
            ax.axvline(phi, color=INK, lw=0.9, ls=(0, (3, 2)))
        style(ax)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.set_title(r"best method, k={0}, pretrained $\phi$={1}°".format(k, phi), fontsize=9, loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ICL_C, FT_C, FTICL_C)]
fig.legend(handles, ["ICL only", "FT only", "FT + ICL"], frameon=False, loc="upper left", ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.015, 1 - 0.3 / height))
fig.suptitle(r"Lowest-risk adaptation method, $\sigma^2$={0:g}.  White dots: winner not > 2 SE better than runner-up (paired).  "
             "Each method's (lr, steps) selected on validation tasks.".format(args.noise_var),
             fontsize=9.5, x=0.02, ha="left", y=1 - 0.05 / height)
fig.savefig(out_dir / "winner_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 4: in-context learning curves before and after zero-context fine-tuning ----------
d_show = [d for d in (0, 30, 90, 180) if d in deltas]
n_ft = [n for n in (16, 256) if n in ns]
fig, axes = plt.subplots(len(args.phis), len(d_show), figsize=(3.0 * len(d_show) + 1.6, 2.7 * len(args.phis)), facecolor=SURFACE,
                         squeeze=False, sharex=True, sharey="row", gridspec_kw={"hspace": 0.4, "wspace": 0.12})
k_axis = np.arange(data[args.phis[0]]["R_icl"].shape[1])
for r, phi in enumerate(args.phis):
    res = data[phi]
    for cidx, d in enumerate(d_show):
        ax = axes[r, cidx]
        m = res["delta"] == d
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, lw=0.6)
        ax.plot(k_axis, res["R_icl"][m].mean(0), color=ICL_C, lw=1.6, label="pretrained (ICL only)")
        for n, c in zip(n_ft, ["#f08a5b", "#8a3412"]):
            i = list(ns).index(n)
            ax.plot(k_axis, res["R_ft_icl_ftckpt"][i, m].mean(0), color=c, lw=1.6, ls=(0, (4, 2)),
                    label="after FT on n={0} (FT-only checkpoint)".format(n))
        ax.set_title(r"$\phi$={0}°, $\delta$={1:.0f}°".format(phi, d), fontsize=9, loc="left")
        if r == len(args.phis) - 1:
            ax.set_xlabel("in-context demonstrations k")
    axes[r, 0].set_ylabel("excess test risk")
axes[0, -1].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(r"Zero-context fine-tuning vs in-context use, $\sigma^2$={0:g}: risk of the fine-tuned model as demonstrations are added "
             "(k=0 is FT only)".format(args.noise_var), fontsize=9.5, x=0.02, ha="left")
fig.savefig(out_dir / "icl_after_ft_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 5: equal data budget m. ICL with k = m demonstrations, FT on n = m examples, and linear
# regression fit on the same m FT examples; bands are +- 2 SE over test tasks ----------
def mean_se(R):
    return R.mean(-1), R.std(-1, ddof=1) / np.sqrt(R.shape[-1])

def band(ax, x, R, floor, **kw):
    mu, se = mean_se(R)
    ax.plot(x, np.maximum(mu, floor), **kw)
    ax.fill_between(x, np.maximum(mu - 2 * se, floor), np.maximum(mu + 2 * se, floor), color=kw["color"], alpha=0.15, linewidth=0)

m_icl = np.arange(1, k_max + 1)
lowest = min(min(data[phi]["R_ft"].mean(1).min(), data[phi]["R_icl"].mean(0).min()) for phi in args.phis)
floor = 10 ** (np.floor(np.log10(lowest)) - 1)
top = 3 * max(data[phi]["R_icl"][:, 0].max() for phi in args.phis)
fig, axes = plt.subplots(len(args.phis), len(d_show), figsize=(3.0 * len(d_show) + 2.2, 2.9 * len(args.phis)), facecolor=SURFACE,
                         squeeze=False, sharex=True, sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.1})
for r, phi in enumerate(args.phis):
    res = data[phi]
    for cidx, d in enumerate(d_show):
        ax = axes[r, cidx]
        m = res["delta"] == d
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, lw=0.6)
        ax.axvspan(k_max, ns[-1] * 1.4, color=GRID, alpha=0.55, lw=0)
        ax.axvline(dim, color=MUTED, lw=0.8, ls=":")
        ax.axhline(res["R_icl"][m, 0].mean(), color=MUTED, lw=0.8, ls=(0, (1, 3)), label="no adaptation")
        band(ax, m_icl, res["R_icl"][m][:, 1:].T, floor, color=ICL_C, lw=1.6, label="ICL only, k = m demonstrations")
        band(ax, ns, res["R_ft"][:, m], floor, color=FT_C, lw=1.6, marker="s", ms=3, label="FT only, n = m examples")
        if has_baselines:
            ax.plot(ns, np.maximum(res["R_ols"][:, m].mean(1), floor), color=INK, lw=1.2,
                    label="OLS on the same m (min-norm for m < d)" + (", = 0 for m ≥ d" if args.noise_var == 0 else ""))
            ax.plot(ns, np.maximum(res["R_ridge"][:, m].mean(1), floor), color=MUTED, lw=1.2, ls=(0, (4, 2)), label="ridge toward 0")
            ax.plot(ns, np.maximum(res["R_ridge_pole"][:, m].mean(1), floor), color=INK, lw=1.2, ls=":", marker="o", ms=2.5,
                    label="ridge toward pole v")
        ax.set_ylim(floor, top)
        ax.set_xlim(0.8, ns[-1] * 1.4)
        ax.set_xticks([1, 4, 16, 64, 256, 1024])
        ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
        ax.set_title(r"$\phi$={0}°, $\delta$={1:.0f}°".format(phi, d), fontsize=9, loc="left")
        if r == 0 and cidx == 0:
            ax.text(dim * 1.08, top / 1.6, "m = d", fontsize=7.5, color=MUTED, va="top")
            ax.text(k_max * 1.12, top / 1.6, "past ICL\ncontext\nwindow", fontsize=7.5, color=MUTED, va="top")
        if r == len(args.phis) - 1:
            ax.set_xlabel("examples m")
    axes[r, 0].set_ylabel("excess test risk")
axes[0, -1].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(r"Equal data budget, $\sigma^2$={0:g}: every method gets the same m labeled examples (bands: $\pm$2 SE over test tasks; "
             r"FT recipe and ridge $\lambda$ selected {1} on validation tasks)".format(
                 args.noise_var, "per (m, δ)" if args.select == "per_delta" else "per m"), fontsize=9.5, x=0.02, ha="left")
fig.savefig(out_dir / "budget_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 6: equal-budget gap Delta_eq(delta, m) = R_ICL(k=m) - R_FT(n=m), for m within the context window ----------
eq_idx = [i for i, n in enumerate(ns) if n <= k_max]
ns_eq = ns[eq_idx]

def gap_eq_stats(res):
    mean = np.zeros((len(eq_idx), len(deltas)))
    se = np.zeros_like(mean)
    for j, d in enumerate(deltas):
        m = res["delta"] == d
        g = res["R_icl"][m][:, ns_eq].T - res["R_ft"][eq_idx][:, m]
        mean[:, j], se[:, j] = mean_se(g)
    return mean, se

gaps_eq = {phi: gap_eq_stats(data[phi]) for phi in args.phis}
vmax_eq = max(np.abs(g[0]).max() for g in gaps_eq.values())
norm_eq = SymLogNorm(linthresh=1e-3, linscale=0.6, vmin=-vmax_eq, vmax=vmax_eq, base=10)
ye_eq = cell_edges(np.log2(ns_eq).astype(float))
fig, axes = plt.subplots(1, len(args.phis), figsize=(4.0 * len(args.phis) + 1.2, 3.0), facecolor=SURFACE, squeeze=False,
                         gridspec_kw={"wspace": 0.25})
for cidx, phi in enumerate(args.phis):
    ax = axes[0, cidx]
    mean, se = gaps_eq[phi]
    mesh = ax.pcolormesh(xe, ye_eq, mean, cmap=div, norm=norm_eq, shading="flat", edgecolors=SURFACE, linewidth=0.6)
    X, Y = np.meshgrid(deltas, np.log2(ns_eq))
    if mean.min() < 0 < mean.max():
        ax.contour(X, Y, mean, levels=[0], colors=INK, linewidths=1.4)
    weak = np.abs(mean) < 2 * se
    ax.scatter(X[weak], Y[weak], s=4, color=MUTED, linewidths=0)
    if phi < 180:
        ax.axvline(phi, color=INK, lw=0.9, ls=(0, (3, 2)))
    style(ax, log2n_y=False)
    ax.set_yticks(np.log2(ns_eq))
    ax.set_yticklabels([str(n) for n in ns_eq])
    ax.set_ylabel("examples m  (k = n = m)")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye_eq[0], ye_eq[-1])
    ax.set_title(r"$\Delta_{{eq}}$ = R$_{{ICL}}$(k=m) $-$ R$_{{FT}}$(n=m),  pretrained $\phi$={0}°".format(phi), fontsize=9, loc="left")
cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
cb_ticks = [t for t in (1e-3, 1e-2, 1e-1, 1) if t <= vmax_eq]
cb.set_ticks([-t for t in cb_ticks[::-1]] + [0] + cb_ticks)
cb.set_ticklabels(["{0:g}".format(-t) for t in cb_ticks[::-1]] + ["0"] + ["{0:g}".format(t) for t in cb_ticks])
cb.minorticks_off()
cb.set_label(r"$\Delta_{eq}$  (< 0: ICL wins | > 0: FT wins)")
cb.outline.set_visible(False)
fig.suptitle(r"Equal data budget, $\sigma^2$={0:g}.  Black line: $\Delta_{{eq}}$=0.  Gray dots: |mean| < 2 SE.  "
             "m is capped by the {1}-demonstration context window.".format(args.noise_var, k_max), fontsize=9.5, x=0.02, ha="left", y=1.04)
fig.savefig(out_dir / "gap_equal_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 7: can the boundary n*(delta) be predicted without fine-tuning this model at this delta? ----------
#alignment only: the boundary is a function of delta alone, so use the other pretrained model's n*(delta).
#frozen-ICL behavior: intersect this model's measured ICL risk R_ICL(delta, k) (no fine-tuning needed) with a
#delta-averaged FT learning curve, either this model's own or, fully transferred, the other model's.
def ft_curve(res):
    """FT risk vs n with its delta-dependence averaged out: geometric mean over delta of the per-delta mean risk."""
    return np.exp(np.mean([np.log(res["R_ft"][:, res["delta"] == d].mean(1)) for d in deltas], axis=0))

predictions = {}
if len(args.phis) == 2:
    for phi in args.phis:
        other = args.phis[1 - args.phis.index(phi)]
        for k in args.ks:
            icl = [data[phi]["R_icl"][data[phi]["delta"] == d, k].mean() for d in deltas]
            predictions[(phi, k)] = {
                "observed": np.array([v for v, _ in summary[(phi, k)]]),
                "alignment only (other model's n* at the same δ)": np.array([v for v, _ in summary[(other, k)]]),
                "frozen ICL risk + own δ-averaged FT curve": np.array([crossover(ns, r - ft_curve(data[phi]))[0] for r in icl]),
                "frozen ICL risk + other model's FT curve": np.array([crossover(ns, r - ft_curve(data[other]))[0] for r in icl])}
    pred_style = [dict(color=MUTED, ls=(0, (4, 2)), lw=1.4), dict(color=INK, ls="-", lw=1.4), dict(color=INK, ls=":", lw=1.6)]
    fig, axes = plt.subplots(len(args.phis), len(args.ks), figsize=(3.6 * len(args.ks) + 0.4, 3.0 * len(args.phis)), facecolor=SURFACE,
                             squeeze=False, sharex=True, sharey=True, gridspec_kw={"hspace": 0.5, "wspace": 0.16})
    for r, phi in enumerate(args.phis):
        for cidx, k in enumerate(args.ks):
            ax = axes[r, cidx]
            style(ax)
            ax.grid(True, color=GRID, lw=0.6)
            P = predictions[(phi, k)]
            errs = []
            for (name, y), st in zip(list(P.items())[1:], pred_style):
                errs.append(np.abs(y - P["observed"]).mean())
                ax.plot(deltas, y, label=name, **st)
            ax.scatter(deltas, P["observed"], s=26, color=INK, zorder=3, edgecolors=SURFACE, linewidths=0.8, label="observed n*")
            if phi < 180:
                ax.axvline(phi, color=INK, lw=0.9, ls=(0, (3, 2)))
            ax.set_xlim(-5, 185)
            ax.set_ylim(ye[0], ye[-1])
            ax.set_title(r"$\phi$={0}°, k={1}" "\n" r"|log$_2$ err|: align {2:.1f} · ICL+own {3:.1f} · ICL+other {4:.1f}".format(
                phi, k, *errs), fontsize=8, loc="left")
            if cidx:
                ax.set_ylabel("")
            if r < len(args.phis) - 1:
                ax.set_xlabel("")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(r"Predicting the crossover $n^*(\delta)$, $\sigma^2$={0:g}: from task alignment alone vs from the frozen model's measured "
                 "ICL risk".format(args.noise_var), fontsize=9.5, x=0.02, ha="left", y=1.02)
    fig.savefig(out_dir / "predict_{0}.png".format(tag), dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- text summary ----------
lines = ["noise_var={0}  select={1}  seeds={2}  test tasks per (phi, delta)={3}".format(
    args.noise_var, args.select, args.seeds, int((data[args.phis[0]]["delta"] == deltas[0]).sum()))]
for phi in args.phis:
    res = data[phi]
    lines.append("\nphi={0}: mean excess risk by delta".format(phi))
    lines.append("  delta " + " ".join("{0:>8}".format("ICLk{0}".format(k)) for k in args.ks) + " " +
                 " ".join("{0:>8}".format("FTn{0}".format(n)) for n in n_show))
    for d in deltas:
        m = res["delta"] == d
        lines.append("  {0:5.0f} ".format(d) + " ".join("{0:8.1e}".format(res["R_icl"][m, k].mean()) for k in args.ks) + " " +
                     " ".join("{0:8.1e}".format(res["R_ft"][list(ns).index(n), m].mean()) for n in n_show))
    for k in args.ks:
        pts = summary[(phi, k)]
        lines.append("  crossover n*(delta), k={0}: ".format(k) + "  ".join(
            "{0:.0f}:{1}".format(d, ">{0}".format(ns[-1]) if s == "above" else ("<={0}".format(ns[0]) if s == "below" else "{0:.0f}".format(2 ** v)))
            for d, (v, s) in zip(deltas, pts)))
        w = win_summary[(phi, k)]
        lines.append("  FT+ICL best in {0}/{1} cells, ICL in {2}, FT in {3}".format(int((w == 2).sum()), w.size, int((w == 0).sum()), int((w == 1).sum())))
    for n in n_show:
        i = list(ns).index(n)
        lines.append("  n={0}: delta, then per k: FT+ICL (own ckpt) / FT+ICL (FT-only ckpt)".format(n))
        for d in deltas:
            m = res["delta"] == d
            lines.append("  {0:5.0f} ".format(d) + "  ".join("k{0}: {1:8.1e} / {2:8.1e}".format(
                k, res["R_ft_icl"][i, m, r].mean(), res["R_ft_icl_ftckpt"][i, m, k].mean()) for r, k in enumerate(args.ks)))
    lines.append("  selected FT lr (rows n, median over tasks): " + ", ".join(
        "n={0}:{1:.0e}@{2:.0f}".format(n, np.median(res["lr"][i]), np.median(res["step"][i])) for i, n in enumerate(ns)))
def examples_to_reach(x, risk, eps):
    """Smallest m (interpolated in log-log) at which the mean risk first drops to eps; inf if it never does."""
    below = np.nonzero(risk <= eps)[0]
    if len(below) == 0:
        return np.inf
    i = below[0]
    if i == 0:
        return float(x[0])
    lx, lr = np.log(x[i - 1:i + 1].astype(float)), np.log(np.maximum(risk[i - 1:i + 1], 1e-300))
    return float(np.exp(lx[0] + (np.log(eps) - lr[0]) / (lr[1] - lr[0]) * (lx[1] - lx[0])))

lines.append("\nequal data budget: examples m needed to reach mean excess risk eps (ICL limited to m <= {0}; inf = never)".format(k_max))
methods = [("ICL", lambda res, m: (m_icl, res["R_icl"][m][:, 1:].mean(0))), ("FT", lambda res, m: (ns, res["R_ft"][:, m].mean(1)))]
if has_baselines:
    methods += [(name, lambda res, m, key=key: (ns, res[key][:, m].mean(1)))
                for name, key in (("OLS", "R_ols"), ("ridge0", "R_ridge"), ("ridge_v", "R_ridge_pole"))]
for eps in (1e-1, 1e-2, 1e-3):
    lines.append("  eps={0:g}".format(eps))
    for phi in args.phis:
        res = data[phi]
        lines.append("    phi={0:3d} delta: ".format(phi) + " ".join("{0:>6.0f}".format(d) for d in deltas))
        for name, curve in methods:
            lines.append("    {0:>16}  ".format(name) + " ".join("{0:>6.0f}".format(examples_to_reach(*curve(res, res["delta"] == d), eps))
                                                            for d in deltas))
for phi in args.phis:
    mean, se = gaps_eq[phi]
    lines.append("\nphi={0}: equal-budget gap Delta_eq = R_ICL(k=m) - R_FT(n=m), mean (2SE) by delta".format(phi))
    for i, n in enumerate(ns_eq):
        lines.append("  m={0:3d} ".format(n) + " ".join("{0:9.1e}({1:.0e})".format(a, 2 * b) for a, b in zip(mean[i], se[i])))
if predictions:
    lines.append("\npredicting n*(delta): mean |log2 error| over delta (values clipped to [{0}, {1}])".format(ns[0], ns[-1]))
    for (phi, k), P in predictions.items():
        lines.append("  phi={0:3d} k={1:2d}  ".format(phi, k) + "  ".join(
            "{0}: {1:.2f}".format(name, np.abs(y - P["observed"]).mean()) for name, y in list(P.items())[1:]))
text = "\n".join(lines)
print(text)
(out_dir / "summary_{0}.txt".format(tag)).write_text(text + "\n")
print("saved figures to", out_dir)
