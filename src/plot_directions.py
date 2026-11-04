import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

#Analysis of run_adaptation.py --directions outputs: two targets with the same angle delta from the pretraining pole,
#A = cos(delta) v + sin(delta) e_2 and B = cos(delta) v + sin(delta) e_3, under an isotropic cap and an anisotropic cap
#that emphasizes e_2 and de-emphasizes e_3. Delta_j(n) = R_ICL,j(k = min(n, k_max)) - R_FT,j(n): < 0 ICL wins, > 0 FT
#wins. Directions that are exchangeable within a model (all 9 under the isotropic cap, e_4..e_10 under the anisotropic
#one) give the null spread of direction differences. FT (lr, step) is selected per n on validation datasets, one
#recipe shared by all directions (mean log risk over directions); everything shown is on paired test datasets.
parser = argparse.ArgumentParser()
parser.add_argument("--iso", type=str, nargs="+",
                    default=["../runs/fig2a/seed{0}_noise0.0_tasks11_angle60".format(s) for s in (0, 1, 2)])
parser.add_argument("--aniso", type=str, nargs="+",
                    default=["../runs/aniso/seed{0}_noise0.0_tasks11_angle60_aniso".format(s) for s in (0, 1, 2)])
parser.add_argument("--file", type=str, default="adaptation_dirs_delta60_noise0.0.npz")
parser.add_argument("--dir_a", type=int, default=1, help="coordinate of target A (1 = e_2)")
parser.add_argument("--dir_b", type=int, default=2, help="coordinate of target B (2 = e_3)")
parser.add_argument("--k_max", type=int, default=16, help="ICL uses k = min(n, k_max) demonstrations")
parser.add_argument("--out_dir", type=str, default="../runs/aniso/")
args = parser.parse_args()

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
A_C, B_C, NULL_C = "#6250d6", "#eb6834", "#9a998f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

def load(run_dir):
    """Test risks of one pretrained model, indexed [direction j, rep r]: icl (J, R, K), ft (N, J, R), ft_icl (N, J, R)
    at k_max, with the global FT recipe per n."""
    D = np.load(Path(run_dir) / args.file)
    dirs, split, rep = D['member_direction'], D['member_split'], D['member_rep']
    J = np.unique(dirs)
    val = lambda j: (split == "val") & (dirs == j)
    test = [np.nonzero((split == "test") & (dirs == j))[0][np.argsort(rep[(split == "test") & (dirs == j)])] for j in J]
    out = dict(J=J, ns=D['ns'], phi=int(D['phi']), choice=[], k=[min(n, args.k_max) for n in D['ns']])
    out["icl"] = np.stack([D['R_icl'][t] for t in test])
    ft, ft_icl = [], []
    for i, n in enumerate(D['ns']):
        score = np.mean([np.log(D['R_ft'][i][:, :, val(j)].mean(-1)) for j in J], axis=0)
        l, c = np.unravel_index(np.argmin(score), score.shape)
        out["choice"].append((D['lrs'][l], D['ckpts'][c]))
        ft.append(np.stack([D['R_ft'][i, l, c, t] for t in test]))
        k = out["k"][i]
        score = np.mean([np.log(D['R_ft_icl'][i][:, :, val(j), k].mean(-1)) for j in J], axis=0)
        l, c = np.unravel_index(np.argmin(score), score.shape)
        ft_icl.append(np.stack([D['R_ft_icl'][i, l, c, t, k] for t in test]))
    out["ft"], out["ft_icl"] = np.stack(ft), np.stack(ft_icl)
    out["gap"] = np.stack([out["icl"][:, :, k] for k in out["k"]]) - out["ft"]
    return out

models = {}
for kind in ("iso", "aniso"):
    found = [p for p in getattr(args, kind) if (Path(p) / args.file).exists()]
    missing = sorted(set(getattr(args, kind)) - set(found))
    if missing:
        print("{0}: missing {1}".format(kind, missing))
    if found:
        models[kind] = [load(p) for p in found]
assert models, "no results found"
first = next(iter(models.values()))[0]
J, ns = list(first["J"]), first["ns"]
ia, ib = J.index(args.dir_a), J.index(args.dir_b)
null_idx = {"iso": [i for i in range(len(J)) if i not in (ia, ib)], "aniso": [i for i in range(len(J)) if i not in (ia, ib)]}
TITLE = {"iso": "isotropic cap (symmetry control)", "aniso": "anisotropic cap"}
NULL_LABEL = {"iso": "other 7 directions (exchangeable)", "aniso": "e$_4$..e$_{10}$ (scale 0.5, exchangeable)"}
name = lambda j: "e$_{{{0}}}$".format(j + 1)

def pooled(kind, key):
    """Concatenate over seeds along the rep axis: icl is (J, R, K), the others (N, J, R)."""
    return np.concatenate([m[key] for m in models[kind]], axis=1 if key == "icl" else -1)

def mean_se(x, axis=-1):
    return x.mean(axis), x.std(axis, ddof=1) / np.sqrt(x.shape[axis])

def style(ax, logy=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6)

def plot_dirs(ax, x, R, kind, marker=None):
    """R: (len(x), J, reps). A and B with +-2 SE bands, null directions as the range of their means."""
    for idx, color, label in ((ia, A_C, "A = " + name(args.dir_a)), (ib, B_C, "B = " + name(args.dir_b))):
        mu, se = mean_se(R[:, idx])
        ax.plot(x, mu, color=color, lw=1.7, marker=marker, ms=3.5, label=label)
        lo = mu - 2 * se if ax.get_yscale() != "log" else np.maximum(mu - 2 * se, mu * 1e-2)
        ax.fill_between(x, lo, mu + 2 * se, color=color, alpha=0.15, linewidth=0)
    nm = R[:, null_idx[kind]].mean(-1)
    ax.fill_between(x, nm.min(1), nm.max(1), color=NULL_C, alpha=0.35, linewidth=0, label=NULL_LABEL[kind] + ": range")

# ---------- Figure 1: ICL vs k, FT vs n, and the gap, for A, B and the null directions ----------
kinds = list(models)
fig, axes = plt.subplots(len(kinds), 3, figsize=(13.5, 3.4 * len(kinds)), facecolor=SURFACE, squeeze=False,
                         gridspec_kw={"hspace": 0.5, "wspace": 0.28})
k_axis = np.arange(first["icl"].shape[-1])
for r, kind in enumerate(kinds):
    seeds = len(models[kind])
    ax = axes[r, 0]
    style(ax)
    plot_dirs(ax, k_axis, pooled(kind, "icl").transpose(2, 0, 1), kind)
    ax.axvline(args.k_max, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("in-context demonstrations k")
    ax.set_ylabel("excess test risk")
    ax.set_title("{0}: ICL only ({1} seeds)".format(TITLE[kind], seeds), fontsize=9, loc="left")
    ax = axes[r, 1]
    style(ax)
    plot_dirs(ax, ns, pooled(kind, "ft"), kind, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("fine-tuning examples n")
    ax.set_title("FT only (one recipe per n for all directions)", fontsize=9, loc="left")
    ax = axes[r, 2]
    style(ax, logy=False)
    plot_dirs(ax, ns, pooled(kind, "gap"), kind, marker="o")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("fine-tuning examples n  (ICL: k = min(n, {0}))".format(args.k_max))
    ax.set_title(r"$\Delta$ = R$_{ICL}$ $-$ R$_{FT}$  (< 0: ICL wins, > 0: FT wins)", fontsize=9, loc="left")
axes[0, -1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(r"Two targets at the same angle $\delta$=60° from the pole of a $\phi$=60° cap, pointing along different perpendicular "
             "directions.  Bands: ±2 SE over paired test datasets pooled over seeds; gray: range over exchangeable directions.",
             fontsize=9.5, x=0.02, ha="left", y=1.0)
fig.savefig(out_dir / "directions_delta60.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- Figure 2: Delta_A and Delta_B per seed and n, against the null spread ----------
fig, axes = plt.subplots(len(kinds), len(ns), figsize=(2.9 * len(ns) + 1.5, 2.9 * len(kinds)), facecolor=SURFACE, squeeze=False,
                         sharey="row", gridspec_kw={"hspace": 0.55, "wspace": 0.1})
for r, kind in enumerate(kinds):
    for i, n in enumerate(ns):
        ax = axes[r, i]
        style(ax, logy=False)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.axhline(0, color=INK, lw=0.9)
        for s, m in enumerate(models[kind]):
            nm = m["gap"][i][null_idx[kind]].mean(-1)
            ax.plot([s, s], [nm.min(), nm.max()], color=NULL_C, lw=7, alpha=0.5, solid_capstyle="butt",
                    label="null range" if s == 0 else None)
            for idx, color, dx, label in ((ia, A_C, -0.12, "A"), (ib, B_C, 0.12, "B")):
                mu, se = mean_se(m["gap"][i][idx])
                ax.errorbar(s + dx, mu, yerr=2 * se, color=color, marker="o", ms=4, lw=1.2, capsize=0,
                            label=label if s == 0 else None)
        ax.set_xticks(range(len(models[kind])))
        ax.set_xticklabels(["seed {0}".format(s) for s in range(len(models[kind]))])
        ax.set_xlim(-0.5, len(models[kind]) - 0.5)
        ax.set_title("{0}\nn={1}, k={2}".format(TITLE[kind] if i == 0 else "", n, min(n, args.k_max)), fontsize=8.5, loc="left")
    axes[r, 0].set_ylabel(r"$\Delta$ = R$_{ICL}$ $-$ R$_{FT}$")
axes[0, -1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle(r"$\Delta_A$ and $\Delta_B$ per pretrained seed (±2 SE over paired test datasets); gray bar: range of $\Delta$ over the "
             "exchangeable directions of the same model", fontsize=9.5, x=0.02, ha="left", y=1.02)
fig.savefig(out_dir / "directions_delta60_seeds.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")

# ---------- text summary ----------
lines = ["targets at delta=60: A = {0}, B = {1}; ICL k = min(n, {2}); FT recipe global over directions".format(
    name(args.dir_a).replace("$", ""), name(args.dir_b).replace("$", ""), args.k_max)]
for kind in kinds:
    lines.append("\n{0}: {1} seeds, {2} paired test datasets per seed".format(kind, len(models[kind]), models[kind][0]["icl"].shape[1]))
    for i, n in enumerate(ns):
        k = min(n, args.k_max)
        lines.append("  n={0:4d} k={1:2d}".format(n, k))
        for s, m in enumerate(models[kind] + [None]):
            src = (lambda key: m[key]) if m is not None else (lambda key: pooled(kind, key))
            icl, ft, ft_icl, gap = src("icl")[:, :, k], src("ft")[i], src("ft_icl")[i], src("gap")[i]
            d_ab = gap[ia] - gap[ib]
            nm = gap[null_idx[kind]].mean(-1)
            fmt = lambda x: "{0:9.2e}({1:.0e})".format(*[f(x) for f in (np.mean, lambda y: 2 * y.std(ddof=1) / np.sqrt(len(y)))])
            lines.append("    {0:7s} ICL A/B {1} {2}  FT A/B {3} {4}  FT+ICL A/B {5:9.2e} {6:9.2e}  Delta A {7} B {8}  "
                         "A-B {9}  null Delta range [{10:.2e}, {11:.2e}]{12}".format(
                             "seed{0}".format(s) if m is not None else "pooled", fmt(icl[ia]), fmt(icl[ib]), fmt(ft[ia]), fmt(ft[ib]),
                             ft_icl[ia].mean(), ft_icl[ib].mean(), fmt(gap[ia]), fmt(gap[ib]), fmt(d_ab), nm.min(), nm.max(),
                             "" if m is None else "  FT lr {0:.0e} @ {1} steps".format(*m["choice"][i])))
text = "\n".join(lines)
print(text)
(out_dir / "directions_delta60.txt").write_text(text + "\n")
print("saved figures to", out_dir)
