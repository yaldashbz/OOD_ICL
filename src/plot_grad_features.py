import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

#Do checkpoint gradient features (run_grad_features.py) predict the direction dependence of the ICL-FT boundary beyond
#the frozen model's ICL risk? Cells are (pretrained model, target direction j, FT set size n) of run_adaptation.py
#--directions runs, with ICL at k = min(n, k_max) and the FT recipe shared by all directions (as in plot_directions.py).
#Every predictor gets the model's direction-averaged log FT risk curve and direction-averaged log ICL risk, and adds
#direction-specific information:
#  none                     no direction information (alignment delta alone, which is the same for every j)
#  ICL only                 the frozen model's ICL risk of direction j
#  gradients only           log FT_j - mean_j log FT predicted from the centered gradient features of direction j
#  ICL + gradients          both
#  oracle                   the measured direction-specific FT risk
#Feature slopes are fit by least squares on centered (within model and n) cells of the other models, per n
#(leave-one-model-out). Errors: log gap log(R_ICL / R_FT), crossover n* (log-log interpolation), and the n=64 reversal.
parser = argparse.ArgumentParser()
parser.add_argument("--iso", type=str, nargs="+",
                    default=["../runs/fig2a/seed{0}_noise0.0_tasks11_angle60".format(s) for s in (0, 1, 2)])
parser.add_argument("--aniso", type=str, nargs="+",
                    default=["../runs/aniso/seed{0}_noise0.0_tasks11_angle60_aniso".format(s) for s in (0, 1, 2)])
parser.add_argument("--file", type=str, default="adaptation_dirs_delta60_noise0.0.npz")
parser.add_argument("--dir_a", type=int, default=1, help="coordinate of target A (1 = e_2)")
parser.add_argument("--dir_b", type=int, default=2, help="coordinate of target B (2 = e_3)")
parser.add_argument("--k_max", type=int, default=16, help="ICL uses k = min(n, k_max) demonstrations")
parser.add_argument("--n_reversal", type=int, default=64)
parser.add_argument("--out_dir", type=str, default="../runs/aniso/")
args = parser.parse_args()

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
A_C, B_C, NULL_C = "#6250d6", "#eb6834", "#9a998f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
out_dir = Path(args.out_dir)
FEATURES = {"norm": r"log $|g_{tr}|$", "align": r"cos($g_{tr}$, $g_{val}$)", "adam": r"cos($u_{tr}$, $g_{val}$)",
            "loss0": "log initial train loss"}

def load(run_dir):
    """Test datasets of one pretrained model, each array (N, J, R): ICL risk at k_max, FT risk with the global recipe,
    and the gradient features."""
    D = np.load(Path(run_dir) / args.file)
    G = np.load(Path(run_dir) / args.file.replace("adaptation_", "gradfeat_"))
    dirs, split, rep = D['member_direction'], D['member_split'], D['member_rep']
    for k in ('member_split', 'member_direction', 'member_rep'):
        assert np.array_equal(D[k], G[k])
    J, ns = np.unique(dirs), D['ns']
    val = lambda j: (split == "val") & (dirs == j)
    test = [np.nonzero((split == "test") & (dirs == j))[0][np.argsort(rep[(split == "test") & (dirs == j)])] for j in J]
    by_test = lambda x: np.stack([x[t] for t in test])
    out = dict(J=J, ns=ns)
    out["icl"] = np.stack([by_test(D['R_icl'][:, min(n, args.k_max)]) for n in ns])
    ft = []
    for i in range(len(ns)):
        score = np.mean([np.log(D['R_ft'][i][:, :, val(j)].mean(-1)) for j in J], axis=0)
        l, c = np.unravel_index(np.argmin(score), score.shape)
        ft.append(by_test(D['R_ft'][i, l, c]))
    out["ft"] = np.stack(ft)
    raw = {"norm": np.log(G['gnorm_tr']), "align": G['cos_val'], "adam": G['cos_adam'], "loss0": np.log(G['L_train0'])}
    out["feats"] = {f: np.stack([by_test(v[i]) for i in range(len(ns))]) for f, v in raw.items()}
    return out

runs = [(kind, s, load(p)) for kind in ("iso", "aniso") for s, p in enumerate(getattr(args, kind))]
kinds = np.array([r[0] for r in runs])
J, ns = list(runs[0][2]["J"]), runs[0][2]["ns"]
ia, ib = J.index(args.dir_a), J.index(args.dir_b)
null = [i for i in range(len(J)) if i not in (ia, ib)]
M, N = len(runs), len(ns)
#cells (M, N, J): log of the mean risk over test datasets, mean feature
L_icl = np.log(np.stack([r[2]["icl"].mean(-1) for r in runs]))
L_ft = np.log(np.stack([r[2]["ft"].mean(-1) for r in runs]))
F = {f: np.stack([r[2]["feats"][f].mean(-1) for r in runs]) for f in FEATURES}
center = lambda x: x - x.mean(-1, keepdims=True)
Y, Fc = center(L_ft), {f: center(v) for f, v in F.items()}

def predict(feats, icl_specific):
    """Leave-one-model-out predictions (log ICL, log FT) of every cell."""
    dev = np.zeros_like(L_ft)
    for m in range(M):
        others = [o for o in range(M) if o != m]
        for i in range(N):
            if feats:
                X = np.stack([Fc[f][others, i].ravel() for f in feats], 1)
                b = np.linalg.lstsq(X, Y[others, i].ravel(), rcond=None)[0]
                dev[m, i] = np.stack([Fc[f][m, i] for f in feats], 1) @ b
    icl = L_icl if icl_specific else np.broadcast_to(L_icl.mean(-1, keepdims=True), L_icl.shape)
    return icl, L_ft.mean(-1, keepdims=True) + dev

PREDICTORS = {"none": ([], False), "ICL only": ([], True), "gradients only (|g|, cos)": (["norm", "align"], False),
              "ICL + |g|": (["norm"], True), "ICL + cos(g, g_val)": (["align"], True),
              "ICL + |g| + cos(g, g_val)": (["norm", "align"], True), "ICL + |g| + Adam cos": (["norm", "adam"], True),
              "ICL + initial loss (control)": (["loss0"], True)}
preds = {name: predict(*p) for name, p in PREDICTORS.items()}
preds["oracle"] = (L_icl, L_ft)

def nstar(gap):
    """Crossing of the log gap log(R_ICL / R_FT) through 0, linear in log n (end segments extrapolated)."""
    ln = np.log(ns)
    i = next((i for i in range(N - 1) if gap[i] < 0 <= gap[i + 1]), 0 if gap[0] >= 0 else N - 2)
    return np.exp(ln[i] - gap[i] * (ln[i + 1] - ln[i]) / (gap[i + 1] - gap[i]))

true_gap = L_icl - L_ft
true_nstar = np.array([[nstar(true_gap[m, :, j]) for j in range(len(J))] for m in range(M)])
i_rev = list(ns).index(args.n_reversal)
aniso = kinds == "aniso"

def scores(icl, ft):
    gap = icl - ft
    n_hat = np.array([[nstar(gap[m, :, j]) for j in range(len(J))] for m in range(M)])
    err = np.abs(np.log2(n_hat / true_nstar))
    out = {}
    for kind in ("iso", "aniso"):
        k = kinds == kind
        out[kind] = dict(rmse=np.sqrt(np.mean((gap[k] - true_gap[k]) ** 2)),
                         r2=1 - np.sum((center(ft)[k] - Y[k]) ** 2) / np.sum(Y[k] ** 2),
                         nerr=err[k].mean(), nerr_ab=err[k][:, [ia, ib]].mean())
    out["nstar"] = n_hat
    out["reversal"] = int(np.sum((gap[aniso, i_rev, ia] < 0) & (gap[aniso, i_rev, ib] > 0)))
    return out

S = {name: scores(*p) for name, p in preds.items()}
gmean = lambda x: np.exp(np.mean(np.log(x)))

#per-dataset predictive value: rank correlation over the test datasets of one cell between a feature and log R_FT
def spearman(x, y):
    rx, ry = x.argsort(-1).argsort(-1).astype(float), y.argsort(-1).argsort(-1).astype(float)
    rx, ry = center(rx), center(ry)
    return (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))

lines = ["Direction dependence of the ICL-FT boundary at delta=60: A = e_{0}, B = e_{1}; ICL k = min(n, {2}); FT recipe "
         "global over directions; {3} iso + {4} aniso models, {5} directions, n = {6}".format(
             args.dir_a + 1, args.dir_b + 1, args.k_max, int((~aniso).sum()), int(aniso.sum()), len(J), [int(n) for n in ns]), ""]
lines.append("gradient features (mean over models and test datasets; +-2 SE over datasets); FT risk for reference")
for kind in ("iso", "aniso"):
    for i, n in enumerate(ns):
        row = "  {0:5s} n={1:4d}".format(kind, n)
        for g, idx in (("A", [ia]), ("B", [ib]), ("null", null)):
            v = {f: np.concatenate([r[2]["feats"][f][i][idx].ravel() for r in runs if r[0] == kind]) for f in FEATURES}
            ft = np.concatenate([r[2]["ft"][i][idx].ravel() for r in runs if r[0] == kind])
            row += "   {0:4s} |g| {1:.3f} cos {2:.3f}({3:.3f}) adam {4:.3f} FT {5:.2e}".format(
                g, np.exp(v["norm"].mean()), v["align"].mean(), 2 * v["align"].std(ddof=1) / np.sqrt(len(v["align"])),
                v["adam"].mean(), ft.mean())
        lines.append(row)
lines += ["", "within-model direction effect: correlation over cells of centered feature with centered log FT risk"]
for kind in ("iso", "aniso"):
    k = kinds == kind
    lines.append("  {0:5s} ".format(kind) + "  ".join(
        "{0} {1}".format(f, " ".join("{0:+.2f}".format(np.corrcoef(Fc[f][k, i].ravel(), Y[k, i].ravel())[0, 1])
                                     for i in range(N))) for f in FEATURES) + "   (per n)")
lines += ["", "predictors (leave-one-model-out): RMSE of log gap over cells, R^2 of the FT direction effect, mean |log2 n* "
          "error| over all directions and over A,B; n* of A and B (geometric mean over aniso seeds); seeds with the "
          "n={0} reversal (Delta_A < 0, Delta_B > 0)".format(args.n_reversal)]
lines.append("  {0:30s} {1:>23s}   {2:>30s}   {3:>15s}  {4:>9s}".format(
    "", "iso: RMSE  R2   n*err", "aniso: RMSE   R2   n*err  n*errAB", "aniso n*_A n*_B", "reversal"))
for name, s in S.items():
    lines.append("  {0:30s} {1:9.3f} {2:5.2f} {3:6.2f}   {4:12.3f} {5:5.2f} {6:6.2f} {7:8.2f}   {8:7.1f} {9:6.1f}  {10:5d}/{11}".format(
        name, s["iso"]["rmse"], s["iso"]["r2"], s["iso"]["nerr"], s["aniso"]["rmse"], s["aniso"]["r2"],
        s["aniso"]["nerr"], s["aniso"]["nerr_ab"], gmean(s["nstar"][aniso, ia]), gmean(s["nstar"][aniso, ib]),
        s["reversal"], int(aniso.sum())))
lines += ["", "per-dataset predictive value: Spearman correlation over the {0} test datasets of a cell between the feature "
          "and log R_FT, mean (+-2 SE) over cells".format(runs[0][2]["ft"].shape[-1])]
for kind in ("iso", "aniso"):
    for i, n in enumerate(ns):
        ft = np.log(np.stack([r[2]["ft"][i] for r in runs if r[0] == kind]))
        rho = {f: spearman(np.stack([r[2]["feats"][f][i] for r in runs if r[0] == kind]), ft).ravel() for f in FEATURES}
        lines.append("  {0:5s} n={1:4d}  ".format(kind, n) + "  ".join(
            "{0} {1:+.2f}({2:.2f})".format(f, v.mean(), 2 * v.std(ddof=1) / np.sqrt(len(v))) for f, v in rho.items()))
text = "\n".join(lines)
print(text)
(out_dir / "gradfeat_delta60.txt").write_text(text + "\n")

def style(ax, logy=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if logy:
        ax.set_yscale("log")

#figure 1: features against n for A, B and the null directions
fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), facecolor=SURFACE)
for row, kind in enumerate(("iso", "aniso")):
    for col, (f, transform) in enumerate((("norm", np.exp), ("align", None), ("adam", None))):
        ax = axes[row, col]
        V = np.concatenate([r[2]["feats"][f] for r in runs if r[0] == kind], axis=-1)   # (N, J, reps)
        for idx, c, label in (([ia], A_C, "A = e$_{0}$".format(args.dir_a + 1)), ([ib], B_C, "B = e$_{0}$".format(args.dir_b + 1))):
            x = V[:, idx].reshape(N, -1)
            mu, se = x.mean(-1), x.std(-1, ddof=1) / np.sqrt(x.shape[-1])
            t = transform or (lambda z: z)
            ax.fill_between(ns, t(mu - 2 * se), t(mu + 2 * se), color=c, alpha=0.18, lw=0)
            ax.plot(ns, t(mu), color=c, marker="o", ms=3.5, lw=1.6, label=label)
        mus = V[:, null].mean(-1)
        t = transform or (lambda z: z)
        ax.fill_between(ns, t(mus.min(1)), t(mus.max(1)), color=NULL_C, alpha=0.35, lw=0, label="other directions: range")
        style(ax, logy=transform is not None)
        if transform is not None:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("fine-tuning examples n")
        ax.set_title("{0}: {1}".format("isotropic cap" if kind == "iso" else "anisotropic cap",
                                       {"norm": r"training-gradient norm $|g_{tr}|$", "align": r"alignment cos($g_{tr}$, $g_{val}$)",
                                        "adam": r"Adam-step alignment cos($u_{tr}$, $g_{val}$)"}[f]), loc="left", fontsize=9)
axes[1, 2].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Gradient features at the pretrained checkpoint (step 0 of FT, zero-context loss on the n FT examples; "
             r"$g_{val}$: 1024 held-out examples).  Bands: ±2 SE over test datasets pooled over seeds.",
             x=0.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out_dir / "gradfeat_delta60.png", dpi=150, facecolor=SURFACE)

#figure 2: FT direction effect and n* predicted with and without gradient features
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
best = "ICL + |g| + cos(g, g_val)"
group = np.full(len(J), "null", dtype=object)
group[ia], group[ib] = "A", "B"
colors = {"A": A_C, "B": B_C, "null": NULL_C}
ax = axes[0]
ft_hat = center(preds[best][1])
for g in ("null", "A", "B"):
    for kind, fill in (("iso", False), ("aniso", True)):
        k, j = kinds == kind, group == g
        ax.scatter(Y[k][:, :, j].ravel(), ft_hat[k][:, :, j].ravel(), s=16, lw=0.8, edgecolor=colors[g],
                   facecolor=colors[g] if fill else "none", label="{0}, {1}".format(g, kind))
lim = np.abs(np.concatenate([Y.ravel(), ft_hat.ravel()])).max() * 1.1
ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls=":")
style(ax)
ax.set_xlabel(r"measured log $R_{FT,j}$ $-$ direction mean")
ax.set_ylabel("predicted from |g|, cos (leave one model out)")
ax.set_title("FT direction effect, R$^2$ (aniso) = {0:.2f}".format(S[best]["aniso"]["r2"]), loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
for ax, name in zip(axes[1:], ("ICL only", best)):
    for g in ("null", "A", "B"):
        for kind, fill in (("iso", False), ("aniso", True)):
            k, j = kinds == kind, group == g
            ax.scatter(true_nstar[k][:, j].ravel(), S[name]["nstar"][k][:, j].ravel(), s=22, lw=0.9, edgecolor=colors[g],
                       facecolor=colors[g] if fill else "none")
    ax.plot([16, 160], [16, 160], color=MUTED, lw=0.8, ls=":")
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = [16, 32, 64, 128]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlim(14, 180)
    ax.set_ylim(14, 180)
    ax.set_xlabel("measured crossover n*")
    ax.set_ylabel("predicted n*")
    ax.set_title("{0}\nmean |log$_2$ n* error|: aniso {1:.2f}, iso {2:.2f}".format(
        name, S[name]["aniso"]["nerr"], S[name]["iso"]["nerr"]), loc="left", fontsize=9)
fig.suptitle("Predicting the direction dependence of the ICL–FT boundary (filled: anisotropic cap, hollow: isotropic). "
             "All predictors use the model's direction-averaged FT curve.", x=0.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_dir / "gradfeat_predict_delta60.png", dpi=150, facecolor=SURFACE)
print("saved figures to", out_dir)
