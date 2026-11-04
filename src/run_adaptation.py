import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from adaptation import load_pretrained, stack_params, batched_forward, make_adaptation_data, make_direction_data

#Compares three ways of adapting a pretrained ICL transformer to a downstream task w_target at angle delta
#from the pretraining pole v, with y = w_target^T x + eps:
#  ICL only:  frozen weights, k demonstrations in context
#  FT only:   fine-tune all weights on n examples with a zero-context objective (each example is the
#             length-1 sequence [x] -> y), evaluate with no demonstrations
#  FT + ICL:  the fine-tuned weights, evaluated with k demonstrations
#Every (task, n, lr) fine-tuning run is an independent copy of the network; copies are trained in parallel
#with batched_forward (Adam is elementwise and the summed loss decouples copies, so this is exact).
#All risks are excess risks E_x[(y_hat - w_target^T x)^2] against the noiseless target. Dropout is off.
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", type=str, required=True, help="run directory of a pretrained model")
parser.add_argument("--noise_var", type=float, default=0., help="label noise variance sigma^2 of the downstream task")
parser.add_argument("--deltas", type=float, nargs="+", default=list(range(0, 181, 15)), help="target angles (degrees)")
parser.add_argument("--tasks_val", type=int, default=4, help="tasks per delta used only to select FT lr / steps")
parser.add_argument("--tasks_test", type=int, default=8, help="tasks per delta used for reporting")
parser.add_argument("--ns", type=int, nargs="+", default=[2**i for i in range(11)], help="fine-tuning set sizes n")
parser.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
parser.add_argument("--ckpts", type=int, nargs="+", default=[0, 10, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000],
                    help="fine-tuning steps at which all risks are evaluated; the last one is the step budget")
parser.add_argument("--n_ctx", type=int, default=64, help="evaluation contexts per task for ICL and FT+ICL")
parser.add_argument("--n_query", type=int, default=1024, help="evaluation points per task for FT only")
parser.add_argument("--chunk", type=int, default=32, help="tasks per chunk in the ICL evaluation forward pass")
parser.add_argument("--data_seed", type=int, default=None, help="defaults to the pretrained model's seed")
parser.add_argument("--directions", type=int, nargs="+", default=None,
                    help="evaluate fixed targets cos(delta) v + sin(delta) e_j for these coordinates j (1..dim-1) at the single "
                         "delta in --deltas, with --tasks_val / --tasks_test paired datasets per direction")
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()
assert args.directions is None or len(args.deltas) == 1, "--directions takes a single --deltas value"

device = 'cuda'
model, margs = load_pretrained(args.pretrained, device)
dim, seq_len = margs['dim'], margs['seq_len']
data_seed = margs['seed'] if args.data_seed is None else args.data_seed
#tasks: for each split, for each delta, tasks_split independent w_target (or, with --directions, fixed targets with
#paired datasets); fresh demonstrations shared by ICL and FT+ICL, FT-only query points, and a nested pool of FT data
if args.directions is None:
    data = make_adaptation_data(dim, seq_len, args.deltas, args.tasks_val, args.tasks_test, args.n_ctx, args.n_query,
                                max(args.ns), np.sqrt(args.noise_var), data_seed, device)
else:
    data = make_direction_data(dim, seq_len, args.deltas[0], args.directions, args.tasks_val, args.tasks_test, args.n_ctx,
                               args.n_query, max(args.ns), np.sqrt(args.noise_var), data_seed, device)
W, member_delta, member_split = data['W'], data['member_delta'], data['member_split']
Xc, Fc, Yc, Xq, Fq, Xtr, Ytr = [data[k] for k in ('Xc', 'Fc', 'Yc', 'Xq', 'Fq', 'Xtr', 'Ytr')]
E = W.shape[0]
print("tasks:", E, "pretrained:", args.pretrained, "noise_var:", args.noise_var, flush=True)

@torch.no_grad()
def eval_in_context(P):
    """Excess risk per task for k = 0..seq_len-1 demonstrations: (E, seq_len)."""
    out = []
    for i in range(0, E, args.chunk):
        Pc = {k: v[i:i + args.chunk] for k, v in P.items()}
        pred = batched_forward(Pc, Xc[i:i + args.chunk], Yc[i:i + args.chunk])
        out.append(((pred - Fc[i:i + args.chunk]) ** 2).mean(1))
    return torch.cat(out).cpu().numpy()

@torch.no_grad()
def eval_zero_context(P):
    """Excess risk per task with no demonstrations: (E,)."""
    pred = batched_forward(P, Xq, torch.zeros(Xq.shape[:3], device=device))[..., 0]
    return ((pred - Fq[..., 0]) ** 2).mean(1).cpu().numpy()

t0 = time.time()
P0 = stack_params(model, E)
R_icl = eval_in_context(P0)
R_zero = eval_zero_context(P0)

shape = (len(args.ns), len(args.lrs), len(args.ckpts))
R_ft = np.zeros(shape + (E,), dtype=np.float32)
R_ft_icl = np.zeros(shape + (E, seq_len), dtype=np.float32)
L_train = np.zeros(shape + (E,), dtype=np.float32)
for i, n in enumerate(args.ns):
    x_tr, y_tr = Xtr[:, :n], Ytr[:, :n, 0]
    for j, lr in enumerate(args.lrs):
        P = {k: v.clone().requires_grad_(True) for k, v in P0.items()}
        optimizer = torch.optim.Adam(P.values(), lr=lr)
        c = 0
        for step in range(args.ckpts[-1] + 1):
            if step == args.ckpts[c]:
                R_ft[i, j, c] = eval_zero_context(P)
                R_ft_icl[i, j, c] = eval_in_context(P)
            pred = batched_forward(P, x_tr, torch.zeros(x_tr.shape[:3], device=device))[..., 0]
            loss_per_task = ((pred - y_tr) ** 2).mean(1)
            if step == args.ckpts[c]:
                L_train[i, j, c] = loss_per_task.detach().cpu().numpy()
                c += 1
                if step == args.ckpts[-1]:
                    break
            optimizer.zero_grad()
            loss_per_task.sum().backward()
            optimizer.step()
        test = member_split == "test"
        print("n={0:5d} lr={1:.0e}: median test R_ft over steps {2}  ({3:.1f} min)".format(
            n, lr, np.array2string(np.median(R_ft[i, j][:, test], axis=1), precision=4), (time.time() - t0) / 60), flush=True)

name = "adaptation_noise{0}.npz" if args.directions is None else "adaptation_dirs_delta{1:g}_noise{0}.npz"
out = Path(args.out or (args.pretrained.rstrip("/") + "/" + name.format(args.noise_var, args.deltas[0])))
extra = {} if args.directions is None else dict(member_direction=data['member_direction'], member_rep=data['member_rep'])
np.savez(out, R_icl=R_icl, R_zero=R_zero, R_ft=R_ft, R_ft_icl=R_ft_icl, L_train=L_train,
         member_delta=member_delta, member_split=member_split, W=W.cpu().numpy(),
         ns=np.array(args.ns), lrs=np.array(args.lrs), ckpts=np.array(args.ckpts),
         phi=margs['angle'], noise_var=args.noise_var, args=json.dumps(vars(args)), **extra)
print("saved", out, "({0:.1f} min)".format((time.time() - t0) / 60))
