import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from adaptation import load_pretrained, stack_params, batched_forward, make_adaptation_data

#Is the loss jump when demonstrations are added after zero-context fine-tuning (icl_after_ft figure) a prompt-format
#mismatch? Fine-tunes pretrained models on n examples of a target task at angle delta in two ways:
#  zero:     each example is the length-1 sequence [x] -> y, as in run_adaptation.py
#  variable: prompts of L = min(n, seq_len) distinct examples from the same n, in random order, with the loss on
#            every position, so training covers 0..L-1 demonstrations in context
#At every checkpoint it evaluates the risk with k = 0..seq_len-1 fresh demonstrations of the same task, and of a
#different task at the same delta (does the model still learn in context, or does it just know the FT task?).
#All pretrained models are stacked as independent copies (one block of tasks each) and trained in one batch.
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", type=str, nargs="+", required=True, help="run directories of pretrained models")
parser.add_argument("--delta", type=float, default=90., help="target angle (degrees)")
parser.add_argument("--tasks_val", type=int, default=8, help="tasks per model used only to select lr / steps")
parser.add_argument("--tasks_test", type=int, default=16, help="tasks per model used for reporting")
parser.add_argument("--ns", type=int, nargs="+", default=[16, 256], help="fine-tuning set sizes n")
parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
parser.add_argument("--ckpts", type=int, nargs="+", default=[0, 10, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000],
                    help="fine-tuning steps at which all risks are evaluated; the last one is the step budget")
parser.add_argument("--prompts", type=int, default=16, help="prompts per step for variable-context fine-tuning")
parser.add_argument("--n_ctx", type=int, default=128, help="evaluation contexts per task")
parser.add_argument("--chunk", type=int, default=32, help="contexts per chunk in the evaluation forward pass")
parser.add_argument("--out", type=str, default="../runs/adaptation/ft_context_phi150_delta90.npz")
args = parser.parse_args()

device = 'cuda'
VARIANTS = ("zero", "variable")
P0, blocks, seq_len = [], [], None
for run_dir in args.pretrained:
    model, margs = load_pretrained(run_dir, device)
    seq_len = margs['seq_len']
    #tasks and data drawn per model from its own seed, so the models' task sets are independent
    blocks.append(make_adaptation_data(margs['dim'], seq_len, [args.delta], args.tasks_val, args.tasks_test, args.n_ctx, 1,
                                       max(args.ns), 0., margs['seed'], device))
    P0.append(stack_params(model, len(blocks[-1]['member_split'])))
    blocks[-1]['seed'] = np.full(len(blocks[-1]['member_split']), margs['seed'])
P0 = {k: torch.cat([p[k] for p in P0]) for k in P0[0]}
cat = lambda key: torch.cat([b[key] for b in blocks]) if torch.is_tensor(blocks[0][key]) else np.concatenate([b[key] for b in blocks])
W, member_split, member_seed = cat('W'), cat('member_split'), cat('seed')
Xc, Fc, Yc, Xtr, Ytr = [cat(k) for k in ('Xc', 'Fc', 'Yc', 'Xtr', 'Ytr')]
E = W.shape[0]
#each copy's "other task": the next task in its own model's block
offsets = np.cumsum([0] + [len(b['member_split']) for b in blocks])
partner = np.concatenate([lo + (np.arange(hi - lo) + 1) % (hi - lo) for lo, hi in zip(offsets[:-1], offsets[1:])])
print("copies:", E, "delta:", args.delta, "pretrained:", args.pretrained, flush=True)

@torch.no_grad()
def eval_curve(P, idx):
    """Excess risk per copy for k = 0..seq_len-1 demonstrations of task idx[e]: (E, seq_len)."""
    out = []
    for i in range(0, args.n_ctx, args.chunk):
        sl = slice(i, i + args.chunk)
        pred = batched_forward(P, Xc[idx, sl], Yc[idx, sl])
        out.append(((pred - Fc[idx, sl]) ** 2).sum(1))
    return (sum(out) / args.n_ctx).cpu().numpy()

gen = torch.Generator(device=device)
gen.manual_seed(1234)

def train_loss(P, variant, x, y):
    """Per-copy training loss on the n examples x (E, n, d), y (E, n)."""
    if variant == "zero":
        pred = batched_forward(P, x[:, :, None], torch.zeros(x.shape[:2] + (1,), device=device))[..., 0]
        return ((pred - y) ** 2).mean(1)
    n, d = x.shape[1:]
    L = min(n, seq_len)
    idx = torch.rand(E, args.prompts, n, generator=gen, device=device).argsort(-1)[..., :L]
    xs = torch.gather(x[:, None].expand(E, args.prompts, n, d), 2, idx[..., None].expand(-1, -1, -1, d))
    ys = torch.gather(y[:, None].expand(E, args.prompts, n), 2, idx)
    return ((batched_forward(P, xs, ys) - ys) ** 2).mean((1, 2))

t0 = time.time()
own = np.arange(E)
shape = (len(VARIANTS), len(args.ns), len(args.lrs), len(args.ckpts), E)
R_own = np.zeros(shape + (seq_len,), dtype=np.float32)
R_other = np.zeros(shape + (seq_len,), dtype=np.float32)
L_train = np.zeros(shape, dtype=np.float32)
for v, variant in enumerate(VARIANTS):
    for i, n in enumerate(args.ns):
        x_tr, y_tr = Xtr[:, :n, 0], Ytr[:, :n, 0]
        for j, lr in enumerate(args.lrs):
            P = {k: p.clone().requires_grad_(True) for k, p in P0.items()}
            optimizer = torch.optim.Adam(P.values(), lr=lr)
            c = 0
            for step in range(args.ckpts[-1] + 1):
                loss_per_task = train_loss(P, variant, x_tr, y_tr)
                if step == args.ckpts[c]:
                    R_own[v, i, j, c] = eval_curve(P, own)
                    R_other[v, i, j, c] = eval_curve(P, partner)
                    L_train[v, i, j, c] = loss_per_task.detach().cpu().numpy()
                    c += 1
                    if step == args.ckpts[-1]:
                        break
                optimizer.zero_grad()
                loss_per_task.sum().backward()
                optimizer.step()
            test = member_split == "test"
            med = lambda R, k: np.array2string(np.median(R[v, i, j, :, test, k], axis=0), precision=3)
            print("{0:8s} n={1:4d} lr={2:.0e}  ({3:.1f} min)\n  own k=0  {4}\n  own k=20 {5}\n  other k=20 {6}".format(
                variant, n, lr, (time.time() - t0) / 60, med(R_own, 0), med(R_own, 20), med(R_other, 20)), flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, R_own=R_own, R_other=R_other, L_train=L_train, variants=np.array(VARIANTS), member_split=member_split,
         member_seed=member_seed, partner=partner, W=W.cpu().numpy(), ns=np.array(args.ns), lrs=np.array(args.lrs),
         ckpts=np.array(args.ckpts), delta=args.delta, args=json.dumps(vars(args)))
print("saved", out, "({0:.1f} min)".format((time.time() - t0) / 60))
