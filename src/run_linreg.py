import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from train_linreg import train

#Paper defaults (Goddard et al. 2025, Sec 3 & App A.1): d=10, n=50, d_h=128, 10 layers, 8 heads,
#AdamW, constant lr 3e-4, batch size 128, 58,000 steps (= epochs * batches_per_epoch)
parser = argparse.ArgumentParser(description="Train one in-context linear regression model on a hyperspherical cap.")
parser.add_argument("--angle", type=int, required=True, help="half-angle phi of the training cap, in degrees")
parser.add_argument("--log2_tasks", type=int, default=11, help="number of pretraining tasks N = 2**log2_tasks")
parser.add_argument("--noise_std", type=float, default=0.)
parser.add_argument("--epochs", type=int, default=58)
parser.add_argument("--batches_per_epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--n_layer", type=int, default=10)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--dim", type=int, default=10)
parser.add_argument("--seq_len", type=int, default=50)
parser.add_argument("--save_freq", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run_dir", type=str, default="runs/")
parser.add_argument("--task_scales", type=float, nargs="+", default=None,
                    help="dim-1 scales of the coordinates perpendicular to the cap pole (anisotropic cap); omit for isotropic")
args = parser.parse_args()
assert args.task_scales is None or len(args.task_scales) == args.dim - 1, "need dim-1 task scales"

np.random.seed(args.seed)
torch.manual_seed(args.seed)

run_dir = Path(args.run_dir) / "seed{0}_noise{1}_tasks{2}_angle{3}{4}".format(
    args.seed, args.noise_std, args.log2_tasks, args.angle, "" if args.task_scales is None else "_aniso")
run_dir.mkdir(parents=True, exist_ok=True)
with open(run_dir / "args.json", "w") as f:
    json.dump(vars(args), f, indent=2)

t0 = time.time()
model, loss_history, test_loss_history, angles, losses = train(
    batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
    seq_len=args.seq_len, d_model=args.d_model, n_layer=args.n_layer, dim=args.dim, noise_std=args.noise_std,
    checkpoint_dir=str(run_dir / "checkpoints") + "/", output_dir=str(run_dir) + "/",
    pretrain_size=2**args.log2_tasks, angle=args.angle, save_freq=args.save_freq, task_scales=args.task_scales)

np.save(run_dir / "test_angles.npy", np.array(angles))
print("Total time: {0:.1f} minutes".format((time.time() - t0) / 60))
