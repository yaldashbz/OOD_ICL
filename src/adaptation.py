import torch
import torch.nn.functional as F
import numpy as np
import transformer

#The pretraining caps from dataset_utils.sample_cone are centered on v = -e_1
def pole(dim, device='cuda'):
    v = torch.zeros(dim, device=device)
    v[0] = -1.
    return v

def sample_tasks_at_angle(n, dim, delta, gen, device='cuda'):
    """Unit-norm task vectors w with angle(w, v) = delta (radians) exactly, uniform over that ring."""
    v = pole(dim, device)
    u = torch.randn(n, dim, generator=gen, device=device)
    u = u - (u @ v)[:, None] * v[None, :]
    u = u / u.norm(dim=1, keepdim=True)
    return np.cos(delta) * v[None, :] + np.sin(delta) * u

def make_adaptation_data(dim, seq_len, deltas, tasks_val, tasks_test, n_ctx, n_query, n_train, sigma, seed, device='cuda'):
    """Downstream tasks and data for run_adaptation.py, drawn in a fixed order from one seeded generator: for each
    split (val, test) and delta (degrees), the task vectors W; then demonstration sequences Xc with noiseless / noisy
    labels Fc / Yc, zero-context query points Xq with noiseless labels Fq, and a nested pool of FT examples Xtr, Ytr
    (the first n are the FT set of size n). CUDA draws are reproducible only on the same GPU model."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    member_delta, member_split, W = [], [], []
    for split, n_tasks in (("val", tasks_val), ("test", tasks_test)):
        for d in deltas:
            W.append(sample_tasks_at_angle(n_tasks, dim, d * np.pi / 180, gen, device))
            member_delta += [d] * n_tasks
            member_split += [split] * n_tasks
    W = torch.cat(W)
    E = W.shape[0]

    def labels(xs):
        f = torch.einsum('e...d,ed->e...', xs, W)
        return f, f + sigma * torch.randn(f.shape, generator=gen, device=device)

    Xc = torch.randn(E, n_ctx, seq_len, dim, generator=gen, device=device)
    Fc, Yc = labels(Xc)
    Xq = torch.randn(E, n_query, 1, dim, generator=gen, device=device)
    Fq, _ = labels(Xq)
    Xtr = torch.randn(E, n_train, 1, dim, generator=gen, device=device)
    _, Ytr = labels(Xtr)
    return dict(W=W, member_delta=np.array(member_delta), member_split=np.array(member_split),
                Xc=Xc, Fc=Fc, Yc=Yc, Xq=Xq, Fq=Fq, Xtr=Xtr, Ytr=Ytr)

def make_direction_data(dim, seq_len, delta, directions, reps_val, reps_test, n_ctx, n_query, n_train, sigma, seed,
                        device='cuda'):
    """Like make_adaptation_data, but for fixed targets w_j = cos(delta) v + sin(delta) e_j (delta in degrees), one per
    perpendicular coordinate j in directions, each repeated with independent datasets. Datasets are paired across
    directions: repetition r of a split uses the same inputs and label noise for every j. Members are ordered by
    split, then direction, then repetition."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    v = pole(dim, device)
    reps = {"val": reps_val, "test": reps_test}
    member_split, member_direction, member_rep, W, base = [], [], [], [], []
    for split in ("val", "test"):
        for j in directions:
            w = np.cos(delta * np.pi / 180) * v
            w[j] += np.sin(delta * np.pi / 180)
            W.append(w[None, :].repeat(reps[split], 1))
            member_split += [split] * reps[split]
            member_direction += [j] * reps[split]
            member_rep += list(range(reps[split]))
            base += list(range(reps[split]) if split == "val" else range(reps_val, reps_val + reps_test))
    W = torch.cat(W)
    base = torch.tensor(base, device=device)
    R = reps_val + reps_test

    def labels(xs, noise):
        f = torch.einsum('e...d,ed->e...', xs, W)
        return f, f + sigma * noise

    Xc = torch.randn(R, n_ctx, seq_len, dim, generator=gen, device=device)[base]
    Fc, Yc = labels(Xc, torch.randn(R, n_ctx, seq_len, generator=gen, device=device)[base])
    Xq = torch.randn(R, n_query, 1, dim, generator=gen, device=device)[base]
    Fq, _ = labels(Xq, torch.zeros(1, device=device))
    Xtr = torch.randn(R, n_train, 1, dim, generator=gen, device=device)[base]
    _, Ytr = labels(Xtr, torch.randn(R, n_train, 1, generator=gen, device=device)[base])
    return dict(W=W, member_delta=np.full(len(member_split), float(delta)), member_split=np.array(member_split),
                member_direction=np.array(member_direction), member_rep=np.array(member_rep),
                Xc=Xc, Fc=Fc, Yc=Yc, Xq=Xq, Fq=Fq, Xtr=Xtr, Ytr=Ytr)

def stack_params(model, E):
    """Copy a RegressionTransformer's weights into a dict of tensors with a leading dim of E independent
    copies (attention heads concatenated along the output dim), for batched_forward."""
    t = model.transformer
    P = {'embed_w': model.embed.weight, 'embed_b': model.embed.bias,
         'pos': t.position_embedding.embd.weight,
         'lnf_w': t.ln_f.weight, 'lnf_b': t.ln_f.bias,
         'out_w': t.lm_head.weight, 'out_b': t.lm_head.bias}
    for l, blk in enumerate(t.blocks):
        P['{0}.q'.format(l)] = torch.cat([h.query.weight for h in blk.sa.heads], 0)
        P['{0}.k'.format(l)] = torch.cat([h.key.weight for h in blk.sa.heads], 0)
        P['{0}.v'.format(l)] = torch.cat([h.value.weight for h in blk.sa.heads], 0)
        P['{0}.proj_w'.format(l)], P['{0}.proj_b'.format(l)] = blk.sa.proj.weight, blk.sa.proj.bias
        P['{0}.ln1_w'.format(l)], P['{0}.ln1_b'.format(l)] = blk.ln1.weight, blk.ln1.bias
        P['{0}.ln2_w'.format(l)], P['{0}.ln2_b'.format(l)] = blk.ln2.weight, blk.ln2.bias
        P['{0}.ff1_w'.format(l)], P['{0}.ff1_b'.format(l)] = blk.ffwd.net[0].weight, blk.ffwd.net[0].bias
        P['{0}.ff2_w'.format(l)], P['{0}.ff2_b'.format(l)] = blk.ffwd.net[2].weight, blk.ffwd.net[2].bias
    return {k: p.detach().unsqueeze(0).repeat(E, *[1] * p.dim()).clone() for k, p in P.items()}

def _linear(x, w, b=None):
    #x: (E, ..., in), w: (E, out, in), b: (E, out)
    E = x.shape[0]
    y = torch.bmm(x.reshape(E, -1, x.shape[-1]), w.transpose(1, 2)).reshape(*x.shape[:-1], w.shape[1])
    if b is not None:
        y = y + b.view(E, *[1] * (x.dim() - 2), -1)
    return y

def _layer_norm(x, w, b, eps=1e-5):
    E = x.shape[0]
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    shape = (E, *[1] * (x.dim() - 2), -1)
    return (x - mu) / torch.sqrt(var + eps) * w.view(shape) + b.view(shape)

def batched_forward(P, xs, ys, n_head=8):
    """Same computation as RegressionTransformer.forward with all dropout disabled, for E stacked copies.
    xs: (E, B, t, d), ys: (E, B, t) scalar labels. Returns predictions (E, B, t), where entry [..., i] is
    the prediction for y_i given the context (x_1, y_1, ..., x_i), i.e. with i in-context demonstrations."""
    E, B, t, d = xs.shape
    T = 2 * t - 1
    n_layer = sum(1 for k in P if k.endswith('.q'))
    ys_pad = torch.zeros(E, B, t, d, device=xs.device)
    ys_pad[..., 0] = ys
    seq = torch.empty(E, B, T, d, device=xs.device)
    seq[:, :, 0::2] = xs
    seq[:, :, 1::2] = ys_pad[:, :, :t - 1]

    h = _linear(seq, P['embed_w'], P['embed_b']) + P['pos'][:, None, :T]
    C = h.shape[-1]
    hs = C // n_head
    for l in range(n_layer):
        a = _layer_norm(h, P['{0}.ln1_w'.format(l)], P['{0}.ln1_b'.format(l)])
        q, k, v = [_linear(a, P['{0}.{1}'.format(l, m)]).reshape(E * B, T, n_head, hs).transpose(1, 2)
                   for m in ('q', 'k', 'v')]
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(E, B, T, C)
        h = h + _linear(att, P['{0}.proj_w'.format(l)], P['{0}.proj_b'.format(l)])
        f = _layer_norm(h, P['{0}.ln2_w'.format(l)], P['{0}.ln2_b'.format(l)])
        f = _linear(F.relu(_linear(f, P['{0}.ff1_w'.format(l)], P['{0}.ff1_b'.format(l)])),
                    P['{0}.ff2_w'.format(l)], P['{0}.ff2_b'.format(l)])
        h = h + f
    h = _layer_norm(h, P['lnf_w'], P['lnf_b'])
    return _linear(h, P['out_w'], P['out_b'])[..., 0][:, :, 0::2]

def load_pretrained(run_dir, device='cuda'):
    import json
    args = json.load(open(run_dir + "/args.json"))
    model = transformer.RegressionTransformer(d_model=args['d_model'], device=device, block_size=args['seq_len'] * 2,
                                              n_layer=args['n_layer'], input_dim=args['dim']).to(device)
    model.load_state_dict(torch.load(run_dir + "/checkpoints/final.th", map_location=device))
    model.eval()
    return model, args
