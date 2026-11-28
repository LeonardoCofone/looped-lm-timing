import contextlib
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from measure.common import base_parser, for_each_run, initial_state
from measure.time_scale import offset_sequences


@contextlib.contextmanager
def scaled_weights(m, c):
    # the weight path theta(c): decay a -> 1 - c(1 - a) (through A_log, the step size dt unchanged), injection B -> cB,
    # output projections of the core branches -> c W; the linear part of the update scales exactly by c
    inj = m.inject
    saved = [inj.A_log.detach().clone(), inj.B.detach().clone()]
    projs = [b.attn.proj.weight for b in m.core] + [b.mlp.proj.weight for b in m.core]
    saved += [w.detach().clone() for w in projs]
    with torch.no_grad():
        dt = F.softplus(inj.dt_bias)
        a = torch.exp(-dt * torch.exp(inj.A_log))
        inj.A_log.copy_(torch.log(-torch.log(1 - c * (1 - a)) / dt))
        inj.B.mul_(c)
        for w in projs:
            w.mul_(c)
    try:
        yield
    finally:
        with torch.no_grad():
            inj.A_log.copy_(saved[0])
            inj.B.copy_(saved[1])
            for w, v in zip(projs, saved[2:]):
                w.copy_(v)


@sdpa_kernel(SDPBackend.MATH)
def branch_shift(m, s, e):
    # E(s) = sum over the core branches j of J_j(x_(j-1)) z_(j-1): how much the branches move because their input
    # moves along theta(c); z_0 = u - s (the linear part), z_j = z_(j-1) + branch_j + J_j z_(j-1), at c = 1
    x = m.inject(s, e)
    o = m.offset["core"]
    out, z = torch.zeros_like(s), x - s
    for i, b in enumerate(m.core):
        ve = m.current_ve.get(o + i)
        for f in (lambda v: b.attn(b.n1(v), m.cos, m.sin, ve), lambda v: b.mlp(b.n2(v))):
            y, jy = torch.func.jvp(f, (x,), (z,))
            out = out + jy
            z = z + y + jy
            x = x + y
    return out


@torch.no_grad()
def loss_at(m, x, y, r, batch, dev, c=1.0, eps=0.0):
    # loss after r loops of s <- s + c (Phi(s) - s) + eps E(s)
    tot, n = 0.0, 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(dev), y[i:i + batch].to(dev)
        e = m.embed(xb)
        s = initial_state(m, e)
        for _ in range(r):
            nxt = s + c * (m.step(s, e) - s)
            if eps:
                nxt = nxt + eps * branch_shift(m, s, e)
            s = nxt
        tot += float(F.cross_entropy(m.readout(s).double().flatten(0, 1), yb.flatten()))
        n += 1
    return tot / n


def derivatives(m, x, y, r, batch, dev, h, direct=False):
    psi = {c: loss_at(m, x, y, r, batch, dev, c=c) for c in (1 - h, 1.0, 1 + h)}
    path = {}
    for c in (1 - h, 1 + h):
        with scaled_weights(m, c):
            path[c] = loss_at(m, x, y, r, batch, dev)
    d_psi = (psi[1 + h] - psi[1 - h]) / (2 * h)
    out = {"h": h, "psi": [psi[1 - h], psi[1.0], psi[1 + h]], "path": [path[1 - h], path[1 + h]],
           "dpsi": d_psi, "G": (path[1 + h] - path[1 - h]) / (2 * h),
           "psi2": (psi[1 + h] - 2 * psi[1.0] + psi[1 - h]) / h ** 2}
    out["N"] = out["G"] - d_psi
    if direct:
        out["N_direct"] = (loss_at(m, x, y, r, batch, dev, eps=h) - loss_at(m, x, y, r, batch, dev, eps=-h)) / (2 * h)
    return out


def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    x, y = offset_sequences(a.data, a.seqs, a.T, a.stride)
    return {"r": r, "seqs": a.seqs, "T": a.T, "steps": [derivatives(m, x, y, r, a.batch, dev, h) for h in a.h]}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1_000_000, help="distance in tokens between the starts of the sequences")
    ap.add_argument("--h", type=float, nargs="+", default=[0.05], help="steps of the central differences in c")
    a = ap.parse_args()
    for_each_run(a, "update_gauge.json", lambda m, cfg, dev: measure(m, cfg, dev, a),
                 keep=lambda cfg, name: "_d1792" not in name)


if __name__ == "__main__":
    main()
