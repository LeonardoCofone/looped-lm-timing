import os

import numpy as np
import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state, validation

SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]


def offset_sequences(data, n, T, stride=1_000_000):
    v = validation(data)
    seq = np.stack([np.asarray(v[i * stride: i * stride + T + 1]) for i in range(n)]).astype(np.int64)
    return torch.from_numpy(seq[:, :-1]), torch.from_numpy(seq[:, 1:])


@torch.no_grad()
def table(m, x, y, K, scales, batch, dev):
    out = {}
    for c in scales:
        tot, n = np.zeros(K), 0
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch].to(dev), y[i:i + batch].to(dev)
            e = m.embed(xb)
            s = initial_state(m, e)
            for k in range(K):
                s = s + c * (m.step(s, e) - s)
                tot[k] += float(F.cross_entropy(m.readout(s).float().flatten(0, 1), yb.flatten()))
            n += 1
        out[str(c)] = [float(v) if np.isfinite(v) else None for v in tot / n]
    return out


def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    x, y = offset_sequences(a.data, a.seqs, a.T, a.stride)
    K = max(32, 2 * r)
    return {"r": r, "seqs": a.seqs, "T": a.T, "table": table(m, x, y, K, SCALES, a.batch, dev)}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1_000_000, help="distance in tokens between the starts of the sequences")
    a = ap.parse_args()
    for_each_run(a, "time_scale.json", lambda m, cfg, dev: measure(m, cfg, dev, a),
                 keep=lambda cfg, name: "_d1792" not in name)


if __name__ == "__main__":
    main()
