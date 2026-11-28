import numpy as np
import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state
from measure.time_scale import offset_sequences

BINS = 15
PAIRS = 16


def kl(p, q):
    return float((p.exp() * (p - q)).sum(-1).mean())


@torch.no_grad()
def curves(m, x, y, K, r, batch, dev):
    # per loop k: loss, KL(p_k || p_k+1), KL(p_k || p_k-1), KL(p_r || p_k); for the first PAIRS loops also
    # KL(p_j || p_k) for every pair and the calibration error, both on predictions stored in float16
    loss, kl_next, kl_prev, kl_from_r = (np.zeros(K + 1) for _ in range(4))
    kl_pairs = np.zeros((PAIRS + 1, PAIRS + 1))
    count, conf, hit = (np.zeros((PAIRS + 1, BINS)) for _ in range(3))
    edges = torch.linspace(0, 1, BINS + 1, device=dev)
    batches = 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(dev), y[i:i + batch].to(dev)
        e = m.embed(xb)
        s = initial_state(m, e)
        states = []
        for _ in range(K):
            s = m.step(s, e)
            states.append(s)
        logp = lambda k: F.log_softmax(m.readout(states[k - 1]).float(), -1)
        at_r = logp(r)
        early = {}
        prev = None
        for k in range(1, K + 1):
            p = logp(k)
            loss[k] += float(F.nll_loss(p.flatten(0, 1), yb.flatten()))
            kl_from_r[k] += kl(at_r, p)
            if prev is not None:
                kl_next[k - 1] += kl(prev, p)
                kl_prev[k] += kl(p, prev)
            prev = p
            if k <= PAIRS:
                early[k] = p.half()
        for j in range(1, min(K, PAIRS) + 1):
            pj = early[j].float()
            top, guess = pj.exp().max(-1)
            b = (torch.bucketize(top.flatten(), edges) - 1).clamp(0, BINS - 1)
            right = (guess == yb).flatten().float()
            for q in range(BINS):
                w = b == q
                count[j, q] += float(w.sum())
                conf[j, q] += float(top.flatten()[w].sum())
                hit[j, q] += float(right[w].sum())
            for k in range(1, min(K, PAIRS) + 1):
                if k != j:
                    kl_pairs[j, k] += kl(pj, early[k].float())
        del states, early
        batches += 1
    ece = (np.abs(conf - hit).sum(1) / np.maximum(count.sum(1), 1)).tolist()
    out = lambda v: (v / batches).tolist()
    return {"loss": out(loss), "kl_next": out(kl_next), "kl_prev": out(kl_prev), "kl_from_r": out(kl_from_r),
            "kl_pairs": out(kl_pairs), "ece": ece}


def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    K = max(32, 2 * r + 4, 3 * r + 2)
    x, y = offset_sequences(a.data, a.seqs, a.T, a.stride)
    return {"r": r, "seqs": a.seqs, "T": a.T, **curves(m, x, y, K, r, a.batch, dev)}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1_000_000, help="distance in tokens between the starts of the sequences")
    a = ap.parse_args()
    for_each_run(a, "predictions.json", lambda m, cfg, dev: measure(m, cfg, dev, a),
                 keep=lambda cfg, name: "_d1792" not in name)


if __name__ == "__main__":
    main()
