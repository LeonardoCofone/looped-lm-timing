import argparse
import json
import os

import torch
import torch.nn.functional as F

from loopedlm.checkpoint import load_model, run_config
from measure.common import device, initial_state
from measure.time_scale import offset_sequences

MIX = [0.05, 0.1, 0.2, 0.3, 0.5]


def new():
    return {"la": 0.0, "lq": 0.0, "kl": 0.0, "mix": [0.0] * len(MIX), "n": 0}


def add(acc, la, lq, y):
    # la, lq: logits of the reference prediction and of the other one
    pa, pq, yf = F.log_softmax(la, -1), F.log_softmax(lq, -1), y.flatten()
    acc["la"] += float(F.nll_loss(pa.flatten(0, 1), yf))
    acc["lq"] += float(F.nll_loss(pq.flatten(0, 1), yf))
    acc["kl"] += float((pa.exp() * (pa - pq)).sum(-1).mean())
    for i, w in enumerate(MIX):
        acc["mix"][i] += float(F.cross_entropy(((1 - w) * la + w * lq).flatten(0, 1), yf))
    acc["n"] += 1


def finish(acc):
    # R = (l(q) - l(p)) / KL(p || q); G = gain of the best mix of the logits over the reference
    n = acc["n"]
    la, lq, kl, mix = acc["la"] / n, acc["lq"] / n, acc["kl"] / n, [v / n for v in acc["mix"]]
    return {"la": la, "lq": lq, "kl": kl, "R": (lq - la) / kl, "G": la - min(mix), "mix": mix}


@torch.no_grad()
def logits(m, xb, loops):
    e = m.embed(xb)
    s = initial_state(m, e)
    out = {}
    for k in range(1, max(loops) + 1):
        s = m.step(s, e)
        if k in loops:
            out[k] = m.readout(s).float()
    return out


def same_model(m, r, x, y, batch, dev):
    # the prediction at r against those at r + j of the same model
    js = [1, 2, 4, r, -2]
    acc = {j: new() for j in js}
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(dev), y[i:i + batch].to(dev)
        L = logits(m, xb, {r + j for j in js} | {r})
        for j in js:
            add(acc[j], L[r], L[r + j], yb)
    return {str(j): finish(a) for j, a in acc.items()}


def two_models(ma, ra, mb, rb, x, y, batch, dev):
    acc = new()
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch].to(dev), y[i:i + batch].to(dev)
        add(acc, logits(ma, xb, {ra})[ra], logits(mb, xb, {rb})[rb], yb)
    return finish(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/ensembles.json")
    ap.add_argument("--self", nargs="*", default=[], help="runs compared with their own later loops")
    ap.add_argument("--pairs", nargs="*", default=[], help="pairs of runs written as a|b")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1_000_000)
    a = ap.parse_args()
    dev = device()
    x, y = offset_sequences(a.data, a.seqs, a.T, a.stride)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    get = lambda name: load_model(os.path.join(a.runs, name), dev)[0]
    rec = lambda name: run_config(os.path.join(a.runs, name))["mean_rec"]
    for name in a.self:
        res["self_" + name] = same_model(get(name), rec(name), x, y, a.batch, dev)
        json.dump(res, open(a.out, "w"))
        print("self", name, flush=True)
    for pair in a.pairs:
        na, nb = pair.split("|")
        res[pair] = two_models(get(na), rec(na), get(nb), rec(nb), x, y, a.batch, dev)
        json.dump(res, open(a.out, "w"))
        print(pair, flush=True)


if __name__ == "__main__":
    main()
