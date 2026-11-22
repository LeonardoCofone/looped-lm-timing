import numpy as np
import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, sequences, state_generator


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    ks = [r, r + 1]
    x, y = sequences(a.data, n=a.seqs)
    rows = {k: [] for k in ks}
    for i in range(0, x.shape[0], a.chunk):
        xb, yb = x[i:i + a.chunk].to(dev), y[i:i + a.chunk].to(dev)
        S = m.trajectory(xb, max(ks), gen=state_generator(dev))[1]
        for k in ks:
            lg = m.readout(S[k]).float()
            rows[k].append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), yb.reshape(-1), reduction="none").cpu())
    loss = torch.stack([torch.cat(rows[k]) for k in ks]).numpy().astype(np.float16)
    T = int(x.shape[1])
    return {"ks": ks, "seqs": a.seqs, "T": T}, {"loss": loss, "ks": np.array(ks), "T": np.array(T)}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    for_each_run(a, "loss.json", lambda m, cfg, dev: measure(m, cfg, dev, a))


if __name__ == "__main__":
    main()
