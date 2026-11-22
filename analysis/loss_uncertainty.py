import glob
import os

import numpy as np

from analysis.common import MODELS, depth, load

RECIPES = [("fixed, through every loop", ["fixed_r8_b8_s%d" % s for s in range(3)]),
           ("loss at every loop", ["fixed_r8_b8_all_s%d" % s for s in range(2)]),
           ("Poisson, through every loop", ["poisson_r8_b32_s%d" % s for s in range(2)]),
           ("Poisson, truncated to 4", ["poisson_r8_b4_s%d" % s for s in range(3)])]


def per_sequence(name, k):
    z = np.load(os.path.join(MODELS, name, "loss_tokens.npz"))
    ks, T = z["ks"].tolist(), int(z["T"])
    return z["loss"][ks.index(k)].reshape(-1, T).mean(1)


def bootstrap_se(x, rng, n=2000):
    b = rng.integers(0, len(x), (n, len(x)))
    return x[b].mean(1).std()


def main():
    print("== Loss at r = 8 and at loop 16 by recipe (descriptive, Section 7)")
    for label, runs in RECIPES:
        L = [load(d, "curves.json")["per_k"] for d in runs]
        print("  %-28s L_8 %s | L_16 %s" % (label, " ".join("%.3f" % p[8]["loss"] for p in L),
                                             " ".join("%.3f" % p[16]["loss"] for p in L)))

    print("== Standard errors over sequences (bootstrap, Appendix A)")
    rng = np.random.default_rng(0)
    se_l, se_d = [], []
    for f in sorted(glob.glob(os.path.join(MODELS, "*", "loss_tokens.npz"))):
        d = os.path.basename(os.path.dirname(f))
        r = depth(d)
        a, b = per_sequence(d, r), per_sequence(d, r + 1)
        se_l.append(bootstrap_se(a, rng))
        se_d.append(bootstrap_se(b - a, rng))
    print("  %d models, %d sequences of 1024 tokens" % (len(se_l), len(a)))
    print("  L_r: %.4f to %.4f | L_(r+1) - L_r: %.4f to %.4f" % (min(se_l), max(se_l), min(se_d), max(se_d)))
    se_p = [bootstrap_se(per_sequence(f, 8) - per_sequence(p, 8), rng)
            for f in RECIPES[0][1] for p in RECIPES[3][1]]
    print("  L_8 fixed - L_8 Poisson truncated to 4: %.4f to %.4f" % (min(se_p), max(se_p)))


if __name__ == "__main__":
    main()
