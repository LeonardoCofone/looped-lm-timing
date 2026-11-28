import math
import os
import re
import statistics as st

import numpy as np

from analysis.common import load

GRID = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
SCALED = ["fixed_r8_b8_s0", "fixed_r8_b8_s1", "fixed_r8_b8_s2", "fixed_r12_b12_s0", "fixed_r16_b16_s0",
          "fixed_r24_b24_s0", "fixed_r8_b8_dec0.9_s0"]
AVERAGED = ["poisson_r8_b32_s0", "poisson_r8_b4_s0", "fixed_r8_b8_all_s0"]
SEEN = "fixed_r8_b8_s0"
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models")


def poisson(lam, K):
    p = [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(K + 1)]
    p[1] += p[0]
    p[0] = 0
    return [x / sum(p) for x in p]


def psi(name):
    v = load(name, "time_scale.json")
    tab, r = v["table"], v["r"]
    if name in AVERAGED:
        K = len(tab["1.0"])
        p = [0] + [1 / 8] * 8 + [0] * (K - 8) if "_all" in name else poisson(8, K)
        return [sum(p[j] * tab[str(c)][j - 1] for j in range(1, K + 1) if p[j] > 0) for c in GRID]
    return [tab[str(c)][r - 1] for c in GRID]


def c_star(ps):
    i = int(np.argmin(ps))
    if i in (0, len(ps) - 1):
        return GRID[i]
    a = np.polyfit(GRID[i - 1:i + 2], ps[i - 1:i + 2], 2)
    return float(-a[1] / (2 * a[0])) if a[0] > 0 else GRID[i]


def asymmetry(loss, k):
    slope = (loss[k + 1] - loss[k - 1]) / 2
    curv = (loss[k + 1] + loss[k - 1] - 2 * loss[k]) / 2
    return abs(slope) / curv if curv > 0 else float("inf")


def scale_test():
    print("Update scaled by c, loss at the trained depth (16 sequences of 1024 tokens)")
    ok = bad = 0
    rise = {}
    for name in SCALED + AVERAGED:
        ps = psi(name)
        cs, l1 = c_star(ps), ps[GRID.index(1.0)]
        gap = l1 - min(ps)
        mono = bool((np.diff(ps) > 0).all() or (np.diff(ps) < 0).all())
        rise[name] = max(ps[0], ps[-1]) - l1
        print("  %-24s c* %.3f  Psi(1) %.3f  above grid minimum %.4f  rise at the ends %.3f%s"
              % (name, cs, l1, gap, rise[name], "" if name in SCALED else "  (averaged, descriptive)"))
        if name in SCALED:
            ok += 0.9 <= cs <= 1.1 and gap <= 0.001
            bad += mono or abs(cs - 1) > 0.2
    n = len(SCALED)
    print("  c* in [0.9, 1.1] with Psi(1) within 0.001 of the minimum: %d of %d; monotone or |c*-1| > 0.2: %d" % (ok, n, bad))
    print("  verdict:", "YES" if ok >= 0.8 * n else "NO" if bad > 0.5 * n else "GREY")
    print("  rise at the ends: fixed %.3f to %.3f, averaged at most %.3f" % (
        min(rise[k] for k in SCALED), max(rise[k] for k in SCALED), max(rise[k] for k in AVERAGED)))
    print("  outside the population (descriptive):", ", ".join("%s c* %.3f" % (n_, c_star(psi(n_))) for n_ in (
        "fixed_r2_b2_s0", "fixed_r4_b4_s0", "fixed_r8_b4_s0", "fixed_r12_b8_s0")))


def fixed_runs():
    out = []
    for name in sorted(os.listdir(MODELS)):
        m = re.match(r"fixed_r(\d+)_b(\d+)(.*)_s(\d+)$", name)
        if m and os.path.exists(os.path.join(MODELS, name, "final_depth_stats.json")):
            out.append((name, int(m[1]), int(m[2]), m[3]))
    return out


def symmetry_test():
    print("Slope over curvature of the loss at r, end-of-training loss curves")
    main, ctrl, other = [], [], []
    for name, r, b, tag in fixed_runs():
        if "_all" in tag or name == SEEN:
            continue
        loss = {x["k"]: x["loss"] for x in load(name, "final_depth_stats.json")}
        if r + 2 not in loss:
            continue
        a = asymmetry(loss, r)
        if b < r or "_us" in tag or "dfz" in tag or r < 4:
            other.append((name, a))
            continue
        main.append((name, a))
        ctrl += [asymmetry(loss, r - 2) if r >= 3 else float("inf"), asymmetry(loss, r + 2)]
    a = [v for _, v in main]
    frac = sum(v <= 0.25 for v in a) / len(a)
    ma, mc = st.median(a), st.median(ctrl)
    print("  A_r <= 0.25 in %d of %d; median A_r %.3f, median at r-2 and r+2 %.2f" % (
        sum(v <= 0.25 for v in a), len(a), ma, mc))
    print("  above 0.25:", ", ".join("%s %.2f" % (n_, v) for n_, v in main if v > 0.25))
    print("  verdict:", "YES" if frac >= 0.8 and mc / ma >= 3 else "NO" if frac < 0.5 or ma >= mc / 1.5 else "GREY")
    print("  separate (descriptive):", ", ".join("%s %.2f" % (n_, v) for n_, v in other))
    paper = []
    for name, r, b, tag in fixed_runs():
        if b < r or r < 4 or re.search(r"all|_us|D100M|d1792", tag):
            continue
        loss = {x["k"]: x["loss"] for x in load(name, "curves.json")["per_k"]}
        if r + 2 in loss:
            paper.append(asymmetry(loss, r))
    print("  on the 32 sequences of curves.json: A_r <= 0.25 in %d of %d, median %.2f (descriptive)" % (
        sum(v <= 0.25 for v in paper), len(paper), st.median(paper)))


def main():
    scale_test()
    print()
    symmetry_test()


if __name__ == "__main__":
    main()
