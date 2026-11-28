"""The registered tests on the predictions and on the scale of the update besides the two of tests_time_scale.py,
and T-GAUGE (the scale of the update against the same scale on the weights).
Thresholds as registered before the data; verdicts YES / NO / GREY, and for the tests on GPTs GENERIC / LOOP-SPECIFIC.
Inputs: results/models/<model>/{final_depth_stats,scaled_update,predictions,update_gauge}.json and results/november/*.json
(short keys such as r8s0 are listed in results/november/keys.json)."""
import json
import os
import re
import statistics as st

import numpy as np
from scipy.stats import spearmanr

from analysis.common import load

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS = os.path.join(ROOT, "models")
NOV = os.path.join(ROOT, "november")
KEYS = json.load(open(os.path.join(NOV, "keys.json")))
POP = ["r8s0", "r8s1", "r8s2", "r12", "r16", "r24", "dec09"]
GRID = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]


def nov(name):
    return json.load(open(os.path.join(NOV, name)))


def pred(key):
    return load(KEYS[key], "predictions.json")


def fixed_base(fname):
    out = []
    for name in sorted(os.listdir(MODELS)):
        m = re.match(r"fixed_r(\d+)_b(\d+)_s(\d)$", name)
        if m and int(m[1]) == int(m[2]) and int(m[1]) >= 4 and os.path.exists(os.path.join(MODELS, name, fname)):
            out.append((name, int(m[1]), int(m[3])))
    return out


def c_star(cs, ps):
    i = int(np.argmin(ps))
    if i in (0, len(cs) - 1):
        return cs[i]
    a = np.polyfit(cs[i - 1:i + 2], ps[i - 1:i + 2], 2)
    return float(-a[1] / (2 * a[0])) if a[0] > 0 else cs[i]


def frac(v, a, b):
    return sum(a <= x <= b for x in v) / max(len(v), 1)


def t_curv():
    print("T-CURV: curvature of the loss at r against the squared step (fixed b = r, r >= 4)")
    rows = []
    for name, r, s in fixed_base("final_depth_stats.json"):
        S = {x["k"]: x for x in load(name, "final_depth_stats.json")}
        C = (S[r + 1]["loss"] + S[r - 1]["loss"] - 2 * S[r]["loss"]) / 2
        n2 = (S[r]["step_norm"] ** 2 + S[r + 1]["step_norm"] ** 2) / 2
        rows.append((r, C, n2))
    a = np.array([(x[1], x[2]) for x in rows])
    slope = np.polyfit(np.log(a[:, 1]), np.log(a[:, 0]), 1)[0]
    rs = sorted({x[0] for x in rows})
    kap = [np.mean([x[1] / x[2] for x in rows if x[0] == q]) for q in rs]
    cv = np.std(kap) / np.mean(kap)
    print("  %d models; slope of log C_r on log n_r^2 %.3f; CV of kappa across r %.3f; kappa %.1e to %.1e" % (
        len(rows), slope, cv, min(kap), max(kap)))
    print("  verdict:", "YES" if 0.8 <= slope <= 1.2 and cv <= 0.35 else "NO" if not 0.6 <= slope <= 1.4 else "GREY")


def t_flow():
    print("T-FLOW: G(r) = loss with the update scaled by 1/8 for 8r loops minus the loss at r")
    out = []
    for name, r, s in fixed_base("scaled_update.json"):
        E = load(name, "scaled_update.json")
        v = {k: E["conv"][k]["loss"] for k in E["conv"] if k.startswith("rp")}
        tgt = "rp%d" % (8 * r) if "rp%d" % (8 * r) in v else sorted(v, key=lambda k: int(k[2:]))[-1]
        o = v[tgt]
        F = o["1.0"] if "1.0" in o else (o["final"] if "final" in o else o[list(o)[-1]])
        out.append((r, s, F - E["L_r"]))
    rs = sorted({o[0] for o in out})
    G = [np.mean([o[2] for o in out if o[0] == q]) for q in rs]
    sp = spearmanr(rs, G).correlation
    g8 = np.mean([o[2] for o in out if o[0] == 8 and o[1] != 0])
    gh = np.mean([o[2] for o in out if o[0] >= 16])
    print("  mean G by r:", " ".join("r%d %.3f" % (q, g) for q, g in zip(rs, G)))
    print("  Spearman(r, G) %.3f; G(r >= 16) / G(8) %.3f" % (sp, gh / g8))
    print("  verdict:", "YES" if sp <= -0.8 and gh <= 0.25 * g8 else "NO" if sp > -0.5 or gh >= 0.75 * g8 else "GREY")


def t_kl():
    print("T-KL: rise of the loss on each side of r over the KL between consecutive predictions")
    ok = bad = 0
    for k in POP:
        m = pred(k)
        r, l, f, b = m["r"], m["loss"], m["kl_next"], m["kl_prev"]
        rp, rm = (l[r + 1] - l[r]) / f[r], (l[r - 1] - l[r]) / b[r]
        print("  %-24s R+ %.2f  R- %.2f" % (KEYS[k], rp, rm))
        ok += 0.75 <= rp <= 1.25 and 0.75 <= rm <= 1.25
        bad += not (0.5 <= rp <= 1.5 and 0.5 <= rm <= 1.5)
    n = len(POP)
    print("  both in [0.75, 1.25]: %d of %d; one outside [0.5, 1.5]: %d" % (ok, n, bad))
    print("  verdict:", "YES" if ok >= 0.8 * n else "NO" if bad > 0.5 * n else "GREY")


def t_suff():
    print("T-SUFF: after r, rise of the loss over KL(p_r || p_r+j), j in {2, 3, 4, r/2, r, 2r}")
    tot = ok = 0
    lo, hi = 9.0, -9.0
    for k in POP:
        m = pred(k)
        r, l, K = m["r"], m["loss"], m["kl_from_r"]
        for j in sorted({2, 3, 4, max(r // 2, 1), r, 2 * r}):
            if r + j >= len(l):
                continue
            R = (l[r + j] - l[r]) / K[r + j]
            tot += 1
            ok += 0.75 <= R <= 1.25
            lo, hi = min(lo, R), max(hi, R)
    print("  in [0.75, 1.25]: %d of %d (ratios %.2f to %.2f)" % (ok, tot, lo, hi))
    print("  verdict:", "YES" if ok >= 0.8 * tot else "NO" if ok < 0.5 * tot else "GREY")
    for k in ("r2", "r4", "r8b4", "r12b8"):
        m = pred(k)
        r, l, K = m["r"], m["loss"], m["kl_from_r"]
        R = [(l[r + j] - l[r]) / K[r + j] for j in sorted({2, 3, 4, max(r // 2, 1), r, 2 * r}) if r + j < len(l)]
        print("  outside the population, %-18s %.2f to %.2f (descriptive)" % (KEYS[k], min(R), max(R)))


def t_ens():
    print("T-ENS: same ratio and ensemble gain G for later loops of one model and for other seeds")
    res = nov("ensembles.json")
    sR, sG, seed, seedG = [], [], [], []
    for k, v in res.items():
        if k.startswith("self_"):
            for j, a in v.items():
                if int(j) > 0:
                    sR.append(a["R"])
                    sG.append(a["G"])
        elif k.startswith("r8s") and k.split("|")[1].startswith("r8s"):
            seed.append(res[k]["R"])
            seedG.append(res[k]["G"])
    mdev, mG, mGs = st.median(abs(x - 1) for x in sR), st.median(sG), st.median(seedG)
    low, high = sum(x <= 0.6 for x in seed), sum(x >= 0.75 for x in seed)
    print("  same model: median |R - 1| %.3f, median G %.4f; other seed: R <= 0.6 in %d of %d, median G %.4f" % (
        mdev, mG, low, len(seed), mGs))
    print("  verdict:", "YES" if mdev <= 0.15 and low >= 5 and mG <= 0.002 and mGs >= 0.01 else "NO" if high >= 4 else "GREY")


def martingale_pairs(m, jmax, only_j=None, skip_j=None, MIN=0.005):
    l, KL = m["loss"], m["kl_pairs"]
    Rs, As = [], []
    for j in range(2, jmax + 1):
        if (only_j and j != only_j) or (skip_j and j == skip_j):
            continue
        for k in range(1, j):
            if KL[j][k] < MIN:
                continue
            Rs.append((l[k] - l[j]) / KL[j][k])
            if j - k >= 2:
                As.append(KL[j][k] / sum(KL[i + 1][i] for i in range(k, j)))
    return Rs, As


def t_martingale():
    print("T-MARTINGALE: are the predictions of the averaged models a martingale over loops 1 to 8?")
    ER, EA, fR = [], [], []
    for k in ("pb32", "pb4", "all"):
        R, A = martingale_pairs(pred(k), 8)
        ER += R
        EA += A
    for k in ("r8s0", "r8s1", "r8s2", "r16"):
        m = pred(k)
        fR += martingale_pairs(m, min(16, m["r"]), skip_j=m["r"])[0]
    a, b, c = frac(ER, .85, 1.15), frac(EA, .85, 1.15), frac(fR, .85, 1.15)
    print("  averaged: |R - 1| <= 0.15 in %.2f of %d pairs, A in [0.85, 1.15] in %.2f of %d (median A %.2f); "
          "fixed, j != r: %.2f" % (a, len(ER), b, len(EA), st.median(EA), c))
    print("  verdict:", "YES" if a >= .8 and b >= .8 and c <= .5 else "NO" if a < .6 or b < .5 else "GREY")


def t_span_gpt():
    res = nov("span.json")
    print("T-SPAN: linear part of the change of the features after r (ridge from h_r), same model against other seeds")
    rmap = {"r8s0": 8, "r8s1": 8, "r8s2": 8, "r12": 12, "r16": 16}
    sel, allp, cross = [], [], []
    for k, r in rmap.items():
        for j, a in res.get("self_" + k, {}).items():
            if int(j) in (-2, 2, 4, r):
                sel.append(a)
            allp.append(a)
    for k in (x for x in res if "|" in x):
        (cross if k.startswith("r8s") else []).append(res[k])
        allp.append(res[k])
    f1 = frac([abs(a["res_lin"]) / a["kl"] for a in allp], 0, 0.1)
    ms, mc = st.median(a["R2"] for a in sel), st.median(a["R2"] for a in cross)
    f2 = frac([abs(a["res_xi"]) / a["kl"] for a in sel], 0, 0.15)
    c3 = sum(abs(a["res_xi"]) >= 0.5 * a["kl"] for a in cross)
    print("  |res_lin| <= 0.1 KL: %.2f of %d; median R2 same model %.3f, other seed %.3f; |res_xi| <= 0.15 KL %.2f; "
          "other seed |res_xi| >= 0.5 KL: %d of %d" % (f1, len(allp), ms, mc, f2, c3, len(cross)))
    print("  verdict:", "YES" if f1 >= .8 and ms >= mc + .1 and f2 >= .8 and c3 >= 5 else "NO" if ms <= mc + .02 else "GREY")
    print("T-GPT: the same ratio in GPTs of 12 and 16 layers, skipping the layers from an inner one to the last two")
    gp = [a for k in ("g12", "g16") for j, a in res.get("self_" + k, {}).items()
          if a["kl"] >= 1e-9 and j.startswith("skip") and 0.02 <= a["kl"] <= 1.0]
    fr = frac([(a["lb"] - a["la"]) / a["kl"] for a in gp], 0.75, 1.25)
    gm = st.median(a["G"] for a in gp)
    print("  %d readable pairs: R in [0.75, 1.25] %.2f; median G %.4f" % (len(gp), fr, gm))
    print("  verdict:", "GENERIC" if fr >= .8 and gm <= .002 else "LOOP-SPECIFIC" if fr < .5 else "GREY")


def t_tele():
    res = nov("telescoping.json")
    print("T-TELE: stationarity of each branch (GPT) and of each loop step (loops), |sum| / sum |.| over tokens")
    crossr = [res[k]["res"]["ratio"] for k in res if k.startswith("cross")]
    g_s, ok, spec = [], [], []
    for k in ("g12", "g16"):
        S = res[k]["stat"]
        n = sum(x.startswith("s") for x in S)
        g_s += [S["s%d" % i]["ratio"] for i in range(n)]
        P = [res[k]["Psi"][str(c)] for c in GRID]
        cs = c_star(GRID, P)
        ok.append(0.9 <= cs <= 1.1 and P[4] - min(P) <= 0.001)
        spec.append(abs(cs - 1) > 0.2 or bool((np.diff(P) > 0).all() or (np.diff(P) < 0).all()))
        print("  GPT %s: c* of the scaled branches %.3f, Psi(1) above the minimum %.4f, Psi(1.3) - Psi(1) %.3f" % (
            k[1:], cs, P[4] - min(P), P[-1] - P[4]))
    pop = []
    for k in ("r8s0", "r8s1", "r8s2", "r12", "r16", "dec09"):
        m = res[k]
        pop += [m["stat"]["Qp%d" % i]["ratio"] for i in range(1, m["r"])]
    ms, mp, mc = st.median(g_s), st.median(pop), st.median(crossr)
    print("  median ratio: GPT %.4f, loops %.4f, between models %.3f" % (ms, mp, mc))
    print("  verdict:", "YES" if ms <= mc / 5 and mp <= mc / 5 else "NO" if ms >= mc / 2 or mp >= mc / 2 else "GREY")
    print("T-SCALE-GPT: is the loss stationary in the scale of the update also in GPTs (all branches times c)?")
    print("  verdict:", "GENERIC" if all(ok) else "LOOP-SPECIFIC" if all(spec) else "GREY")


def t_external():
    print("T-EXTERNAL: Ouro-1.4B and 2.6B (r = 4, loss at every step), mean loss over loops 1 to 4 with the update scaled by c")
    cs, Rs, Gs = [], [], []
    for f in ("ouro_1.4b.json", "ouro_2.6b.json"):
        m = nov(f)
        t = m["table"]
        c1 = c_star(GRID, [float(np.mean(t[str(c)][:4])) for c in GRID])
        cs.append(c1)
        for k in (1, 2, 3, 5, 6, 8):
            a = m["pairs"]["4|%d" % k]
            Gs.append(a["G"])
            if a["kl"] >= 0.005:
                Rs.append(a["R"])
        print("  %s: c* %.3f (largest gap to the official forward %.1e)" % (m["model"], c1, max(m["check_vs_forward"])))
    fr, gm = frac(Rs, 0.75, 1.25), st.median(Gs)
    gc = min(a["G"] for a in nov("ouro_cross.json").values())
    print("  R(4, k) in [0.75, 1.25] %.2f of %d; median G(4, k) %.4f; G between the two models at least %.3f" % (
        fr, len(Rs), gm, gc))
    yes = all(0.9 <= c <= 1.1 for c in cs) and fr >= .8 and gm <= .002 and gc >= .01
    no = all(not 0.8 <= c <= 1.2 for c in cs) or fr < .5
    print("  verdict:", "YES" if yes else "NO" if no else "GREY")


def t_flow_gpt():
    res = nov("gpt_substeps.json")
    print("T-FLOW-GPT: every branch of a GPT split into 8 substeps, loss change")
    G = []
    for k in ("g12", "g16"):
        l = res[k]["loss"]
        G.append(l["8"] - l["1"])
        print("  GPT %s: %.3f (substeps 1, 2, 4, 8: %s)" % (k[1:], G[-1], " ".join("%.3f" % l[n] for n in ("1", "2", "4", "8"))))
    print("  verdict:", "GENERIC" if min(G) >= .0125 else "LOOP-SPECIFIC" if max(G) <= .005 else "GREY")


def t_shorter():
    res = nov("shorter_longer.json")
    print("T-SHORTER: r/2 loops with the update doubled (Gd) and 2r loops with it halved (Gu), against L_r "
          "(16 batches of 4 sequences of 256 tokens)")
    A = lambda k, q: np.array([x[q] for x in res[k]])
    for k in res:
        print("  %-22s L_r %.3f  Gd %+.3f  Gu %+.4f" % (KEYS[k], A(k, "L").mean(), A(k, "Gd").mean(), A(k, "Gu").mean()))
    nb = nn = 0
    for a, c in (("r24", "r12"), ("r16", "r8s0"), ("r8s0", "r4"), ("r4", "r2")):
        W = A(a, "L") + A(a, "Gd") - A(c, "L")
        se = W.std(ddof=1) / np.sqrt(len(W))
        nb += W.mean() <= -0.015 and abs(W.mean()) > 2 * se
        nn += W.mean() >= 0 or abs(W.mean()) <= 2 * se
        print("  shortened %s against trained %s: %+.2f" % (KEYS[a], KEYS[c], W.mean()))
    print("  A, the shortened model beats the one trained at r/2: verdict", "YES" if nb >= 2 else "NO" if nn >= 3 else "GREY")
    g1, g2 = A("r24", "Gd").mean(), A("r16", "Gd").mean()
    print("  B, Gd <= 0.03 at r = 24 and 16: %.2f, %.2f; verdict" % (g1, g2),
          "YES" if g1 <= .03 and g2 <= .03 else "NO" if g1 >= .1 or g2 >= .1 else "GREY")
    rat = [A(k, "Gd").mean() / A(k, "Gu").mean() for k in ("r16", "r12", "r8s0", "r8s1", "r8s2", "r4")]
    n = sum(2 <= x <= 6 for x in rat)
    print("  C, Gd / Gu in [2, 6]: %d of 6 (ratios %.0f to %.0f); verdict" % (n, min(rat), max(rat)),
          "YES" if n >= 5 else "NO" if n < 3 else "GREY")
    cost = []
    for name, r, s in fixed_base("scaled_update.json"):
        E = load(name, "scaled_update.json")
        c = [E["conv"]["rp%d" % (k * r)]["loss"]["1.0"] - E["L_r"] for k in (2, 4, 8)]
        cost.append((c, c[0] <= c[1] <= c[2]))
    print("  lengthening, r' = 2r, 4r, 8r loops scaled by r/r' (exploratory): rise above L_r monotone in %d of %d models, "
          "at most %.3f at 8r" % (sum(m for _, m in cost), len(cost), max(c[2] for c, _ in cost)))


def t_negative():
    res = nov("jacobian_negative.json")
    print("T-NEGATIVE: isolated negative real Ritz value of the one-loop Jacobian at loop r/2 in at least 2 of 3 sequences")
    fixed, averaged = ["r8s0", "r8s1", "r8s2", "r12", "r16", "r24"], ["pb32", "pb4", "all"]
    rr = {"r12": 12, "r16": 16, "r24": 24}

    def has(k):
        r = rr.get(k, 8)
        iso = [res[k]["s%d_k%d" % (q, r // 2)]["iso"] for q in range(3)]
        print("  %-22s %s" % (KEYS[k], "; ".join(" ".join("%.2f" % x for x in v) or "none" for v in iso)))
        return sum(len(v) >= 1 for v in iso) >= 2
    a, b = sum(has(k) for k in fixed), sum(has(k) for k in averaged)
    print("  fixed %d of 6, averaged %d of 3; verdict" % (a, b), "YES" if a >= 4 and b <= 1 else "NO" if a <= 2 else "GREY")


def t_gauge():
    print("T-GAUGE: G = d/dc of the loss at r along the weight path theta(c) equals Psi'(1) + N; shifts of c* in units of "
          "Psi''(1), delta_G = G / Psi'' and delta_N = N / Psi'' (h = 0.05)")
    ok = bad = n = neg = 0
    gap, gmax = [], 0.0
    for k in POP + ["pb4", "pb32", "all"]:
        S = {x["h"]: x for x in load(KEYS[k], "update_gauge.json")["steps"]}
        s, t = S[0.05], S[0.02]
        dG, dN = s["G"] / s["psi2"], s["N"] / s["psi2"]
        if k in POP:
            n += 1
            ok += abs(dG) <= 0.03 and abs(dN) <= 0.03
            bad += abs(dN) >= 0.10
            neg += s["G"] < 0
            gmax = max(gmax, abs(s["G"]))
            tab = load(KEYS[k], "time_scale.json")["table"]
            r = load(KEYS[k], "update_gauge.json")["r"]
            cs = sorted(tab, key=float)
            gap.append(abs(1 - s["dpsi"] / s["psi2"] - c_star([float(c) for c in cs], [tab[c][r - 1] for c in cs])))
            print("  %-22s Psi' %+.4f G %+.4f N %+.4f Psi'' %.3f delta_G %+.4f delta_N %+.4f (h = 0.02: %+.4f %+.4f)" % (
                KEYS[k], s["dpsi"], s["G"], s["N"], s["psi2"], dG, dN, t["G"] / t["psi2"], t["N"] / t["psi2"]))
        else:
            print("  %-22s Psi' %+.4f G %+.4f N %+.4f Psi'' %.3f (descriptive: l_r is not its training objective)" % (
                KEYS[k], s["dpsi"], s["G"], s["N"], s["psi2"]))
    print("  |delta_G| and |delta_N| <= 0.03 in %d of %d, |delta_N| >= 0.10 in %d; verdict" % (ok, n, bad),
          "YES" if ok >= 6 else "NO" if bad >= 4 else "GREY")
    print("  descriptive: G < 0 (expected from the weight decay) in %d of %d; |G| at most %.3f nats; 1 - Psi'/Psi'' "
          "against the parabolic c* of T-SCALE: at most %.4f apart" % (neg, n, gmax, max(gap)))


def main():
    for t in (t_curv, t_flow, t_kl, t_suff, t_ens, t_martingale, t_span_gpt, t_tele, t_external, t_flow_gpt,
              t_shorter, t_negative, t_gauge):
        t()
        print()


if __name__ == "__main__":
    main()
