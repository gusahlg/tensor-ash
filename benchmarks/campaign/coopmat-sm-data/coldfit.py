#!/usr/bin/env python3
"""Refit the coopmat tile rule on the cold-L2 sweeps (coopmat-sm.md §11).

    python3 coldfit.py                 # regret of candidate rules on cold-4060 / cold-3070 (+ warm checks)
    python3 coldfit.py table DIR GPU   # markdown table of one sweep with old/new picks
"""
import itertools, math, sys
import statistics as st
import analysis as A

MIB = A.MIB
TILES = ["128x64", "64x64", "64x64m", "128x128", "64x128"]


def feats(d, sms):
    b, m, n, k, a16 = d["b"], d["m"], d["n"], d["k"], d["a16"]
    rows, cols = m // 64, n // 64
    w = sms * 7
    ab, bb = m * k * (2 if a16 else 4), k * n * 2
    xp = math.ceil(rows / max(1.0, w / cols))
    mp = math.ceil(cols / max(1.0, w / rows))
    return dict(b=b, m=m, n=n, rows=rows, cols=cols, ab=ab, bb=bb, xp=xp, mp=mp,
                tr=(ab * mp + bb) / (bb * xp + ab), a16=a16)


def score(tiles, sms):
    per = -(-tiles // sms)
    return tiles / (sms * per) * min(1.0, per / 3.0)


def make_rule(t_r=0.75, are_max=8 * MIB, exempt_fit=True, rho=None, one_pass_mfast=False):
    """Parameterized version of routing.rs coopmat_tile (f32 accumulate)."""
    def rule(d, sms, l2):
        f = feats(d, sms)
        large = l2 >= 16 * MIB
        mfast = (not (exempt_fit and large and f["bb"] <= l2)) and f["ab"] * (f["mp"] - 1) <= are_max \
            and (f["tr"] < t_r or (one_pass_mfast and f["xp"] == 1 and f["rows"] >= 8))
        t64 = "64x64m" if mfast else "64x64"
        if f["m"] % 128:
            return t64
        r = rho(f, large, mfast) if rho else A_rho(f, large, mfast)
        s64 = score(f["b"] * f["rows"] * f["cols"], sms) * r
        s128 = score(f["b"] * (f["m"] // 128) * f["cols"], sms)
        return t64 if s64 >= s128 else "128x64"
    return rule


def new_rule(d, sms, l2):
    """Mirror of routing.rs coopmat_tile after the cold-L2 refit."""
    f = feats(d, sms)
    large = l2 >= 16 * MIB
    cheap_a = f["ab"] * (f["mp"] - 1) <= 8 * MIB or (large and 2 * f["ab"] <= l2)
    mfast = cheap_a and (f["tr"] < 0.75 or (f["xp"] == 1 and f["rows"] >= 8 and f["cols"] >= 32))
    t64 = "64x64m" if mfast else "64x64"
    if f["m"] % 128:
        return t64
    r = new_rho(f, large, mfast)
    s64 = score(f["b"] * f["rows"] * f["cols"], sms) * r
    return t64 if s64 >= score(f["b"] * (f["m"] // 128) * f["cols"], sms) else "128x64"


def new_rho(f, large, mfast):  # f16w small-L2 keyed on xp, not on the raster
    if large or f["a16"]:
        return A_rho(f, large, mfast)
    return 0.9 if f["xp"] > 1 else (1.0 if f["ab"] <= 4 * MIB else 0.85)


def A_rho(f, large, mfast):  # shipped ρ table
    if large:
        return 1.02 if f["a16"] else 0.92
    if f["a16"]:
        return 0.8 if (not mfast and f["xp"] > 1) else 1.02
    return 0.9 if (mfast or f["xp"] > 1) else (1.0 if f["ab"] <= 4 * MIB else 0.85)


def evaluate(name, rule, sets, verbose=False):
    out = []
    for d, g, l2 in sets:
        hw = A.GPUS[g]
        S = A.load_dir(d)
        reg, gain, worse = [], [], []
        for case, s in sorted(S.items()):
            ms = {t: s["cells"][t]["ms"] for t in TILES if t in s["cells"]}
            pick, old = rule(s, hw["sms"], l2), A.grokmerge(s)
            if pick not in ms or old not in ms:
                continue
            reg.append(ms[pick] / min(ms.values()) - 1)
            gain.append(ms[old] / ms[pick])
            if ms[pick] / ms[old] - 1 > 0.01:
                worse.append((case, pick, old, ms[pick] / ms[old] - 1))
        out.append((d, len(reg), st.mean(reg), max(reg), st.geometric_mean(gain), worse))
    print(f"--- {name}")
    for d, n, rm, rx, gm, worse in out:
        print(f"  {d:12s} n={n:2d} regret mean {rm*100:5.2f}% max {rx*100:5.1f}%  vs grokmerge x{gm:.3f}  slower>1%: {len(worse)}"
              + ("  " + ", ".join(f"{w[0]} {w[1]}<-{w[2]} {w[3]*100:+.1f}%" for w in worse) if verbose or len(worse) <= 4 else ""))
    return out


if __name__ == "__main__" and len(sys.argv) == 1:
    SETS = [("cold-4060", "4060", 24 * MIB), ("cold-3070", "3070", MIB),
            ("verify-4060", "4060", 24 * MIB), ("verify-3070", "3070", MIB)]
    evaluate("shipped (100df0d)", make_rule(), SETS)
    evaluate("drop large-L2 fit exemption", make_rule(exempt_fit=False), SETS)
    evaluate("small-L2 rule everywhere", make_rule(), [(d, g, MIB) for d, g, _ in SETS])
    evaluate("NEW: cold-fitted raster (routing.rs)", new_rule, SETS, verbose=True)


def md_table(d, gpu, l2):
    """Markdown table of one same-run sweep: TF/s per tile, picks, new/old."""
    S = A.load_dir(d)
    sms = A.GPUS[gpu]["sms"]
    old_rule = make_rule()
    nm = lambda t: t.replace("64x64m", "64x64 M-fast")
    print("| shape | 128x64 | 64x64 | 64x64 M-fast | 128x128 | 64x128 | grokmerge | 100df0d | new | new/grokmerge |")
    print("|---|---:|---:|---:|---:|---:|---|---|---|---:|")
    for case, s in sorted(S.items(), key=lambda kv: (kv[1]["a16"], kv[1]["m"], kv[1]["n"], kv[1]["k"])):
        c = s["cells"]
        ms = {t: c[t]["ms"] for t in TILES if t in c}
        best = min(ms, key=ms.get)
        fl = 2.0 * s["b"] * s["m"] * s["n"] * s["k"]
        cell = lambda t: (("**%.2f**" if t == best else "%.2f") % (fl / ms[t] * 1e-9)) if t in ms else "n/a"
        g, o, n = A.grokmerge(s), old_rule(s, sms, l2), new_rule(s, sms, l2)
        tag = "a16 " if s["a16"] else ""
        print(f"| {tag}{s['m']}x{s['n']}x{s['k']} | " + " | ".join(cell(t) for t in TILES)
              + f" | {nm(g)} | {nm(o)} | {nm(n)} | {ms[g] / ms[n]:.3f} |")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "table":
    md_table(sys.argv[2], sys.argv[3], (24 if sys.argv[3] == "4060" else 1) * MIB)
