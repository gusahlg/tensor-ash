#!/usr/bin/env python3
"""Reproduce the coopmat-sm.md numbers from the raw `ml_bench ab` CSVs.

    python3 analysis.py model        # §6: per-GPU fits, cross-GPU prediction error
    python3 analysis.py policies     # §6: tile-choice regret of each policy
    python3 analysis.py rule         # §7: shipped rule vs exp/grokmerge on the sweeps
    python3 analysis.py verify DIR GPU [--md]   # §8: same-run new-auto vs old pick

Needs numpy + scipy.  Run from this directory.
"""
import csv
import glob
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIB = 1 << 20
GPUS = {  # hardware inputs (S from VK_NV_shader_sm_builtins; L2/BW datasheet)
    "4060": dict(sms=24, l2=24 * MIB, bw=272e9, large_l2=True),
    "3070": dict(sms=46, l2=4 * MIB, bw=448e9, large_l2=False),
}
TILES = {"128x128": (128, 128, 2), "64x64": (64, 64, 7), "64x128": (64, 128, 3), "128x64": (128, 64, 4)}
ORDER = ["128x128", "64x64", "64x128", "128x64"]
KMAP = {
    "f16w_coopmat_aligned": "128x128", "f16w_a16_coopmat_aligned": "128x128",
    "f16w_coopmat_m64n64": "64x64", "f16w_a16_coopmat_m64n64": "64x64",
    "f16w_coopmat_m64n128": "64x128", "f16w_a16_coopmat_m64n128": "64x128",
    "f16w_coopmat_m128n64": "128x64", "f16w_a16_coopmat_m128n64": "128x64",
    "f16w_coopmat_m64n64_mfast": "64x64m", "f16w_a16_coopmat_m64n64_mfast": "64x64m",
    "auto": "auto",
}


# ---------------------------------------------------------------- loading
def load_dir(d):
    """{case: {b,m,n,k,a16,cells:{tile:{ms,tf,route}}, clk}} from abrun.sh CSVs."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        if p.endswith(".smi.csv"):
            continue
        hdr = None
        for line in open(p):
            if line.startswith("#") or line.startswith("ab protocol") or not line.strip():
                continue
            if line.startswith("case,"):
                hdr = line.strip().split(",")
                continue
            r = dict(zip(hdr, next(csv.reader([line.strip()]))))
            case = r["case"].replace("x4096x4096", "").replace("sq2048x2048x2048", "sq2048")
            s = out.setdefault(case, {"b": int(r["b"]), "m": int(r["m"]), "n": int(r["n"]), "k": int(r["k"]),
                                      "a16": r["case"].startswith("a"), "cells": {}})
            tile = KMAP.get(r["kernel"], r["kernel"])
            try:
                ms = float(r["gpu_ms"])
            except ValueError:
                continue
            s["cells"][tile] = {"ms": ms, "tf": float(r["tflops"]), "route": KMAP.get(r.get("route", ""), r.get("route"))}
    return out


def median_clock(d):
    vals = []
    for p in glob.glob(os.path.join(d, "*.smi.csv")):
        for line in open(p):
            f = [x.strip() for x in line.split(",")]
            try:
                clk, util = float(f[1].split()[0]), float(f[5].split()[0])
            except (ValueError, IndexError):
                continue
            if util >= 90:
                vals.append(clk)
    return st.median(vals) if vals else {"4060": 2535.0, "3070": 1815.0}.get(os.path.basename(d)[-4:])


# ---------------------------------------------------------------- rules
def grokmerge(d):
    """exp/grokmerge coopmat_tile (== routing.rs coopmat_tile_fallback)."""
    b, m, n = d["b"], d["m"], d["n"]
    if m <= 512:
        if m == 512 and n > 4 * m:
            return "128x64"
        if m == 512 and n % 128 == 0 and m > 4 * n:
            return "64x128"
        return "64x64"
    if m % 128 or n % 128:
        return "128x64" if m % 128 == 0 else ("64x128" if n % 128 == 0 else "64x64")
    return "64x64" if b * (m // 128) * (n // 128) < 96 else "128x128"


def shipped(d, sms, l2_bytes):
    """Mirror of routing.rs coopmat_tile (SM count known)."""
    b, m, n, k, a16 = d["b"], d["m"], d["n"], d["k"], d["a16"]
    rows, cols = m // 64, n // 64
    w = sms * 7
    a_bytes, b_bytes = m * k * (2 if a16 else 4), k * n * 2
    xp = math.ceil(rows / max(1.0, w / cols))
    mp = math.ceil(cols / max(1.0, w / rows))
    large = (l2_bytes or 0) >= 16 * MIB
    mfast = (not (large and b_bytes <= l2_bytes) and a_bytes * (mp - 1) <= 8 * MIB
             and a_bytes * mp + b_bytes < 0.75 * (b_bytes * xp + a_bytes))
    t64 = "64x64m" if mfast else "64x64"
    if m % 128:
        return t64
    if large:
        rho = 1.02 if a16 else 0.92
    elif a16:
        rho = 0.8 if (not mfast and xp > 1) else 1.02
    else:
        rho = 0.9 if (mfast or xp > 1) else (1.0 if a_bytes <= 4 * MIB else 0.85)

    def score(tiles, r):
        per = -(-tiles // sms)
        return tiles / (sms * per) * min(1.0, per / 3.0) * r
    return t64 if score(b * rows * cols, rho) >= score(b * (m // 128) * cols, 1.0) else "128x64"


# ---------------------------------------------------------------- model
def sm_time(n, c, p_inf, e1):
    rate = lambda j: p_inf * min(1.0, j * e1)
    full, r = divmod(n, c)
    return full * c / rate(c) + (r / rate(r) if r else 0.0)


def predict_ms(tile, d, gpu, clock, params, eta=0.85):
    bm, bn, c = TILES[tile]
    b, m, n, k = d["b"], d["m"], d["n"], d["k"]
    if m % bm or n % bn:
        return None
    p_inf, e1, k0 = params
    tiles = b * (m // bm) * (n // bn)
    comp = 2.0 * bm * bn * (k + k0) * sm_time(math.ceil(tiles / gpu["sms"]), c, p_inf, e1) / (clock * 1e6) * 1e3
    bbytes = k * n * 2
    rb = 1 if bbytes <= gpu["l2"] else math.ceil((m // bm) / max(1.0, gpu["sms"] * c / (n // bn)))
    ab = 2 if d["a16"] else 4
    mem = b * (m * k * ab + m * n * ab + bbytes * rb) / (gpu["bw"] * eta) * 1e3
    return max(comp, mem)


def fit(data, gpu, clock):
    import numpy as np
    from scipy.optimize import least_squares
    params = {}
    for a16 in (False, True):
        for tile in ORDER:
            rows = [(d, d["cells"][tile]["ms"]) for d in data.values() if d["a16"] == a16 and tile in d["cells"]]
            res = least_squares(lambda x: [math.log(predict_ms(tile, d, gpu, clock, x) / ms) for d, ms in rows],
                                [400.0, 0.6, 64.0], bounds=([10, 0.05, -1024], [2048, 1.0, 8192]))
            params[(tile, a16)] = (res.x, float(np.sqrt(np.mean(res.fun ** 2))))
    return params


def sweeps():
    data = {g: load_dir(os.path.join(HERE, f"sweep-{g}")) for g in GPUS}
    clocks = {g: median_clock(os.path.join(HERE, f"sweep-{g}")) for g in GPUS}
    return data, clocks


def cmd_model():
    import numpy as np
    data, clocks = sweeps()
    fits = {g: fit(data[g], GPUS[g], clocks[g]) for g in GPUS}
    print("clocks", clocks)
    for (t, a16) in fits["4060"]:
        x4, x3 = fits["4060"][(t, a16)][0], fits["3070"][(t, a16)][0]
        print(f"{t:8s} {'f16' if a16 else 'f32'} A: p_inf 4060 {x4[0]:.0f} 3070 {x3[0]:.0f} ratio {x3[0]/x4[0]:.2f}")
    for fg, pg in (("4060", "3070"), ("3070", "4060"), ("4060", "4060"), ("3070", "3070")):
        errs, regrets = [], []
        for d in data[pg].values():
            pred = {t: predict_ms(t, d, GPUS[pg], clocks[pg], fits[fg][(t, d["a16"])][0]) for t in ORDER if t in d["cells"]}
            pred = {t: v for t, v in pred.items() if v}
            meas = {t: d["cells"][t]["ms"] for t in pred}
            errs += [abs(pred[t] / meas[t] - 1) for t in pred]
            regrets.append(meas[min(pred, key=pred.get)] / min(meas.values()) - 1)
        e, r = np.array(errs), np.array(regrets)
        print(f"fit {fg} -> {pg}: |err| median {np.median(e)*100:.1f}% p90 {np.percentile(e,90)*100:.1f}%; "
              f"regret mean {r.mean()*100:.2f}% max {r.max()*100:.1f}% zero {np.mean(r==0)*100:.0f}%")


def cmd_rule():
    data, _ = sweeps()
    cands = {g: load_dir(os.path.join(HERE, f"candidates-{g}")) for g in GPUS}
    for g, hw in GPUS.items():
        l2 = 24 * MIB if hw["large_l2"] else MIB
        same = better = 0
        worse, gains = [], []
        for case, d in data[g].items():
            new, old = shipped(d, hw["sms"], l2), grokmerge(d)
            ms = {t: c["ms"] for t, c in d["cells"].items()}
            if new == "64x64m":  # M-fast only in the candidate runs: scale by its in-run ratio to 64x64
                c = cands[g][case]["cells"]
                ms["64x64m"] = ms["64x64"] * c["64x64m"]["ms"] / c["64x64"]["ms"]
            r = ms[new] / ms[old] - 1
            gains.append(1 / (1 + r))
            if new == old:
                same += 1
            elif r > 0.01:
                worse.append((case, new, old, r))
            elif r < -0.01:
                better += 1
        print(f"{g}: same {same}, faster {better}, slower {len(worse)} (>1%); geomean x{st.geometric_mean(gains):.3f}")
        for w in worse:
            print(f"   slower {w[0]:18s} new {w[1]:7s} old {w[2]:7s} {w[3]*100:+.1f}%")


def cmd_verify(d, gpu, md):
    shapes = load_dir(d)
    rows, worse, better, same, rn, ro = [], [], 0, 0, [], []
    for case, s in sorted(shapes.items(), key=lambda kv: (kv[1]["a16"], kv[1]["m"], kv[1]["n"], kv[1]["k"])):
        c = s["cells"]
        tiles = {t: c[t]["ms"] for t in ORDER + ["64x64m"] if t in c}
        if "auto" not in c or not tiles:
            continue
        old, new = grokmerge(s), c["auto"]["route"]
        best = min(tiles, key=tiles.get)
        r = tiles[new] / tiles[old] - 1
        rn.append(tiles[new] / tiles[best] - 1)
        ro.append(tiles[old] / tiles[best] - 1)
        if new == old:
            same += 1
        elif r > 0.01:
            worse.append((case, new, old, r, c["auto"]["ms"] / tiles[old] - 1))
        elif r < -0.01:
            better += 1
        rows.append((case, s, old, new, best, tiles, r))
    print(f"{gpu}: {len(rows)} shapes; same pick {same}; faster {better}; slower {len(worse)} (>1%, pinned cells, same run)")
    print(f"regret vs best: new {st.mean(rn)*100:.2f}% (max {max(rn)*100:.1f}%), grokmerge {st.mean(ro)*100:.2f}% (max {max(ro)*100:.1f}%)")
    print(f"geomean new/old x{st.geometric_mean([1/(1+x[6]) for x in rows]):.3f}")
    for w in worse:
        print(f"  slower {w[0]:18s} new {w[1]:7s} old {w[2]:7s} {w[3]*100:+.1f}% (auto cell {w[4]*100:+.1f}%)")
    if md:
        names = ORDER[:1] + ["64x64", "64x64m", "64x128", "128x64"]
        print("\n| shape | 128x128 | 64x64 | 64x64 M-fast | 64x128 | 128x64 | old pick | new pick | new/old |")
        print("|---|---:|---:|---:|---:|---:|---|---|---:|")
        for case, s, old, new, best, tiles, r in rows:
            fl = 2.0 * s["b"] * s["m"] * s["n"] * s["k"]
            cell = lambda t: ("**%.2f**" if t == best else "%.2f") % (fl / tiles[t] * 1e-9) if t in tiles else "n/a"
            nm = lambda t: t.replace("64x64m", "64x64 M-fast")
            tag = "a16 " if s["a16"] else ""
            print(f"| {tag}{s['m']}x{s['n']}x{s['k']} | " + " | ".join(cell(t) for t in names) +
                  f" | {nm(old)} | {nm(new)} | {1/(1+r):.3f} |")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rule"
    if cmd == "model":
        cmd_model()
    elif cmd == "rule":
        cmd_rule()
    elif cmd == "verify":
        cmd_verify(sys.argv[2], sys.argv[3], "--md" in sys.argv)
    elif cmd == "policies":
        data, clocks = sweeps()
        fits = {g: fit(data[g], GPUS[g], clocks[g]) for g in GPUS}
        for g, hw in GPUS.items():
            other = "3070" if g == "4060" else "4060"
            def regret(pick):
                out = []
                for d in data[g].values():
                    ms = {t: d["cells"][t]["ms"] for t in ORDER}
                    out.append(ms[pick(d)] / min(ms.values()) - 1)
                return st.mean(out) * 100
            def model_pick(d, fg, sg):
                p = {t: predict_ms(t, d, GPUS[sg], 1000.0, fits[fg][(t, d["a16"])][0]) for t in ORDER}
                return min((t for t in p if p[t]), key=lambda t: p[t])
            print(f"{g}: grokmerge {regret(grokmerge):.2f}%  always-128x64 {regret(lambda d: '128x64'):.2f}%  "
                  f"model({other} fit, own S) {regret(lambda d: model_pick(d, other, g)):.2f}%  "
                  f"model({other} fit, {other} S) {regret(lambda d: model_pick(d, other, other)):.2f}%")
