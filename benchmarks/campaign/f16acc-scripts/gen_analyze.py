import sys
R = sys.argv[1]
CASES = ("qo","qkv","gu","dn","s1k","s2k")
def load(f):
    rows = {}
    for line in open(R + f):
        p = line.split()
        if len(p) >= 7 and p[0] in CASES:
            try: rows.setdefault(p[0], {})[p[1]] = float(p[2])
            except ValueError: pass
    return rows
def twins(r, pre, suf=""):
    lit = [pre + "coopmat" if pre == "a16_" else "f16w_coopmat_aligned", pre + "coopmat_m64", pre + "coopmat_m64n128", pre + "coopmat_m128n64"]
    for i, l in enumerate(lit):
        g = f"{pre}coopmat_g_t{i+1}{suf}"
        if g not in next(iter(r.values())): continue
        l2 = l if l in next(iter(r.values())) else None
        print(f"  t{i+1} {g:28s}", " ".join(f"{c}:{(r[c][g]/r[c][l]-1)*100:+5.1f}%" for c in r if l in r[c]))
def s2(r, ref, pat):
    for c, d in r.items():
        ns = {k: v for k, v in d.items() if pat in k}
        best = min(ns, key=ns.get)
        print(f"  {c:4s} {ref} {d[ref]:.3f}  best new {best} {ns[best]:.3f} ({(d[ref]/ns[best]-1)*100:+.1f}%)")
for f, pre, ref, pat, suf in [("gen-f16w-4060.txt","", "auto", "_g_n", ""), ("gen-f16w-4060-rep.txt","", "auto", "_g_n", ""),
                    ("gen-acc16-a16-4060.txt","a16_","auto@f16","_g_n",  "_acc16"), ("gen-acc16-a16-4060-rep.txt","a16_","auto@f16","_g_n","_acc16")]:
    r = load(f); print("==", f)
    if pre == "": twins(r, "")
    s2(r, ref, pat)
