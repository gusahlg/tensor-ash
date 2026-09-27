#!/usr/bin/env python3
"""Token agreement + prefill-logit deltas.
usage: tokcmp.py <gen-TAG.txt> <ref.json> <logits-dir>"""
import json, struct, sys, os, math
gen_path, ref_path, ldir = sys.argv[1:4]
gen = {}
for line in open(gen_path):
    parts = line.split()
    if len(parts) >= 3:
        gen[(parts[0], parts[1])] = [int(x) for x in parts[2].split(",")]
ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
def upto_eos(a):
    return a[:a.index(2) + 1] if 2 in a else a
prompts = sorted({p for p, _ in gen})
print("| prompt | T | f16 vs f32 | promote vs f32 | f32 vs llama.cpp | f16 vs llama.cpp | promote vs llama.cpp |")
print("|---|---|---|---|---|---|---|")
for p in prompts:
    f32, f16, pro = gen.get((p, "f32")), gen.get((p, "f16")), gen.get((p, "promote"))
    r = ref.get(p + ".csv", {})
    rt, T = r.get("tokens"), r.get("prompt_len", "?")
    def cell(a, b):
        if a is None or b is None:
            return "n/a"
        n = min(len(a), len(b))
        d = first_div(a, b)
        eos = " (EOS)" if 2 in a[:d + 1] else ""
        return f"{d}/{n}" + ("" if d == n else f" (first diff @{d}{eos})") if d < n else f"{n}/{n}"
    print(f"| {p} | {T} | {cell(f16, f32)} | {cell(pro, f32)} | {cell(f32, rt)} | {cell(f16, rt)} | {cell(pro, rt)} |")
print()
print("| prompt | mode | argmax = f32 | top-5 overlap | max abs logit delta vs f32 | rms delta | max |logit| |")
print("|---|---|---|---|---|---|---|")
def load(p, m):
    path = os.path.join(ldir, f"{p}-{m}.bin")
    if not os.path.exists(path):
        return None
    b = open(path, "rb").read()
    return list(struct.unpack(f"<{len(b)//4}f", b))
for p in prompts:
    base = load(p, "f32")
    if base is None:
        continue
    top = lambda v: set(sorted(range(len(v)), key=lambda i: -v[i])[:5])
    for m in ("f16", "promote"):
        v = load(p, m)
        if v is None:
            continue
        d = [abs(x - y) for x, y in zip(v, base)]
        rms = math.sqrt(sum(x * x for x in d) / len(d))
        am = max(range(len(v)), key=v.__getitem__) == max(range(len(base)), key=base.__getitem__)
        print(f"| {p} | {m} | {'yes' if am else 'NO'} | {len(top(v) & top(base))}/5 | {max(d):.4f} | {rms:.4f} | {max(abs(x) for x in base):.2f} |")
