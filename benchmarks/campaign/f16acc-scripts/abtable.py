#!/usr/bin/env python3
"""ab text output -> markdown: per case, gpu_ms / TF/s per kernel and
speedup vs a named baseline kernel label (default: first row of the case)."""
import sys, collections
path = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else None
rows = collections.OrderedDict()
for line in open(path):
    parts = line.split()
    if len(parts) >= 7 and parts[0] not in ("case", "ab") and not line.startswith("-") and not line.startswith("=="):
        try:
            ms = float(parts[2]); tf = float(parts[4])
        except ValueError:
            continue
        rows.setdefault(parts[0], []).append((parts[1], ms, float(parts[3]), tf, parts[6]))
labels = []
for case, rs in rows.items():
    for r in rs:
        if r[0] not in labels: labels.append(r[0])
print("| case | " + " | ".join(labels) + " |")
print("|---|" + "---|" * len(labels))
for case, rs in rows.items():
    d = {r[0]: r for r in rs}
    b = d.get(base, rs[0]) if base else rs[0]
    cells = []
    for l in labels:
        if l in d:
            r = d[l]; cells.append(f"{r[1]:.3f} ms / {r[3]:.1f} TF/s ({b[1]/r[1]:.2f}x)")
        else:
            cells.append("n/a")
    print(f"| {case} | " + " | ".join(cells) + " |")
