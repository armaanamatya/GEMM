import csv
import glob
import math
import os

GPU = os.environ.get("GPU", "5060ti")
pattern = f"benchmarks/results/10x{GPU}/run_*/results.csv"
files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit(f"No CSVs found matching {pattern}")

# Collect all rows: {(n, col): [float, ...]}
from collections import defaultdict
data = defaultdict(list)
cols = []

for path in files:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not cols:
            cols = [c for c in reader.fieldnames if c.startswith("tflops_")]
        for row in reader:
            n = int(float(row["n"]))
            for c in cols:
                v = row[c]
                if v and v != "None":
                    data[(n, c)].append(float(v))

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

sizes = [512, 1024, 2048, 4096, 8192]
hdrs = ["n=" + str(n) for n in sizes]
print(f"\nRTX {GPU.upper()}  —  {len(files)} runs  |  TFLOPS  mean +/- std\n")
print(f"{'Backend':<26}" + "".join(f"{h:>16}" for h in hdrs))
print("-" * 106)
for c in cols:
    label = c.replace("tflops_", "")
    row = f"{label:<26}"
    for n in sizes:
        vals = data[(n, c)]
        if vals:
            row += f"  {mean(vals):>5.2f}+/-{std(vals):.2f}  "
        else:
            row += f"{'N/A':>16}"
    print(row)
print()
