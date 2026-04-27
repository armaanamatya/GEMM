"""
Generate final paper figures from benchmark CSV results.

Usage:
  pip install matplotlib pandas
  python benchmarks/generate_plots.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RESULTS, "plots")
os.makedirs(OUT, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────

# 5060 Ti
ti_baseline = pd.read_csv(os.path.join(RESULTS, "5060tirun", "sweep.csv"))
ti_auto = pd.read_csv(os.path.join(RESULTS, "5060tirun", "autotune_sweep2.csv"))

# 3080
rtx_baseline = pd.read_csv(os.path.join(RESULTS, "3080run", "sweep.csv"))
rtx_auto = pd.read_csv(os.path.join(RESULTS, "3080run", "autotune_sweep.csv"))

# RX 7900
amd_baseline = pd.read_csv(os.path.join(RESULTS, "amdrun", "sweep.csv"))
amd_auto = pd.read_csv(os.path.join(RESULTS, "amdrun", "autotune_sweep.csv"))

sizes = [512, 1024, 2048, 4096, 8192]
x = np.arange(len(sizes))
w = 0.18

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Fig 1: TFLOPS comparison — both GPUs, fixed vs autotuned vs cuBLAS ─────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

# 5060 Ti
ax1.bar(x - w*1.5, ti_baseline["tflops_triton_fp32_acc"], w, label="Fixed Triton FP32", color="#4e79a7")
ax1.bar(x - w*0.5, ti_auto["tflops_auto_fp32"], w, label="Autotuned Triton FP32", color="#59a14f")
ax1.bar(x + w*0.5, ti_auto["tflops_auto_fp16"], w, label="Autotuned Triton FP16", color="#76b7b2")
ax1.bar(x + w*1.5, ti_baseline["tflops_cublas"], w, label="cuBLAS", color="#e15759")
ax1.set_xlabel("Matrix Size (M = N = K)")
ax1.set_ylabel("TFLOPS")
ax1.set_title("RTX 5060 Ti")
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend(fontsize=9)

# 3080
ax2.bar(x - w*1.5, rtx_baseline["tflops_triton_fp32_acc"], w, label="Fixed Triton FP32", color="#4e79a7")
ax2.bar(x - w*0.5, rtx_auto["tflops_auto_fp32"], w, label="Autotuned Triton FP32", color="#59a14f")
ax2.bar(x + w*0.5, rtx_auto["tflops_auto_fp16"], w, label="Autotuned Triton FP16", color="#76b7b2")
ax2.bar(x + w*1.5, rtx_baseline["tflops_cublas"], w, label="cuBLAS", color="#e15759")
ax2.set_xlabel("Matrix Size (M = N = K)")
ax2.set_ylabel("TFLOPS")
ax2.set_title("RTX 3080")
ax2.set_xticks(x)
ax2.set_xticklabels(sizes)
ax2.legend(fontsize=9)

fig.suptitle("GEMM Throughput: Fixed vs Autotuned Triton vs cuBLAS", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_tflops_both_gpus.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_tflops_both_gpus.pdf"), bbox_inches="tight")
print("Saved fig1_tflops_both_gpus")


# ── Fig 2: % of cuBLAS — autotuned, both GPUs ──────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sizes, ti_auto["pct_cublas_auto_fp32"], "o-", label="5060 Ti (FP32 acc)", color="#4e79a7", linewidth=2)
ax.plot(sizes, ti_auto["pct_cublas_auto_fp16"], "s--", label="5060 Ti (FP16 acc)", color="#76b7b2", linewidth=2)
ax.plot(sizes, rtx_auto["pct_cublas_auto_fp32"], "^-", label="3080 (FP32 acc)", color="#e15759", linewidth=2)
ax.plot(sizes, rtx_auto["pct_cublas_auto_fp16"], "D--", label="3080 (FP16 acc)", color="#f28e2b", linewidth=2)
ax.axhline(100, color="gray", linestyle=":", linewidth=1, label="cuBLAS = 100%")
ax.set_xlabel("Matrix Size (M = N = K)")
ax.set_ylabel("% of cuBLAS Throughput")
ax.set_title("Autotuned Triton as % of cuBLAS", fontweight="bold")
ax.set_xticks(sizes)
ax.legend()
ax.set_ylim(bottom=80)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_pct_cublas.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_pct_cublas.pdf"), bbox_inches="tight")
print("Saved fig2_pct_cublas")


# ── Fig 3: Autotuning speedup over fixed tiles ─────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - w, ti_auto["speedup_auto_vs_fixed_fp32"], w*2, label="5060 Ti", color="#4e79a7")
ax.bar(x + w, rtx_auto["speedup_auto_vs_fixed_fp32"], w*2, label="3080", color="#e15759")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Matrix Size (M = N = K)")
ax.set_ylabel("Speedup (Autotuned / Fixed)")
ax.set_title("Autotuning Speedup over Fixed 64x64x32 Tiles", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()

for i, (v1, v2) in enumerate(zip(ti_auto["speedup_auto_vs_fixed_fp32"], rtx_auto["speedup_auto_vs_fixed_fp32"])):
    ax.text(i - w, v1 + 0.03, f"{v1:.2f}x", ha="center", fontsize=8)
    ax.text(i + w, v2 + 0.03, f"{v2:.2f}x", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_autotune_speedup.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_autotune_speedup.pdf"), bbox_inches="tight")
print("Saved fig3_autotune_speedup")


# ── Fig 4: FP16 accumulation error scaling ──────────────────────────────────

# Error data from correctness tests
err_5060ti = [0.125, 0.3125, 0.75, 1.25, 3.75]
err_3080   = [0.125, 0.3125, 0.75, 1.50, 3.25]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sizes, err_5060ti, "o-", label="5060 Ti", color="#4e79a7", linewidth=2, markersize=8)
ax.plot(sizes, err_3080, "^-", label="3080", color="#e15759", linewidth=2, markersize=8)
ax.set_xlabel("Matrix Size (M = N = K)")
ax.set_ylabel("Max Absolute Error")
ax.set_title("FP16 Accumulation Error vs Matrix Size", fontweight="bold")
ax.set_xticks(sizes)
ax.legend()
ax.set_yscale("log")

# Annotate values
for i, (s, e1, e2) in enumerate(zip(sizes, err_5060ti, err_3080)):
    ax.annotate(f"{e1}", (s, e1), textcoords="offset points", xytext=(10, 5), fontsize=8, color="#4e79a7")
    ax.annotate(f"{e2}", (s, e2), textcoords="offset points", xytext=(10, -12), fontsize=8, color="#e15759")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_error_scaling.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_error_scaling.pdf"), bbox_inches="tight")
print("Saved fig4_error_scaling")


# ── Fig 5: Cross-GPU raw TFLOPS (autotuned FP32 acc) ───────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sizes, ti_auto["tflops_auto_fp32"], "o-", label="5060 Ti — Triton Autotuned", color="#4e79a7", linewidth=2, markersize=8)
ax.plot(sizes, ti_auto["tflops_cublas"], "o--", label="5060 Ti — cuBLAS", color="#76b7b2", linewidth=2, markersize=8)
ax.plot(sizes, rtx_auto["tflops_auto_fp32"], "^-", label="3080 — Triton Autotuned", color="#e15759", linewidth=2, markersize=8)
ax.plot(sizes, rtx_auto["tflops_cublas"], "^--", label="3080 — cuBLAS", color="#f28e2b", linewidth=2, markersize=8)
ax.set_xlabel("Matrix Size (M = N = K)")
ax.set_ylabel("TFLOPS")
ax.set_title("Cross-GPU Performance: Autotuned Triton vs cuBLAS (FP32 Acc)", fontweight="bold")
ax.set_xticks(sizes)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_cross_gpu.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_cross_gpu.pdf"), bbox_inches="tight")
print("Saved fig5_cross_gpu")

# ── Fig 6: 3-GPU summary — Autotuned Triton FP32 vs vendor library ──────────
#
# AMD FP16 autotuner is unreliable at large N (returns configs worse than fixed),
# so this figure shows FP32 accumulation only across all three GPUs.
# AMD FP16 fixed-config numbers are included as a separate series to show the
# precision inversion at small N and collapse at large N.

fig, ax = plt.subplots(figsize=(10, 6))

# colour palette: blue=5060Ti, red=3080, green=AMD
C_TI  = "#4e79a7"
C_RTX = "#e15759"
C_AMD = "#59a14f"

# Triton autotuned FP32 — solid lines, filled markers
ax.plot(sizes, ti_auto["tflops_auto_fp32"],    "o-",  color=C_TI,  lw=2.2, ms=8, label="RTX 5060 Ti — Triton (FP32 acc)")
ax.plot(sizes, rtx_auto["tflops_auto_fp32"],   "s-",  color=C_RTX, lw=2.2, ms=8, label="RTX 3080 — Triton (FP32 acc)")
ax.plot(sizes, amd_auto["tflops_auto_fp32"],   "^-",  color=C_AMD, lw=2.2, ms=8, label="RX 7900 — Triton (FP32 acc)")

# Vendor library baselines — dashed lines, open markers
ax.plot(sizes, ti_auto["tflops_cublas"],        "o--", color=C_TI,  lw=1.5, ms=7, mfc="none", label="RTX 5060 Ti — cuBLAS")
ax.plot(sizes, rtx_auto["tflops_cublas"],       "s--", color=C_RTX, lw=1.5, ms=7, mfc="none", label="RTX 3080 — cuBLAS")
ax.plot(sizes, amd_auto["tflops_cublas"],       "^--", color=C_AMD, lw=1.5, ms=7, mfc="none", label="RX 7900 — rocBLAS")

# AMD FP16 fixed-config (autotuned is broken at large N) — thin dotted
ax.plot(sizes, amd_baseline["tflops_triton_fp16_acc"], "^:",
        color=C_AMD, lw=1.3, ms=6, alpha=0.65, label="RX 7900 — Triton (FP16 acc, fixed)")

ax.set_xlabel("Matrix Size  (M = N = K)", fontsize=12)
ax.set_ylabel("TFLOPS", fontsize=12)
ax.set_title(
    "Triton GEMM vs Vendor Library — RTX 3080 · RTX 5060 Ti · RX 7900\n"
    "Autotuned FP32 accumulation (dashed = vendor baseline)",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.grid(True, alpha=0.3)

# Annotate peak Triton values at N=8192
for y, label, color in [
    (amd_auto["tflops_auto_fp32"].iloc[-1],  "90.7", C_AMD),
    (rtx_auto["tflops_auto_fp32"].iloc[-1],  "58.8", C_RTX),
    (ti_auto["tflops_auto_fp32"].iloc[-1],   "49.9", C_TI),
]:
    ax.annotate(f"{label} TFLOPS", xy=(8192, y),
                xytext=(7600, y + 2.5), fontsize=8, color=color, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_3gpu_summary.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig6_3gpu_summary.pdf"), bbox_inches="tight")
print("Saved fig6_3gpu_summary")

print(f"\nAll plots saved to {OUT}")
