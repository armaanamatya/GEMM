"""
Generate publication-quality figures from 10x benchmark runs.

Figures produced (PDF + PNG):
  fig1_performance        -- TFLOPS vs N, two panels (5060 Ti / 3080), shaded ±1σ
  fig2_vs_cublas          -- Normalized to cuBLAS %, highlights where Triton wins
  fig3_autotune_necessity -- Fixed vs autotuned speedup bar chart with annotations
  fig4_cutlass_ramp       -- CUTLASS JIT cold-start on RTX 3080 (linear + log)

Usage (from repo root):
  python benchmarks/generate_plots.py [--show]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZES   = [512, 1024, 2048, 4096, 8192]
OUT_DIR = Path("benchmarks/results/plots")

# Palette: colorblind-safe, prints well in greyscale via line-style variety
PAL = {
    "cublas":            "#1B2A4A",   # deep navy   — authoritative reference
    "triton_auto_fp32":  "#C0392B",   # bold red    — main contribution
    "triton_auto_fp16":  "#E97B72",   # soft red    — fp16 variant
    "triton_fixed_fp32": "#2471A3",   # clear blue  — naive baseline
    "triton_fixed_fp16": "#7FB3D3",   # light blue
    "cutlass":           "#1E8449",   # forest green
    "pytorch_fp32":      "#AAB7B8",   # neutral grey
}

# (linestyle, linewidth)
LSW = {
    "cublas":            ("-",  2.5),
    "triton_auto_fp32":  ("-",  2.2),
    "triton_auto_fp16":  ("--", 1.8),
    "triton_fixed_fp32": ("-",  1.8),
    "triton_fixed_fp16": ("--", 1.4),
    "cutlass":           ("-",  2.0),
    "pytorch_fp32":      (":",  1.2),
}

MARKERS = {
    "cublas":            "D",
    "triton_auto_fp32":  "o",
    "triton_auto_fp16":  "o",
    "triton_fixed_fp32": "s",
    "triton_fixed_fp16": "s",
    "cutlass":           "^",
    "pytorch_fp32":      "x",
}

LEGEND_LABELS = {
    "cublas":            "cuBLAS FP16 (reference)",
    "triton_auto_fp32":  "Triton Autotuned  FP32-acc  ← our result",
    "triton_auto_fp16":  "Triton Autotuned  FP16-acc",
    "triton_fixed_fp32": "Triton Fixed      FP32-acc  (no swizzle)",
    "triton_fixed_fp16": "Triton Fixed      FP16-acc",
    "cutlass":           "CUTLASS 3.5 FP16",
    "pytorch_fp32":      "PyTorch FP32 (TF32)",
}

# Maps backend key → CSV column name
TFLOPS_COL = {
    "cublas":            "tflops_cublas_fp16",
    "triton_auto_fp32":  "tflops_triton_auto_fp32",
    "triton_auto_fp16":  "tflops_triton_auto_fp16",
    "triton_fixed_fp32": "tflops_triton_fixed_fp32",
    "triton_fixed_fp16": "tflops_triton_fixed_fp16",
    "cutlass":           "tflops_cutlass",
    "pytorch_fp32":      "tflops_pytorch_fp32",
}

XTICKS  = SIZES
XLABELS = ["512", "1K", "2K", "4K", "8K"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gpu(gpu: str) -> dict[str, dict[int, list[float]]]:
    """Load all 10 run CSVs for a GPU into {backend: {n: [tflops, ...]}}."""
    pattern = f"benchmarks/results/10x{gpu}/run_*/results.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [warn] no CSVs matching {pattern}", file=sys.stderr)
        return {}
    data: dict[str, dict[int, list[float]]] = {bk: defaultdict(list) for bk in TFLOPS_COL}
    for path in files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                n = int(float(row["n"]))
                for bk, col in TFLOPS_COL.items():
                    v = row.get(col, "")
                    if v and v not in ("None", ""):
                        try:
                            data[bk][n].append(float(v))
                        except ValueError:
                            pass
    return data


def _ms(data: dict, bk: str, n: int) -> tuple[float, float]:
    """Return (mean, std) for a backend+size; (nan, nan) if no data."""
    vals = data.get(bk, {}).get(n, [])
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
    return m, s


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            pass
    plt.rcParams.update({
        "font.family":          "serif",
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      7.5,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#CCCCCC",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.06,
    })


def _size_axis(ax: plt.Axes, ylabel: str = "Throughput (TFLOPS)",
               ymin: float | None = 0) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Matrix Size N  (N×N FP16 GEMM)", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xlim(380, 11000)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)


def _plot_line(ax: plt.Axes, data: dict, bk: str,
               label: str | None = None, alpha_band: float = 0.13) -> None:
    xs, ys, es = [], [], []
    for n in SIZES:
        m, s = _ms(data, bk, n)
        if not math.isnan(m):
            xs.append(n); ys.append(m); es.append(s)
    if not xs:
        return
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)
    ls, lw = LSW[bk]
    mk = MARKERS[bk]
    lbl = label if label is not None else LEGEND_LABELS[bk]
    ax.plot(xs, ys, ls, marker=mk, markersize=5, color=PAL[bk],
            lw=lw, label=lbl, zorder=4)
    ax.fill_between(xs, ys - es, ys + es, color=PAL[bk],
                    alpha=alpha_band, zorder=2)


def _annotate_delta(ax: plt.Axes, data: dict, bk: str,
                    ref_bk: str, n: int, xytext: tuple) -> None:
    m_bk,  _ = _ms(data, bk,     n)
    m_ref, _ = _ms(data, ref_bk, n)
    if math.isnan(m_bk) or math.isnan(m_ref):
        return
    pct  = 100.0 * (m_bk - m_ref) / m_ref
    sign = "+" if pct >= 0 else ""
    ax.annotate(
        f"{sign}{pct:.1f}%\nvs cuBLAS",
        xy=(n, m_bk), xytext=xytext,
        fontsize=7, color=PAL[bk], fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=PAL[bk],
                        lw=0.9, mutation_scale=8),
    )


def _save(fig: plt.Figure, name: str, show: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  saved  {p}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — TFLOPS vs N, both GPUs
# ---------------------------------------------------------------------------

def fig1_performance(d5: dict, d3: dict, show: bool = False) -> None:
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)

    GPU_PANELS = [
        ("(a)  RTX 5060 Ti", d5, False),
        ("(b)  RTX 3080",    d3, True),
    ]
    SHOW = ["cublas", "triton_auto_fp32", "triton_auto_fp16",
            "triton_fixed_fp32", "pytorch_fp32"]

    for ax, (title, data, show_cutlass) in zip(axes, GPU_PANELS):
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            ax.set_title(title, loc="left", fontweight="bold")
            continue

        for bk in SHOW:
            _plot_line(ax, data, bk)
        if show_cutlass:
            _plot_line(ax, data, "cutlass")

        _size_axis(ax)
        ax.set_title(title, loc="left", fontweight="bold")
        n_runs = len(glob.glob(f"benchmarks/results/10x{'5060ti' if 'Ti' in title else '3080'}/run_*/results.csv"))
        ax.text(0.97, 0.03, f"n = {n_runs} runs  |  shading = ±1σ",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
                color="#888888")

        # Small value labels at N=8192 endpoint
        for bk in ("triton_auto_fp32", "cublas"):
            m, _ = _ms(data, bk, 8192)
            if not math.isnan(m):
                ax.annotate(f"{m:.1f}", xy=(8192, m),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=6.5, color=PAL[bk], va="center",
                            fontweight="bold")

    # Shared legend below both panels
    handles, labels = [], []
    seen: set[str] = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.22), frameon=True)

    fig.suptitle("FP16 GEMM Throughput — Triton vs cuBLAS vs CUTLASS",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_performance", show)


# ---------------------------------------------------------------------------
# Figure 2 — % of cuBLAS
# ---------------------------------------------------------------------------

def fig2_vs_cublas(d5: dict, d3: dict, show: bool = False) -> None:
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    GPU_PANELS = [("(a)  RTX 5060 Ti", d5), ("(b)  RTX 3080", d3)]
    SHOW = ["triton_auto_fp32", "triton_auto_fp16", "triton_fixed_fp32"]

    for ax, (title, data) in zip(axes, GPU_PANELS):
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue

        # Green band marks "beating cuBLAS"
        ax.axhspan(100, 118, color="#D5F5E3", alpha=0.45, zorder=0)
        ax.axhline(100, color=PAL["cublas"], lw=1.6, ls="--",
                   label="cuBLAS  =  100 %", zorder=3)

        for bk in SHOW:
            xs, ys, es = [], [], []
            for n in SIZES:
                m_bk,  s_bk  = _ms(data, bk,      n)
                m_ref, s_ref = _ms(data, "cublas", n)
                if math.isnan(m_bk) or math.isnan(m_ref) or m_ref == 0:
                    continue
                pct = 100.0 * m_bk / m_ref
                err = pct * math.sqrt((s_bk / m_bk) ** 2 + (s_ref / m_ref) ** 2) \
                      if m_bk > 0 else 0.0
                xs.append(n); ys.append(pct); es.append(err)
            if not xs:
                continue
            xs, ys, es = np.array(xs), np.array(ys), np.array(es)
            ls, lw = LSW[bk]
            ax.plot(xs, ys, ls, marker=MARKERS[bk], markersize=5,
                    color=PAL[bk], lw=lw, label=LEGEND_LABELS[bk], zorder=4)
            ax.fill_between(xs, ys - es, ys + es, color=PAL[bk],
                            alpha=0.13, zorder=2)

        _size_axis(ax, ylabel="Throughput relative to cuBLAS (%)", ymin=None)
        ax.set_ylim(25, 118)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="lower right", fontsize=7)

        # Label the "beats cuBLAS" shading
        ax.text(520, 108, "  beats cuBLAS", fontsize=6.5,
                color="#1E8449", va="center")

    fig.suptitle("Triton GEMM Performance Relative to cuBLAS  (shading = ±1σ propagated error)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_vs_cublas", show)


# ---------------------------------------------------------------------------
# Figure 3 — Why autotuning matters (grouped bars)
# ---------------------------------------------------------------------------

def fig3_autotune_necessity(d5: dict, d3: dict, show: bool = False) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))

    gpus   = [("5060 Ti", d5), ("3080", d3)]
    ns_show = [4096, 8192]
    BAR_KEYS = [
        ("triton_fixed_fp32", "Fixed tiling\n(64×64, no swizzle)"),
        ("triton_auto_fp32",  "Autotuned\n(128×256 + swizzle)"),
        ("cublas",            "cuBLAS FP16\n(reference)"),
    ]

    bar_w  = 0.20
    group_gap = 0.55   # gap between n=4K and n=8K within one GPU
    gpu_gap   = 0.85   # gap between GPUs

    centers: list[float] = []
    group_labels: list[str] = []
    x = 0.0

    for gi, (gpu_label, data) in enumerate(gpus):
        for ni, n in enumerate(ns_show):
            cx = x + (len(BAR_KEYS) - 1) * bar_w / 2
            centers.append(cx)
            group_labels.append(f"{gpu_label}\nn={'4K' if n == 4096 else '8K'}")

            for bi, (bk, _) in enumerate(BAR_KEYS):
                m, s = _ms(data, bk, n)
                bx   = x + bi * bar_w
                show_label = (gi == 0 and ni == 0)
                ax.bar(bx, m if not math.isnan(m) else 0,
                       width=bar_w * 0.86, color=PAL[bk], alpha=0.88, zorder=3,
                       label=BAR_KEYS[bi][1] if show_label else None)
                if not math.isnan(m):
                    ax.errorbar(bx, m, yerr=s, fmt="none",
                                color="#333333", capsize=3.5, lw=1.2, zorder=5)

            # Annotate speedup of auto over fixed
            m_auto, _ = _ms(data, "triton_auto_fp32",  n)
            m_fix,  _ = _ms(data, "triton_fixed_fp32", n)
            if not (math.isnan(m_auto) or math.isnan(m_fix)) and m_fix > 0:
                spd = m_auto / m_fix
                bx_auto = x + 1 * bar_w  # bar index 1 = auto
                ax.text(bx_auto, m_auto + _ms(data, "triton_auto_fp32", n)[1] + 1.8,
                        f"{spd:.2f}×", ha="center", va="bottom",
                        fontsize=7.5, color=PAL["triton_auto_fp32"], fontweight="bold")

            x += len(BAR_KEYS) * bar_w + group_gap
        x += gpu_gap

    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_ylim(0, max(
        (_ms(d3, "cutlass",  8192)[0] or 0),
        (_ms(d3, "triton_auto_fp32", 8192)[0] or 0),
        60,
    ) * 1.22)
    ax.set_title(
        "Autotuning Necessity: Fixed Tiling vs Autotuned Triton vs cuBLAS\n"
        "Annotated ratio = autotuned ÷ fixed throughput",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7.5, ncol=3)
    ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig3_autotune_necessity", show)


# ---------------------------------------------------------------------------
# Figure 4 — CUTLASS cold-start profile (3080 only)
# ---------------------------------------------------------------------------

def fig4_cutlass_ramp(d3: dict, show: bool = False) -> None:
    _setup_style()
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # Short labels to prevent legend text overflow
    SHORT = {
        "cublas":           "cuBLAS FP16",
        "triton_auto_fp32": "Triton Auto FP32",
        "cutlass":          "CUTLASS 3.5",
    }
    SHOW = ["cublas", "triton_auto_fp32", "cutlass"]

    for ax, use_log in [(ax_lin, False), (ax_log, True)]:
        for bk in SHOW:
            _plot_line(ax, d3, bk, label=SHORT[bk])
        _size_axis(ax, ymin=None)
        ax.set_xlim(380, 11000)

        if use_log:
            ax.set_yscale("log")
            ax.set_ylabel("Throughput (TFLOPS, log scale)")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_title("(b)  Log y-axis — reveals cold-start magnitude",
                         loc="left", fontweight="bold")
            ax.legend(loc="lower right", fontsize=7.5)
        else:
            ax.set_title("(a)  Linear y-axis", loc="left", fontweight="bold")
            ax.legend(loc="upper left", fontsize=7.5)

            # CUTLASS n=512 cold-start: annotate near the point, arrow points down-left
            m_c512, _ = _ms(d3, "cutlass", 512)
            if not math.isnan(m_c512):
                ax.annotate(
                    f"CUTLASS @ n=512\n{m_c512:.2f} TFLOPS\n(JIT cold-start)",
                    xy=(512, m_c512), xytext=(700, 28),
                    fontsize=6.5, color=PAL["cutlass"],
                    arrowprops=dict(arrowstyle="-|>", color=PAL["cutlass"],
                                    lw=0.9, mutation_scale=8),
                )

            # CUTLASS win at n=8192: corner text box, no arrow into crowded area
            m_c8, _ = _ms(d3, "cutlass",  8192)
            m_b8, _ = _ms(d3, "cublas",   8192)
            if not (math.isnan(m_c8) or math.isnan(m_b8)):
                pct = 100.0 * (m_c8 - m_b8) / m_b8
                ax.text(0.97, 0.40,
                        f"CUTLASS @ N=8192\n+{pct:.0f}% vs cuBLAS",
                        transform=ax.transAxes, fontsize=7, ha="right", va="center",
                        color=PAL["cutlass"], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=PAL["cutlass"], alpha=0.9, lw=0.8))

    fig.suptitle(
        "CUTLASS 3.5 Cold-Start Profile — RTX 3080\n"
        "JIT-compiles CUDA kernel on first call; "
        "do_bench warmup=25 absorbs compilation cost",
        fontsize=9, y=1.04,
    )
    fig.tight_layout()
    _save(fig, "fig4_cutlass_ramp", show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper figures from 10x runs")
    ap.add_argument("--show", action="store_true", help="Display interactively")
    args = ap.parse_args()

    print("Loading 10x benchmark data …")
    d5 = load_gpu("5060ti")
    d3 = load_gpu("3080")

    if not d5 and not d3:
        sys.exit("ERROR: No CSVs found. Run from the project root.")

    runs5 = len(glob.glob("benchmarks/results/10x5060ti/run_*/results.csv"))
    runs3 = len(glob.glob("benchmarks/results/10x3080/run_*/results.csv"))
    print(f"  5060 Ti: {runs5} runs     3080: {runs3} runs")

    print("\nGenerating figures …")
    fig1_performance(d5, d3, args.show)
    fig2_vs_cublas(d5, d3, args.show)
    fig3_autotune_necessity(d5, d3, args.show)
    if d3:
        fig4_cutlass_ramp(d3, args.show)

    print(f"\nAll figures → {OUT_DIR}/")


if __name__ == "__main__":
    main()
