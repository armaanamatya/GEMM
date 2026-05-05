import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HDR    = "#1F497D"
WHITE  = "#FFFFFF"
LTBLUE = "#DDEEFF"
YELLOW = "#FFF2CC"
GREEN  = "#C6EFCE"
RED    = "#FFBCBC"
GRAY   = "#F2F2F2"

rows = [
    ("Architecture",          "Ampere (GA102)",       "Blackwell (GB206)",    "RDNA3 (Navi31)",      LTBLUE),
    ("Compute Units",         "68 SMs",               "36 SMs",               "96 CUs",              GRAY  ),
    ("CUDA / Stream Procs",   "8,704 cores",          "4,608 cores",          "6,144 shaders",       GRAY  ),
    ("Warp / Wavefront",      "32 threads",           "32 threads",           "64 threads ⚠",        YELLOW),
    ("VRAM",                  "10 GB GDDR6X",         "16 GB GDDR7",          "24 GB GDDR6",         GRAY  ),
    ("Memory Bandwidth",      "760 GB/s",             "672 GB/s",             "960 GB/s ★",          GREEN ),
    ("L2 Cache",              "5 MB ⚠",              "~32 MB ★",            "6 MB",                YELLOW),
    ("Shared Mem / SM (LDS)", "100 KB",               "100 KB",               "64 KB ⚠",            RED   ),
    ("Registers / SM",        "65,536 × 32-bit",      "65,536 × 32-bit",      "512 VGPRs / wvfnt",  GRAY  ),
    ("Best Tile @ N=8192",    "128×128×32",           "128×64×32",            "128×256×32",          GRAY  ),
    ("Peak — Our Kernel",     "57.3 TFLOPS\n(97.3% cuBLAS)",
                              "50.8 TFLOPS\n(106% cuBLAS)",
                              "89.1 TFLOPS\n(97.6% rocBLAS)",               GREEN ),
]

col_labels = ["Property", "RTX 3080\n(Ampere)", "RTX 5060 Ti\n(Blackwell)", "AMD RX 7900 XTX\n(RDNA3)"]
col_widths = [0.28, 0.22, 0.22, 0.24]

cell_text   = [[r[0], r[1], r[2], r[3]] for r in rows]
cell_colors = [[r[4]]*4 for r in rows]

fig, ax = plt.subplots(figsize=(14, 8.0))
fig.subplots_adjust(bottom=0.08)
ax.set_axis_off()

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellColours=cell_colors,
    colWidths=col_widths,
    loc='upper center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)

for j in range(4):
    cell = tbl[0, j]
    cell.set_facecolor(HDR)
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_height(0.075)

tall = {11}
for i in range(1, len(rows) + 1):
    for j in range(4):
        cell = tbl[i, j]
        cell.set_height(0.082 if i in tall else 0.065)
        if j == 0:
            cell.set_text_props(ha='left', fontweight='bold', fontsize=11)
        if i == 1:
            cell.set_text_props(fontweight='bold')
        if i == len(rows) and j in (1, 2, 3):
            cell.set_text_props(fontweight='bold', fontsize=11)

plt.title("Hardware Architecture Comparison  —  Three GPU Platforms",
          fontsize=13.5, fontweight='bold', color='#1F497D', pad=10)

out = r"C:\Users\Armaan\Desktop\classes\spring 25-26\COSC-4397-Parallel COmp of GPU\project\benchmarks\results\plots\hw_comparison_table.png"
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print("Saved:", out)
