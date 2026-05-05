import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HDR   = "#1F497D"
WHITE = "#FFFFFF"
RED   = "#FFBCBC"
GREEN = "#C6EFCE"
GOLD  = "#FFF2CC"

cols = ["N", "Naive\n64x64", "Large Tiles\n(no pipeline)", "+Shared-mem\nSwizzle",
        "+Async\nPipeline", "Full Autotune\n(our kernel)", "cuBLAS/rocBLAS\n(vendor)"]

col_widths = [0.055, 0.10, 0.135, 0.135, 0.11, 0.135, 0.125]

# Mean TFLOPS over 10 runs per GPU
# 5060 Ti: V3/V4 intermediate not collected in ablation -> "—"
gpu_data = {
    "RTX 3080  (Ampere)": [
        ["512",  "23.7", "2.1",  "2.2",  "6.5",  "25.5", "17.7"],
        ["1024", "47.4", "7.1",  "7.4",  "27.4", "50.1", "41.0"],
        ["2048", "54.8", "4.9",  "5.4",  "57.4", "55.7", "54.7"],
        ["4096", "43.9", "4.8",  "5.2",  "56.4", "55.4", "56.2"],
        ["8192", "40.7", "4.7",  "5.1",  "58.8", "57.3", "58.9"],
    ],
    "RTX 5060 Ti  (Blackwell)": [
        ["512",  "16.7", "4.0",  "3.2",  "8.3",  "19.2", "18.7"],
        ["1024", "33.9", "14.6", "12.2", "37.5", "36.6", "34.9"],
        ["2048", "47.4", "19.5", "19.2", "42.5", "45.6", "46.2"],
        ["4096", "45.8", "21.3", "20.9", "47.0", "50.0", "48.7"],
        ["8192", "22.6", "22.0", "21.6", "48.8", "49.1", "47.7"],
    ],
    "AMD RX 7900 XTX  (RDNA3)": [
        ["512",  "9.2",  "2.2",  "2.3",  "6.4",  "14.5", "16.7"],
        ["1024", "33.0", "9.4",  "9.8",  "25.3", "39.5", "42.1"],
        ["2048", "55.9", "12.5", "13.4", "62.8", "74.2", "82.0"],
        ["4096", "62.7", "12.4", "13.8", "76.2", "85.8", "100.6"],
        ["8192", "43.4", "12.5", "13.5", "82.8", "89.1", "91.3"],
    ],
}

def cell_color(col_idx):
    if col_idx == 0:       return WHITE
    if col_idx in (2, 3):  return RED
    if col_idx in (4, 5):  return GREEN
    if col_idx == 6:       return GOLD
    return WHITE

n_cols = len(cols)
n_rows = 5
row_colors = [[cell_color(j) for j in range(n_cols)] for _ in range(n_rows)]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.subplots_adjust(hspace=0.55, bottom=0.07)

for ax, (gpu_label, rows) in zip(axes, gpu_data.items()):
    ax.set_axis_off()
    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellColours=row_colors,
        colWidths=col_widths,
        loc='upper center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(HDR)
        cell.set_text_props(color='white', fontweight='bold', fontsize=10.5)
        cell.set_height(0.22)

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_height(0.16)
            if j == 0:
                cell.set_text_props(fontweight='bold')
            if j in (2, 3):
                cell.set_text_props(color='#990000', fontweight='bold')
            if j in (4, 5):
                cell.set_text_props(fontweight='bold')
            if j == 6:
                cell.set_text_props(fontweight='bold')

    ax.set_title(gpu_label, fontsize=12, fontweight='bold',
                 color='#1F497D', pad=4, loc='left')

patches = [
    mpatches.Patch(color=WHITE, ec='#AAAAAA', label='Naive baseline'),
    mpatches.Patch(color=RED,   label='No pipeline (collapses at large N)'),
    mpatches.Patch(color=GREEN, label='Pipeline / our full kernel'),
    mpatches.Patch(color=GOLD,  label='Vendor cuBLAS / rocBLAS'),
]
fig.legend(handles=patches, loc='lower center', ncol=4,
           fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Ablation Study  —  TFLOPS across all matrix sizes  (mean over 10 runs)",
             fontsize=13.5, fontweight='bold', color='#1F497D', y=0.98)

out = r"C:\Users\Armaan\Desktop\classes\spring 25-26\COSC-4397-Parallel COmp of GPU\project\benchmarks\results\plots\ablation_table.png"
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print("Saved:", out)
