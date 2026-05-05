#!/usr/bin/env bash
# =============================================================
#  10-run Ablation study harness
#  Usage: bash 10xablation5060ti.sh
# =============================================================
set -uo pipefail

GPU_NAME="5060ti"
NUM_RUNS=10
OUT_BASE="benchmarks/results/10xablation5060ti"
BENCH="benchmarks/ablation.py"
WARMUP=25
REPS=100

echo "========================================================"
echo "  Ablation Study Harness  - GPU ${GPU_NAME}"
echo "  Runs   : ${NUM_RUNS}"
echo "  Output : ${OUT_BASE}/"
echo "  Warmup : ${WARMUP}   Reps: ${REPS}"
echo "========================================================"
echo ""

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ERROR: No CUDA device found -- aborting."; exit 1; }
echo ""

FAILED_RUNS=()

for i in $(seq -f "%02g" 1 ${NUM_RUNS}); do
    RUN_DIR="${OUT_BASE}/run_${i}"
    mkdir -p "${RUN_DIR}"

    echo "--------------------------------------------------------"
    echo "  Run ${i} / $(printf '%02d' ${NUM_RUNS})   ->  ${RUN_DIR}"
    echo "--------------------------------------------------------"

    python "${BENCH}" \
        --gpu-name   "${GPU_NAME}" \
        --run-id    "${i}" \
        --output-dir "${RUN_DIR}" \
        --warmup    "${WARMUP}" \
        --rep      "${REPS}" \
    || {
        echo "  WARNING: Run ${i} exited with errors -- continuing to next run."
        FAILED_RUNS+=("${i}")
    }

    echo ""
done

echo "========================================================"
echo "  All ${NUM_RUNS} runs attempted."
echo "  Output directory: ${OUT_BASE}/"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "  FAILED runs: ${FAILED_RUNS[*]}"
else
    echo "  All runs completed successfully."
fi
echo "========================================================"