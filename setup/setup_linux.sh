#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "Creating virtual environment..."
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch CUDA 12.8 build..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing Triton..."
pip install triton

echo "Running sanity checks..."
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('cuda available', torch.cuda.is_available())"
python -c "import torch; print('torch cuda', torch.version.cuda)"
python -c "import torch; print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import triton; print('triton', triton.__version__)"

echo "Done."
echo "Next run:"
echo "  python triton_gemm.py"
echo "  python test_correctness.py"
echo "  python run_week1.py"
