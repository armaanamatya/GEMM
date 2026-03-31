# Setup Guide — RTX 5060 Ti 16GB

This guide is for your RTX 5060 Ti. The 5060 Ti is a Blackwell-family GPU, and current PyTorch stable builds expose CUDA 12.8 as an install option. Use a recent NVIDIA driver and install the CUDA 12.8 PyTorch wheel; PyTorch bundles its own CUDA runtime, so your system reporting CUDA 13.1 is fine as long as the driver is current.

## Recommended versions
- Python: 3.10 or 3.11
- PyTorch: latest stable, CUDA 12.8 wheel
- Triton: latest pip package

## Windows setup
1. Install Python 3.10 or 3.11.
2. Open PowerShell in the project folder.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

5. Install PyTorch for CUDA 12.8:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

6. Install Triton:

```powershell
pip install triton
```

## Linux setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install triton
```

## Verify install
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import triton; print(triton.__version__)"
```

Expected:
- `torch.cuda.is_available()` -> `True`
- `torch.version.cuda` may print `12.8`
- device name should show your RTX 5060 Ti

## Run the project
```bash
python triton_gemm.py
python test_correctness.py
python run_week1.py
```

## If pip cannot find torch
Common causes:
- wrong Python version, especially Python 3.13+
- old pip
- stale environment

Try:
```bash
python -m pip install --upgrade pip
python --version
```

Then recreate the environment with Python 3.10 or 3.11.
