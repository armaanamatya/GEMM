# Setup Guide — RTX 3080

This guide is for an RTX 3080. The safest current path is still to install a recent stable PyTorch build and Triton in a fresh environment. For the 3080, the current stable CUDA 12.8 wheel is fine and keeps both your 3080 and 5060 Ti on one consistent software stack.

## Recommended versions
- Python: 3.10 or 3.11
- PyTorch: latest stable, CUDA 12.8 wheel
- Triton: latest pip package

## Windows setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
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

## Run the project
```bash
python triton_gemm.py
python test_correctness.py
python run_week1.py
```

## Notes
- Your 3080 is more than enough for this Week 1 GEMM work.
- Staying on the same PyTorch/Triton stack as the 5060 Ti makes team debugging easier.
