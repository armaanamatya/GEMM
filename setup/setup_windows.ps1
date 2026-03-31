Param(
    [ValidateSet("5060ti","3080")]
    [string]$Gpu = "5060ti",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.11"
)

Write-Host "Creating virtual environment..."
& $PythonLauncher -$PythonVersion -m venv .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing PyTorch CUDA 12.8 build..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing Triton..."
pip install triton

Write-Host "Running sanity checks..."
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('cuda available', torch.cuda.is_available())"
python -c "import torch; print('torch cuda', torch.version.cuda)"
python -c "import torch; print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import triton; print('triton', triton.__version__)"

Write-Host "Done. GPU profile selected: $Gpu"
Write-Host "Next run:"
Write-Host "  python triton_gemm.py"
Write-Host "  python test_correctness.py"
Write-Host "  python run_week1.py"
