# =============================================================
#  10-run Ablation benchmark harness
#  Usage: .\10xablation5060ti.ps1
# =============================================================
param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [int]$NumRuns = 10,
    [int]$Warmup = 25,
    [int]$Reps = 100
)

$GpuName  = "5060ti"
$OutBase  = "benchmarks\results\10xablation5060ti"
$Bench    = "benchmarks\ablation.py"

Write-Host "========================================================"
Write-Host "  Ablation Study Harness  - GPU ${GpuName}"
Write-Host "  Runs   : ${NumRuns}"
Write-Host "  Output : ${OutBase}\"
Write-Host "  Python : ${PythonExe}"
Write-Host "========================================================"
Write-Host ""

& $PythonExe -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Sanity check failed -- aborting." -ForegroundColor Red
    exit 1
}

Write-Host ""

$FailedRuns = @()
for ($i = 1; $i -le $NumRuns; $i++) {
    $RunId  = "{0:D2}" -f $i
    $RunDir = "$OutBase\run_$RunId"
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    Write-Host "--------------------------------------------------------"
    Write-Host "  Run $RunId / $("{0:D2}" -f $NumRuns)   ->  $RunDir"
    Write-Host "--------------------------------------------------------"

    & $PythonExe $Bench `
        --gpu-name   $GpuName `
        --run-id     $RunId `
        --output-dir $RunDir `
        --warmup     $Warmup `
        --rep        $Reps

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: Run $RunId exited with errors -- continuing." -ForegroundColor Yellow
        $FailedRuns += $RunId
    }

    Write-Host ""
}

Write-Host "========================================================"
Write-Host "  All $NumRuns runs attempted."
Write-Host "  Output: $OutBase\"
if ($FailedRuns.Count -gt 0) {
    Write-Host "  FAILED runs: $($FailedRuns -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "  All runs completed successfully." -ForegroundColor Green
}
Write-Host "========================================================"