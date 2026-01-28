Param(
  [string]$Python = "python",
  [string]$WheelDir = "wheels"
)

$ErrorActionPreference = "Stop"

Write-Host "== FLUX LoRA DLC: offline install ==" -ForegroundColor Cyan

if (!(Test-Path $WheelDir)) {
  throw "Wheelhouse not found: '$WheelDir'. Run scripts\\build_wheels.ps1 on an online machine first."
}

& $Python -m pip install -U pip
& $Python -m pip install --no-index --find-links $WheelDir -r "requirements.offline.lock.txt"

Write-Host "`nOffline install complete." -ForegroundColor Green
Write-Host "Run: `$env:FLUX_OFFLINE=1; $Python app.py" -ForegroundColor Green
