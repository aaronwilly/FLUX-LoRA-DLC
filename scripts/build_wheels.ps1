Param(
  [string]$Python = "python",
  [string]$WheelDir = "wheels"
)

$ErrorActionPreference = "Stop"

Write-Host "== FLUX LoRA DLC: build offline wheelhouse ==" -ForegroundColor Cyan

if (!(Test-Path $WheelDir)) {
  New-Item -ItemType Directory -Path $WheelDir | Out-Null
}

# Upgrade build tooling
& $Python -m pip install -U pip setuptools wheel

Write-Host "`n[1/3] Building wheels for git-based deps (accelerate/diffusers/peft)..." -ForegroundColor Cyan
& $Python -m pip wheel --no-deps -r "requirements.vcs.txt" -w $WheelDir

Write-Host "`n[2/3] Downloading CUDA PyTorch wheels (cu121)..." -ForegroundColor Cyan
# NOTE: torch CUDA wheels are hosted on PyTorch's index, not PyPI.
& $Python -m pip download -r "requirements.torch.lock.txt" -d $WheelDir --index-url "https://download.pytorch.org/whl/cu121"

Write-Host "`n[3/3] Downloading all remaining PyPI wheels..." -ForegroundColor Cyan
# Prefer wheels only; if something doesn't have a wheel for your platform, this will fail (by design).
& $Python -m pip download -r "requirements.pypi.lock.txt" -d $WheelDir --only-binary=:all:

Write-Host "`nDone. Wheelhouse is in '$WheelDir'." -ForegroundColor Green
Write-Host "Next: copy the whole project folder (including '$WheelDir') to the offline machine, then run scripts\\install_offline.ps1" -ForegroundColor Green
