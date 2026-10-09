# Build Fission for Windows (x64): a portable folder/zip and a normal installer (Setup.exe).
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
#     -> dist\Fission-<version>-Windows-x64-Setup.exe   and   dist\Fission-<version>-Windows-x64-portable.zip
#
# Needs: Windows 10/11 x64, Python 3.11-3.13 from python.org (the "py" launcher), internet for the first pip install,
#        Inno Setup 6 for the installer (winget install JRSoftware.InnoSetup  or  choco install innosetup).
# Optional code signing: set SIGN_PFX (path to a .pfx) and SIGN_PASSWORD before running.
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
$Src = if ($env:FISSION_SRC) { $env:FISSION_SRC } else { Join-Path $Root "fission.py" }
$Version = $env:FISSION_VERSION
if (-not $Version) {
  $m = Select-String -Path $Src -Pattern 'Fission (\d+(\.\d+)+)' | Select-Object -First 1
  $Version = if ($m) { $m.Matches[0].Groups[1].Value } else { "0.0" }
}
Write-Host "==> Fission $Version for Windows x64"

if (-not (Test-Path ".venv-win")) {
  $py = if ($env:PYTHON) { $env:PYTHON } else { "py" }
  if ($py -eq "py") { & py -3.12 -m venv .venv-win; if ($LASTEXITCODE) { & py -3 -m venv .venv-win } } else { & $py -m venv .venv-win }
}
$Python = Join-Path $Root ".venv-win\Scripts\python.exe"
& $Python -m pip install -q --upgrade pip
& $Python -m pip install -q -r packaging\requirements-build.txt
if ($LASTEXITCODE) { throw "pip install failed" }

& $Python packaging\make_icons.py $Src
$env:FISSION_SRC = $Src; $env:FISSION_VERSION = $Version
& $Python -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\fission.spec
if ($LASTEXITCODE) { throw "PyInstaller failed" }

Write-Host "==> self-test"
New-Item -ItemType Directory -Force build | Out-Null
$st = Join-Path $Root "build\selftest.txt"; Remove-Item $st -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "dist\Fission\Fission.exe" -ArgumentList "--selftest", "`"$st`"" -Wait -PassThru
Get-Content $st
if ($p.ExitCode -ne 0) { throw "self-test failed" }

if ($env:SIGN_PFX) {
  Write-Host "==> signing"
  $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" | Sort-Object FullName | Select-Object -Last 1
  & $signtool.FullName sign /f $env:SIGN_PFX /p $env:SIGN_PASSWORD /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 "dist\Fission\Fission.exe"
}

Write-Host "==> portable zip"
$zip = "dist\Fission-$Version-Windows-x64-portable.zip"; Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "dist\Fission" -DestinationPath $zip -CompressionLevel Optimal

$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
  Write-Host "==> installer"
  & $iscc /Qp "/DAppVersion=$Version" "/DSourceDir=$Root\dist\Fission" "/DOutDir=$Root\dist" "/DAssets=$Root\packaging\assets" packaging\windows\fission.iss
  if ($LASTEXITCODE) { throw "Inno Setup failed" }
  if ($env:SIGN_PFX) { & $signtool.FullName sign /f $env:SIGN_PFX /p $env:SIGN_PASSWORD /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 "dist\Fission-$Version-Windows-x64-Setup.exe" }
} else {
  Write-Warning "Inno Setup 6 not found - skipped the installer (the portable zip is ready). Install it with: winget install JRSoftware.InnoSetup"
}
Write-Host "==> done"; Get-ChildItem dist -File | ForEach-Object { Write-Host "   $($_.Name)  $([math]::Round($_.Length/1MB)) MB" }
