<#
.SYNOPSIS
    Build the HCFT Windows 64-bit executable and installer.

.DESCRIPTION
    1. Creates a clean build environment (build\windows\venv) from uv.lock
    2. Renders the icons from hcft\resources\hcft_icon.svg
    3. Builds dist\hcft\hcft.exe with PyInstaller
    4. Builds dist\hcft-<version>-win64-setup.exe with Inno Setup

    The version is taken from hcft\version.py.

    Requirements: uv (https://docs.astral.sh/uv/) and Inno Setup 6
    (winget install --id JRSoftware.InnoSetup -e).

.PARAMETER SkipInstaller
    Only build the executable (dist\hcft), not the installer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
#>
param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$BuildDir = Join-Path $Root 'build\windows'
Set-Location $Root

function Invoke-Step([string]$Title, [scriptblock]$Command) {
    Write-Host "`n==> $Title" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE) { throw "'$Title' failed with exit code $LASTEXITCODE" }
}

function Find-InnoSetupCompiler {
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not installed, see https://docs.astral.sh/uv/getting-started/installation/'
}
$Iscc = $null
if (-not $SkipInstaller) {
    $Iscc = Find-InnoSetupCompiler
    if (-not $Iscc) {
        throw 'Inno Setup 6 was not found. Install it with: winget install --id JRSoftware.InnoSetup -e'
    }
}

$Version = [regex]::Match((Get-Content (Join-Path $Root 'hcft\version.py') -Raw),
                          "__version__\s*=\s*['""]([^'""]+)['""]").Groups[1].Value
Write-Host "Building HCFT $Version (Windows 64-bit)" -ForegroundColor Green

# Separate environment for the build, so the dev .venv is never touched and
# the exe contains exactly the locked dependencies.
$env:UV_PROJECT_ENVIRONMENT = Join-Path $BuildDir 'venv'

Invoke-Step 'Syncing build environment from uv.lock' {
    uv sync --locked --group build
}

Invoke-Step 'Rendering icons' {
    uv run --locked --group build python packaging\windows\make_icon.py
}

Invoke-Step 'Building executable with PyInstaller' {
    uv run --locked --group build pyinstaller packaging\windows\hcft.spec `
        --noconfirm --clean --distpath dist --workpath (Join-Path $BuildDir 'pyinstaller')
}

if (-not $SkipInstaller) {
    Invoke-Step 'Building installer with Inno Setup' {
        & $Iscc /Qp "/DMyAppVersion=$Version" packaging\windows\hcft_installer.iss
    }
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  Executable: $(Join-Path $Root 'dist\hcft\hcft.exe')"
if (-not $SkipInstaller) {
    Write-Host "  Installer:  $(Join-Path $Root "dist\hcft-$Version-win64-setup.exe")"
}
