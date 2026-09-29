<#
.SYNOPSIS
    Publish the built HCFT installer as a GitHub release.

.DESCRIPTION
    1. Checks the installer dist\hcft-<version>-win64-setup.exe exists
    2. Creates and pushes the tag v<version> if it doesn't exist yet
    3. Creates the GitHub release v<version> with the installer attached
       (the release notes are generated from the commits since the last release)

    The version is taken from hcft\version.py. Run build.ps1 first.

    Requirements: GitHub CLI (winget install --id GitHub.cli -e), logged in
    once with: gh auth login

.PARAMETER Draft
    Create the release as a draft, to check it on GitHub before publishing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\windows\release.ps1
#>
param(
    [switch]$Draft
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $Root

function Invoke-Step([string]$Title, [scriptblock]$Command) {
    Write-Host "`n==> $Title" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE) { throw "'$Title' failed with exit code $LASTEXITCODE" }
}

function Find-GitHubCli {
    $command = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        "$env:ProgramFiles\GitHub CLI\gh.exe",
        "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe"
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

$Gh = Find-GitHubCli
if (-not $Gh) {
    throw 'GitHub CLI was not found. Install it with: winget install --id GitHub.cli -e'
}
& $Gh auth status *> $null
if ($LASTEXITCODE) { throw 'GitHub CLI is not logged in, run: gh auth login' }

$Version = [regex]::Match((Get-Content (Join-Path $Root 'hcft\version.py') -Raw),
                          "__version__\s*=\s*['""]([^'""]+)['""]").Groups[1].Value
$Tag = "v$Version"
$Installer = Join-Path $Root "dist\hcft-$Version-win64-setup.exe"
if (-not (Test-Path $Installer)) {
    throw "Installer $Installer not found, build it first with build.ps1"
}
Write-Host "Releasing HCFT $Tag" -ForegroundColor Green

# The tag has to exist on GitHub for the release, it's created on the current commit if it doesn't exist yet
git rev-parse -q --verify "refs/tags/$Tag" *> $null
if ($LASTEXITCODE) {
    if (git status --porcelain) {
        throw "There are uncommitted changes, commit them first so tag $Tag contains them"
    }
    Invoke-Step "Creating tag $Tag" { git tag $Tag }
}
else {
    Write-Host "Tag $Tag already exists on $(git rev-parse --short "$Tag^{commit}")"
}
Invoke-Step "Pushing tag $Tag" { git push origin $Tag }

$ReleaseArgs = @('release', 'create', $Tag, "$Installer#HCFT $Version Windows 64-bit installer",
                 '--title', $Tag, '--generate-notes', '--verify-tag')
if ($Draft) { $ReleaseArgs += '--draft' }
Invoke-Step 'Creating GitHub release' { & $Gh @ReleaseArgs }

Write-Host "`nDone." -ForegroundColor Green
