<#
.SYNOPSIS
    Developer driver for the nn Fashion-MNIST classifier on Windows.

.DESCRIPTION
    Wraps the CMake + Ninja + ctest loop so every workflow is a single command. The Linux
    equivalent is dd.sh. See docs/design.md for what the project actually does.

.EXAMPLE
    .\dd.ps1 build
    .\dd.ps1 run
    .\dd.ps1 test
    .\dd.ps1 test -Label ci
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('build', 'run', 'test', 'clean')]
    [string]$Command = 'run',

    [ValidateSet('release', 'debug')]
    [string]$Config = 'release',

    [string]$Label,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$preset = "windows-$Config"
$buildDir = Join-Path $repoRoot "build\$preset"
$binaryPath = Join-Path $buildDir 'nn.exe'

function Get-VisualStudioPath {
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswherePath)) { return $null }

    $installationPath = & $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installationPath) { return $null }
    return $installationPath
}

# Ninja needs cl.exe on PATH, so import the Visual Studio environment into this session.
function Enter-DeveloperEnvironment {
    if ($env:VSCMD_VER) { return }

    $installationPath = Get-VisualStudioPath
    if (-not $installationPath) {
        throw 'Visual Studio with the C++ toolset was not found. Install Visual Studio Build Tools.'
    }

    $devShellModule = Join-Path $installationPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    if (-not (Test-Path $devShellModule)) {
        throw "The Visual Studio developer shell module was not found: $devShellModule"
    }

    Import-Module $devShellModule
    Enter-VsDevShell -VsInstallPath $installationPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}

function Resolve-Tool {
    param([string]$Name, [string]$Fallback)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $installationPath = Get-VisualStudioPath
    if ($installationPath) {
        $candidate = Join-Path $installationPath $Fallback
        if (Test-Path $candidate) { return $candidate }
    }

    throw "$Name was not found on PATH or in the Visual Studio installation."
}

function Get-CMakePath {
    return Resolve-Tool 'cmake.exe' 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
}

function Invoke-Build {
    Enter-DeveloperEnvironment

    $cmake = Get-CMakePath
    $ninja = Resolve-Tool 'ninja.exe' 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'

    Write-Host "Configuring and building preset $preset" -ForegroundColor Cyan
    & $cmake --preset $preset "-DCMAKE_MAKE_PROGRAM=$ninja"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }

    & $cmake --build --preset $preset
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path $binaryPath)) { throw "Expected executable was not produced: $binaryPath" }
}

switch ($Command) {
    'clean' {
        if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
        Write-Host "Removed $buildDir" -ForegroundColor Cyan
    }
    'build' {
        Invoke-Build
    }
    'run' {
        Invoke-Build
        Write-Host "Running $binaryPath" -ForegroundColor Cyan
        # The default dataset paths are relative, so the model runs from the repository root.
        Push-Location $repoRoot
        try { & $binaryPath @Rest }
        finally { Pop-Location }
        exit $LASTEXITCODE
    }
    'test' {
        Invoke-Build
        $ctest = Join-Path (Split-Path -Parent (Get-CMakePath)) 'ctest.exe'
        $arguments = @('--preset', $preset)
        if ($Label) { $arguments += @('-L', $Label) }

        & $ctest @arguments
        exit $LASTEXITCODE
    }
}
