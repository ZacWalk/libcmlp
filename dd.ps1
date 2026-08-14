<#
.SYNOPSIS
    Developer driver for the nn Fashion-MNIST classifier.

.DESCRIPTION
    Wraps the MSBuild + run + smoke-test loop so every workflow is a single command.
    See docs/design.md for what the project actually does.

.EXAMPLE
    .\dd.ps1 build
    .\dd.ps1 run
    .\dd.ps1 test
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('build', 'run', 'test')]
    [string]$Command = 'run',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$solutionPath = Join-Path $repoRoot 'nn.sln'
$binaryPath = Join-Path $repoRoot 'bin\nn.exe'

# Smoke-test thresholds. One epoch is enough to catch a broken forward/backward pass.
$testEpochs = 1
$testSeed = 12345
$testMinimumAccuracy = 7500
$testSampleCount = 10000

function Resolve-MSBuild {
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

    if (Test-Path $vswherePath) {
        $installationPath = & $vswherePath -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            foreach ($relative in @('MSBuild\Current\Bin\amd64\MSBuild.exe', 'MSBuild\Current\Bin\MSBuild.exe')) {
                $candidate = Join-Path $installationPath $relative
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }

    $command = Get-Command msbuild.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    throw 'MSBuild.exe was not found. Install Visual Studio Build Tools or add MSBuild to PATH.'
}

function Invoke-Build {
    $msbuildPath = Resolve-MSBuild
    Write-Host "Building Release|x64 with $msbuildPath" -ForegroundColor Cyan
    & $msbuildPath $solutionPath '/p:Configuration=Release' '/p:Platform=x64' '/v:minimal' '/nologo'

    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path $binaryPath)) { throw "Expected executable was not produced: $binaryPath" }
}

# Runs nn.exe from the repo root so the relative dataset paths resolve.
function Invoke-Model {
    param(
        [hashtable]$Environment = @{},
        [switch]$Capture
    )

    $saved = @{}
    foreach ($key in $Environment.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key)
        [Environment]::SetEnvironmentVariable($key, [string]$Environment[$key])
    }

    Push-Location $repoRoot
    try {
        if ($Capture) {
            $output = & $binaryPath 2>&1 | ForEach-Object { $_.ToString() }
        }
        else {
            & $binaryPath @Rest
            $output = @()
        }
        return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $output }
    }
    finally {
        Pop-Location
        foreach ($key in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($key, $saved[$key])
        }
    }
}

$script:failures = @()

function Test-Case {
    param([string]$Name, [scriptblock]$Body)

    Write-Host "  $Name ... " -NoNewline
    try {
        & $Body
        Write-Host 'PASS' -ForegroundColor Green
    }
    catch {
        Write-Host 'FAIL' -ForegroundColor Red
        Write-Host "    $($_.Exception.Message)" -ForegroundColor Red
        $script:failures += $Name
    }
}

function Get-EvaluationLine {
    param([string[]]$Output)

    $line = $Output | Where-Object { $_ -match '\[EVALUATION\]' } | Select-Object -First 1
    if (-not $line) { throw "No [EVALUATION] line in output:`n$($Output -join "`n")" }
    return $line.Trim()
}

function Invoke-Test {
    Write-Host 'Running smoke tests' -ForegroundColor Cyan

    $baseline = @{ NN_EPOCHS = $testEpochs; NN_SEED = $testSeed }
    $first = Invoke-Model -Environment $baseline -Capture

    Test-Case 'model exits cleanly' {
        if ($first.ExitCode -ne 0) { throw "Exit code was $($first.ExitCode)" }
    }

    Test-Case 'datasets load' {
        if (-not ($first.Output | Where-Object { $_ -match 'out of 60000' })) {
            throw 'Training set did not report 60000 samples'
        }
    }

    Test-Case "evaluation accuracy >= $testMinimumAccuracy / $testSampleCount after $testEpochs epoch(s)" {
        $line = Get-EvaluationLine $first.Output
        if ($line -notmatch '\[ACCURACY\s+(\d+)\s+out of\s+(\d+)\]') { throw "Unparsable evaluation line: $line" }
        $correct = [int]$Matches[1]
        $total = [int]$Matches[2]
        if ($total -ne $testSampleCount) { throw "Expected $testSampleCount evaluation samples, saw $total" }
        if ($correct -lt $testMinimumAccuracy) { throw "Accuracy $correct / $total is below the $testMinimumAccuracy floor" }
        Write-Host "($correct / $total) " -NoNewline -ForegroundColor DarkGray
    }

    Test-Case 'seeded runs are reproducible' {
        $second = Invoke-Model -Environment $baseline -Capture
        $expected = Get-EvaluationLine $first.Output
        $actual = Get-EvaluationLine $second.Output
        if ($expected -ne $actual) { throw "Seeded runs diverged:`n  $expected`n  $actual" }
    }

    Test-Case 'invalid topology is rejected' {
        $invalid = Invoke-Model -Environment @{ NN_EPOCHS = 1; NN_HIDDEN1 = 0 } -Capture
        if ($invalid.ExitCode -eq 0) { throw 'A zero-width hidden layer was accepted' }
    }

    if ($script:failures.Count -gt 0) {
        Write-Host "`n$($script:failures.Count) test(s) failed." -ForegroundColor Red
        exit 1
    }

    Write-Host "`nAll tests passed." -ForegroundColor Green
}

switch ($Command) {
    'build' {
        Invoke-Build
    }
    'run' {
        Invoke-Build
        Write-Host "Running $binaryPath" -ForegroundColor Cyan
        $result = Invoke-Model
        exit $result.ExitCode
    }
    'test' {
        Invoke-Build
        Invoke-Test
    }
}
