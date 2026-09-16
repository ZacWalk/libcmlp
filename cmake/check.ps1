#requires -Version 7.4
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$request = [Console]::In.ReadLine() | ConvertFrom-Json -AsHashtable -ErrorAction Stop
$root = Split-Path $PSScriptRoot
if ($request.schema -ne 1 -or $request.command -notin @('scalar', 'asan') -or
    $request.projectRoot -ne $root -or $request.dryRun -isnot [bool] -or
    $request.parameters -isnot [hashtable] -or $request.parameters.Count) {
    throw 'Expected a schema 1 scalar/asan request with no parameters.'
}
. (Join-Path $root '.dd/core.ps1')
foreach ($module in Get-DDRuntimeModules) { . (Join-Path $root ".dd/$module") }
$platform = Get-DDPlatform
if ($request.command -eq 'asan' -and $platform -ne 'x64-linux') {
    throw 'ASan/UBSan validation requires Linux.'
}
$configuration = $request.command
$preset = $(if ($IsWindows) { 'windows-' } else { 'linux-' }) + $configuration
$results = @()
if (-not $request.dryRun) {
    $scratch = Join-Path $root 'tmp'
    $null = New-Item -ItemType Directory -Force -Path $scratch
    $env:TEMP = $scratch
    $env:TMP = $scratch
    $env:TMPDIR = $scratch
    $manifest = Read-DDManifest $root -ForBuild
    Assert-DDDependencies $root
    $manifest.build[$platform][$configuration] = $preset
    $results += Invoke-DDBuild $root $manifest $configuration @{}
    $results += Invoke-DDTests $root $manifest $configuration @{ label = '^ci$' } @()
}
@{ schema = 1; data = @{ preset = $preset; results = $results }; files = @("build/$platform/$configuration") } |
    ConvertTo-Json -Depth 15 -Compress | Write-Output
