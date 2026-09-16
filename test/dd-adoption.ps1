#requires -Version 7.4
param([string]$BuildDir)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path $PSScriptRoot
$scratch = Join-Path $root 'tmp'
$null = New-Item -ItemType Directory -Force -Path $scratch
$env:TEMP = $scratch
$env:TMP = $scratch
$env:TMPDIR = $scratch
$pwsh = Join-Path $PSHOME $(if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' })

& cmake "-DROOT=$root" "-DBUILD_DIR=$BuildDir" -P (Join-Path $PSScriptRoot 'dd-adoption.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Static adoption checks failed' }

function Invoke-Driver([string[]]$Arguments, [int]$ExpectedExit = 0) {
    $output = & $pwsh -NoProfile -File (Join-Path $root 'dd.ps1') @Arguments --project $root --non-interactive --json
    $code = $LASTEXITCODE
    $result = $output | ConvertFrom-Json -AsHashtable -ErrorAction Stop
    if ($code -ne $ExpectedExit -or $result.exitCode -ne $code -or $result.ok -ne ($code -eq 0)) {
        throw "Unexpected dd status: $output"
    }
    return $result
}
$manifest = Import-PowerShellDataFile (Join-Path $root 'dd.psd1')
$presets = Get-Content (Join-Path $root 'CMakePresets.json') -Raw | ConvertFrom-Json -AsHashtable
if ($manifest.project.type -ne 'library' -or $manifest.project.'default-target' -ne 'mlp-cli' -or
    $manifest.dependencies.owner -ne 'application') { throw 'Unexpected library/CLI ownership contract' }
foreach ($platform in @('x64-windows', 'x64-linux')) {
    foreach ($config in @('debug', 'release')) {
        $name = $manifest.build[$platform][$config]
        $configure = @($presets.configurePresets | Where-Object name -eq $name)
        if ($configure.Count -ne 1 -or $configure[0].binaryDir -ne ('${sourceDir}/build/' + $platform + '/' + $config)) {
            throw "Unexpected $platform $config configure path"
        }
        foreach ($phase in @('buildPresets', 'testPresets')) {
            $preset = @($presets[$phase] | Where-Object name -eq $name)
            if ($preset.Count -ne 1 -or $preset[0].configurePreset -ne $name) { throw "Incorrect $phase reference" }
        }
    }
}
$targets = (Invoke-Driver @('targets')).data.targets
$library = @($targets | Where-Object id -eq 'cmlp')
$cli = @($targets | Where-Object id -eq 'mlp-cli')
if ($targets.Count -ne 2 -or $library.Count -ne 1 -or $cli.Count -ne 1 -or
    $library[0].kind -ne 'library' -or $library[0].runnable -or -not $cli[0].runnable -or
    $library[0].cmakeTarget -ne 'cmlp' -or $cli[0].cmakeTarget -ne 'mlp-cli') {
    throw 'Unexpected native targets'
}
foreach ($target in $manifest.targets) {
    foreach ($config in @('debug', 'release')) {
        $filename = if ($target.kind -eq 'library') { '{libprefix}cmlp{lib}' } else { 'mlp-cli{exe}' }
        if ($target["$config-path"] -ne "build/{platform}/$config/$filename") {
            throw "Incorrect artifact path for $($target.id)"
        }
    }
}
foreach ($verb in @('run', 'launch')) {
    $response = Invoke-Driver @($verb, 'cmlp') 2
    if (($response.errors -join ' ') -notmatch 'library.*no executable') { throw 'Library execution not refused' }
}
$commands = @('scalar')
if ($IsLinux) { $commands += 'asan' }
foreach ($command in $commands) {
    $plan = (Invoke-Driver @($command, '--dry-run')).data.result
    $expected = $(if ($IsWindows) { 'windows-' } else { 'linux-' }) + $command
    if ($plan.preset -ne $expected -or $plan.results.Count) { throw "Incorrect $command dry-run" }
    $null = Invoke-Driver @($command, '--unknown', 'true', '--dry-run') 2
}
$cmake = Get-Content (Join-Path $root 'CMakeLists.txt') -Raw
if ($cmake -notmatch 'LANGUAGES C\)' -or $cmake -match 'platform-h|LANGUAGES CXX|\.cpp') {
    throw 'Library must remain plain C without platform-h'
}
Write-Host 'PASS upstream dd target metadata, native paths, non-runnable library and validation commands'
