@{
    schema = 1
    project = @{
        name = 'libcmlp'
        type = 'library'
        'default-target' = 'mlp-cli'
    }
    dependencies = @{ owner = 'application' }
    build = @{
        'x64-windows' = @{ debug = 'windows-debug'; release = 'windows-release' }
        'x64-linux' = @{ debug = 'linux-debug'; release = 'linux-release' }
    }
    targets = @(
        @{
            id = 'cmlp'
            kind = 'library'
            'cmake-target' = 'cmlp'
            'test-label' = 'cmlp'
            'debug-path' = 'build/{platform}/debug/{libprefix}cmlp{lib}'
            'release-path' = 'build/{platform}/release/{libprefix}cmlp{lib}'
        },
        @{
            id = 'mlp-cli'
            kind = 'cli'
            'cmake-target' = 'mlp-cli'
            'test-label' = 'cmlp'
            'debug-path' = 'build/{platform}/debug/mlp-cli{exe}'
            'release-path' = 'build/{platform}/release/mlp-cli{exe}'
        }
    )
    commands = @{
        scalar = @{
            description = 'Build and run the six offline CI tests without AVX2.'
            script = 'cmake/check.ps1'
            effects = 'write'
            'supports-dry-run' = $true
            'timeout-secs' = 7200
            parameters = @{}
        }
        asan = @{
            description = 'Run the six offline CI tests with ASan and fatal UBSan on Linux.'
            script = 'cmake/check.ps1'
            platforms = @('x64-linux')
            effects = 'write'
            'supports-dry-run' = $true
            'timeout-secs' = 7200
            parameters = @{}
        }
    }
}
