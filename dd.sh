#!/usr/bin/env bash
#
# Developer driver for the nn Fashion-MNIST classifier on Linux and WSL.
# The Windows equivalent is dd.ps1. See docs/design.md for what the project actually does.
#
#   ./dd.sh build
#   ./dd.sh run
#   ./dd.sh test
#   ./dd.sh test ci        # only the label 'ci' tests, which need no dataset

set -euo pipefail

command=${1:-run}
label=${2:-}

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config=${NN_CONFIG:-release}
preset="linux-${config}"
build_dir="${repo_root}/build/${preset}"
binary="${build_dir}/nn"

require() {
    command -v "$1" >/dev/null 2>&1 || { echo "error: $1 is not installed" >&2; exit 1; }
}

build() {
    require cmake
    require ninja
    echo "Configuring and building preset ${preset}"
    cmake --preset "${preset}"
    cmake --build --preset "${preset}"
    [ -x "${binary}" ] || { echo "error: expected executable was not produced: ${binary}" >&2; exit 1; }
}

case "${command}" in
    clean)
        rm -rf "${build_dir}"
        echo "Removed ${build_dir}"
        ;;
    build)
        build
        ;;
    run)
        build
        echo "Running ${binary}"
        # The default dataset paths are relative, so the model runs from the repository root.
        cd "${repo_root}"
        exec "${binary}"
        ;;
    test)
        build
        if [ -n "${label}" ]; then
            ctest --preset "${preset}" -L "${label}"
        else
            ctest --preset "${preset}"
        fi
        ;;
    *)
        echo "usage: $0 {build|run|test|clean} [ctest-label]" >&2
        exit 1
        ;;
esac
