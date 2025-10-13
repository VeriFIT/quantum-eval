#!/bin/bash
set -e
BASE_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$BASE_DIR/common.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <simulator-name> [--all]"
    echo "Available: medusa-sylvan, medusa-motobuddy, sliqsim, ddsim, quasimodo, autoq, dirac-ddsim, quokka"
    exit 1
fi

if [[ "$1" == "--all" ]]; then
    for sim in medusa-sylvan medusa-motobuddy sliqsim ddsim quasimodo autoq dirac-ddsim quokka; do
        bash "$BASE_DIR/installers/install_${sim}.sh"
    done
else
    sim="$1"
    bash "$BASE_DIR/installers/install_${sim}.sh"
fi
