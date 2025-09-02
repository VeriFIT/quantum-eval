#!/bin/bash
set -e

BASE_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$BASE_DIR/common.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <simulator-name> [--all]"
    echo "Available: medusa-sylvan, medusa-motobuddy, sliqsim, ddsim, quasimodo"
    exit 1
fi

uninstall_simulator() {
    local sim="$1"
    SIM_DIR="$BASE_DIR/../simulators/$sim"
    if [[ -d "$SIM_DIR" ]]; then
        echo "Removing $sim folder..."
        rm -rf "$SIM_DIR"
        echo "$sim uninstalled successfully."
    else
        echo "$sim not found, skipping."
    fi
}

if [[ "$1" == "--all" ]]; then
    for sim in medusa-sylvan medusa-motobuddy sliqsim ddsim quasimodo; do
        uninstall_simulator "$sim"
    done
else
    uninstall_simulator "$1"
fi
