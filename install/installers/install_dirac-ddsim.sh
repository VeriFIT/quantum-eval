#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/dirac-ddsim.conf"

log "Installing Dirac DDSIM..."

# Check dependencies
for dep in "${DEPS[@]}"; do
    if ! dpkg -s "$dep" >/dev/null 2>&1; then
        log "Installing $dep"
        sudo apt-get install -y "$dep"
    else
        log "$dep is already installed"
    fi
done

for pipdep in "${PIPDEPS[@]}"; do
    pip install "$pipdep"
done

# Clone repo
clone_repo "$REPO" "$DIR"

log "DIRAC DDSIM ready"
