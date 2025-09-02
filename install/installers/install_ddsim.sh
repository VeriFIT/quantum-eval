#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/ddsim.conf"

log "Installing DDSIM..."

# Check dependencies
for dep in "${DEPS[@]}"; do
    if ! dpkg -s "$dep" >/dev/null 2>&1; then
        log "Installing $dep"
        sudo apt-get install -y "$dep"
    else
        log "$dep is already installed"
    fi
done

# Clone repo
clone_repo "$REPO" "$DIR"

# Build DDSIM
cd "$DIR"
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --config Release --target mqt-ddsim-simple
log "DDSIM installation completed!"
