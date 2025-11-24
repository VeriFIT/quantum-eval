#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/qsylvan.conf"

log "Installing Q-Sylvan..."

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

# Build Qsylvan
cd "$DIR"
log "Building Q-Sylvan..."
mkdir -p build
cd build
cmake ..
make
log "Q-Sylvan installation completed..."

