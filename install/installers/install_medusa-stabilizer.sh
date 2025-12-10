#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/medusa-stabilizer.conf"

MAKEFILE="$SCRIPT_DIR/../misc/CMakeLists.txt"

log "Installing MEDUSA..."

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

# Build MEDUSA
cd "$DIR"
log "Initializing dependencies (Sylvan, Lace)..."
make init
log "Editing CMakeLists.txt..."
cd lib/sylvan
rm -rf build
rm CMakeLists.txt
cp "$MAKEFILE" .
mkdir -p build
cd build
cmake ..
make
cd "$DIR"
log "Building MEDUSA..."
make
log "MEDUSA installation completed!"
