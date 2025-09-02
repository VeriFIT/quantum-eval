#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/sliqsim.conf"

log "Installing SLIQSIM..."

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

# Build SLIQSIM
cd "$DIR"
log Configuring CUDD...
cd cudd
./configure --enable-dddmp --enable-obj --enable-shared --enable-static 
cd ..
log Making SLIQSIM...
make
log "SLIQSIM installation completed!"
