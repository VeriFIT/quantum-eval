#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/quokka.conf"

log "Installing Quokka..."

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

# Clone GPMC
cd "$DIR"
log "Cloning GPMC..."
clone_repo https://github.com/System-Verification-Lab/GPMC.git ./GPMC

log "Building GPMC..."
cd GPMC
./build.sh r

log "GPMC built..."
cd "$DIR"

# exporting config
export QUOKKA_CONFIG=./config.json

log "Quokka-sim ready to use..."
