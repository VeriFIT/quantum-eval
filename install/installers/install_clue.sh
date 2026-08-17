#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/clue.conf"

log "Installing CLUE..."

# System dependencies
for dep in "${DEPS[@]}"; do
    if ! dpkg -s "$dep" >/dev/null 2>&1; then
        log "Installing $dep"
        sudo apt-get install -y "$dep"
    else
        log "$dep is already installed"
    fi
done

# Clone repository if it doesn't exist, otherwise update it
if [ ! -d "$DIR" ]; then
    log "Cloning $REPO into $DIR"
    git clone --branch "$BRANCH" --recurse-submodules "$REPO" "$DIR"
else
    log "Repo already exists at $DIR"
    git -C "$DIR" fetch origin
    git -C "$DIR" checkout "$BRANCH"
    git -C "$DIR" pull origin "$BRANCH"
    git -C "$DIR" submodule update --init --recursive
fi

cd "$DIR/cpp/clue"

# Configure
log "Configuring CLUE"
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build

# Build QASM executable
log "Building CLUE_main_QASM"
cmake --build build --config Release --target CLUE_main_QASM

log "CLUE installation completed!"