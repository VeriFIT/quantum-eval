#!/bin/bash
# Shared installer body for MoToBuddy PR #24 builds (inline / inline+LTO).
# Sourced by install_medusa-motobuddy-inline{,-lto}.sh after loading a conf.

if [[ -z "${DIR:-}" || -z "${REPO:-}" ]]; then
    error "install_medusa-motobuddy-opt.sh requires REPO and DIR from a conf file"
    exit 1
fi

REF="${REF:-perf/inline-hot-wrappers}"
LTO="${LTO:-0}"

log "Installing MEDUSA MoToBuddy (ref=$REF, LTO=$LTO) into $DIR"

for dep in "${DEPS[@]}"; do
    if ! dpkg -s "$dep" >/dev/null 2>&1; then
        log "Installing $dep"
        sudo apt-get install -y "$dep"
    else
        log "$dep is already installed"
    fi
done

clone_repo "$REPO" "$DIR" "$REF"

cd "$DIR"
log "Initializing MoToBuddy dependency (make init)..."
make init

log "Building buddy_gmp (LTO=$LTO)..."
make buddy_gmp "LTO=$LTO"

log "Building buddy_doubles_all (f32/f64/f80/f128, LTO=$LTO)..."
make buddy_doubles_all "LTO=$LTO"

write_build_info "$DIR" "$LTO"

log "MEDUSA MoToBuddy install completed: $DIR"
ls -la "$DIR"/MEDUSA_buddy_gmp "$DIR"/MEDUSA_buddy_doubles_* 2>/dev/null || true
