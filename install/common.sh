#!/bin/bash

log() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

check_command() {
    command -v "$1" >/dev/null 2>&1 || {
        error "Command $1 not found. Please install it."
        exit 1
    }
}

install_dependencies() {
        sudo apt-get update && sudo apt-get install -y "${@:1}"
}

clone_repo() {
    local repo_url="$1"
    local target_dir="$2"
    local ref="${3:-}"
    if [ ! -d "$target_dir" ]; then
        log "Cloning $repo_url into $target_dir"
        if [ -n "$ref" ]; then
            git clone --branch "$ref" "$repo_url" "$target_dir" \
                || git clone "$repo_url" "$target_dir"
        else
            git clone "$repo_url" "$target_dir"
        fi
    else
        log "Repo already exists at $target_dir, fetching updates"
        git -C "$target_dir" fetch --all --tags
    fi
    if [ -n "$ref" ]; then
        log "Checking out ref: $ref"
        git -C "$target_dir" checkout --force "$ref"
        # Update branch tip if ref is a branch; no-op for a detached SHA.
        git -C "$target_dir" pull --ff-only 2>/dev/null || true
    elif [ -d "$target_dir/.git" ]; then
        git -C "$target_dir" pull --ff-only 2>/dev/null || true
    fi
}

write_build_info() {
    local target_dir="$1"
    local lto="${2:-0}"
    {
        echo "repo=$(git -C "$target_dir" remote get-url origin 2>/dev/null || echo unknown)"
        echo "ref=$(git -C "$target_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        echo "sha=$(git -C "$target_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "LTO=$lto"
        echo "built_at=$(date -Iseconds)"
        echo "gcc=$(gcc -dumpversion 2>/dev/null || echo unknown)"
    } >"$target_dir/BUILD_INFO.txt"
    log "Wrote $target_dir/BUILD_INFO.txt"
}
