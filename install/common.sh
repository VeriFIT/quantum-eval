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
    if [ ! -d "$target_dir" ]; then
        log "Cloning $repo_url into $target_dir"
        git clone "$repo_url" "$target_dir"
    else
        log "Repo already exists at $target_dir, pulling latest changes"
        git -C "$target_dir" pull
    fi
}
