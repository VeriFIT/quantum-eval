#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/medusa-motobuddy-inline-lto.conf"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/install_medusa-motobuddy-opt.sh"
