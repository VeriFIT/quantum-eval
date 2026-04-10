#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

QASM_FILE="${1:-}"
if [[ -z "$QASM_FILE" ]]; then
    echo "Usage: $0 <qasm-file>" >&2
    exit 1
fi

QASM_DIR=$(dirname "$QASM_FILE")
QASM_BASE=$(basename "$QASM_FILE")

if [[ "$QASM_BASE" == NL_* ]]; then
    echo "###runtime:NA"
    echo "###memory:NA"
    exit 1
fi

QASM_NL_FILE="$QASM_DIR/NL_${QASM_BASE}"
if [[ -f "$QASM_NL_FILE" ]]; then
    echo "# Using NL variant: $QASM_NL_FILE" >&2
    QASM_FILE="$QASM_NL_FILE"
fi

QASM_FILE_ABS=$(realpath "$QASM_FILE") || {
    echo "###runtime:NA"
    echo "###memory:NA"
    exit 1
}

QUOKKA_DIR="$SCRIPT_DIR/../simulators/quokka"
CONFIG_FILE_ABS=$(realpath "$QUOKKA_DIR/config.json")

cd "$QUOKKA_DIR" || { echo "Error: cannot enter $QUOKKA_DIR" >&2; exit 1; }

set +e
SIM_OUTPUT=$(env QUOKKA_CONFIG="$CONFIG_FILE_ABS" python3 quokka-sim.py "$QASM_FILE_ABS" 2>&1)
EXIT_CODE=$?
set -e

echo "$SIM_OUTPUT"

RUNTIME=$(echo "$SIM_OUTPUT" | grep -oP 'Time:\s*\K[0-9.]+' || true)
MEMORY=$(echo "$SIM_OUTPUT"  | grep -oP 'Peak memory usage:\s*\K[0-9.]+' || true)

[[ -z "$RUNTIME" ]] && RUNTIME="NA"
[[ -z "$MEMORY"  ]] && MEMORY="NA"

echo "###runtime:$RUNTIME"
echo "###memory:$MEMORY"

exit $EXIT_CODE