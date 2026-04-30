#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

QASM_FILE="${1:-}"
USE_CB="${2:-}"   # second argument: if non-empty, use computational basis

if [[ -z "$QASM_FILE" ]]; then
    echo "Usage: $0 <qasm-file> [cb]" >&2
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

QUASM_QUOKKA_FILE="$QASM_DIR/../quokka-edited/${QASM_BASE}"
if [[ -f "$QUASM_QUOKKA_FILE" ]]; then
    QASM_FILE="$QUASM_QUOKKA_FILE"
fi

QUASM_QUOKKA_FILE="$QASM_DIR/../no-mcx/${QASM_BASE}"
if [[ -f "$QUASM_QUOKKA_FILE" ]]; then
    QASM_FILE="$QUASM_QUOKKA_FILE"
fi

QASM_FILE_ABS=$(realpath "$QASM_FILE") || {
    echo "###runtime:NA"
    echo "###memory:NA"
    exit 1
}

QUOKKA_DIR="$SCRIPT_DIR/../simulators/quokka"
CONFIG_FILE_ABS=$(realpath "$QUOKKA_DIR/config.json")

cd "$QUOKKA_DIR" || { echo "Error: cannot enter $QUOKKA_DIR" >&2; exit 1; }

# build python args based on flag
PY_ARGS=()
if [[ -n "$USE_CB" ]]; then
    # second arg present -- use computational basis
    PY_ARGS+=(--computational_basis)
fi
PY_ARGS+=("$QASM_FILE_ABS")

set +e
SIM_OUTPUT=$(env QUOKKA_CONFIG="$CONFIG_FILE_ABS" python3 quokka-sim.py "${PY_ARGS[@]}" 2>&1)
EXIT_CODE=$?
set -e

RUNTIME=$(echo "$SIM_OUTPUT" | grep -oP 'Time:\s*\K[0-9.]+' || true)
MEMORY=$(echo "$SIM_OUTPUT"  | grep -oP 'Peak memory usage:\s*\K[0-9.]+' || true)

[[ -z "$RUNTIME" ]] && RUNTIME="NA"
[[ -z "$MEMORY"  ]] && MEMORY="NA"

echo "###runtime:$RUNTIME"
echo "###memory:$MEMORY"

exit $EXIT_CODE