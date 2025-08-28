#!/bin/bash

ABSOLUTE_SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${ABSOLUTE_SCRIPT_PATH}")

QASM_FILE="$1"

SLIQSIM_EXE="${SCRIPT_DIR}/SliQSim" # TODO Change executable

SLIQSIM_OUT=$("$SLIQSIM_EXE" --print_info --type 1 --sim_qasm "$QASM_FILE" 2>&1)

EXIT_CODE=$?

RUNTIME=$(echo "$SLIQSIM_OUT" | grep -oP 'Runtime:\s*\K[0-9.]+' || echo "NA")

MEMORY_BYTES=$(echo "$SLIQSIM_OUT" | grep -oP 'Peak memory usage:\s*\K[0-9]+' || echo "NA")

if [ "$MEMORY_BYTES" != "NA" ]; then
    MEMORY_KB=$(awk "BEGIN {print int($MEMORY_BYTES/1024)}")
else
    MEMORY_KB="NA"
fi

# Print in pycobench format
echo "###runtime:$RUNTIME"
echo "###memory:$MEMORY_KB"

exit $EXIT_CODE