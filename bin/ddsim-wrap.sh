#!/bin/bash

ABSOLUTE_SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${ABSOLUTE_SCRIPT_PATH}")

QASM_FILE="$1"

# finding NL (no-loop) version as ddsim does not support for {...} syntax
QASM_DIR=$(dirname "$QASM_FILE")
QASM_BASE=$(basename "$QASM_FILE")

# skip if the file is NL already
if [[ "$QASM_BASE" == NL_* ]]; then
    echo "###runtime:NA"
    echo "###memory:NA"
    exit 1
fi

QASM_NL_FILE="$QASM_DIR/NL_${QASM_BASE}"

# if such file exists, use it
if [ -f "$QASM_NL_FILE" ]; then
    QASM_FILE="$QASM_NL_FILE"
fi

DDSIM_EXE="${SCRIPT_DIR}/mqt-ddsim-simple" # TODO Change executable
DDSIM_OUT=$("$DDSIM_EXE" --ps --simulate_file "$QASM_FILE" 2>&1)

EXIT_CODE=$?

RUNTIME=$(echo "$DDSIM_OUT" | grep -oP '"simulation_time":\s*\K[0-9.e+-]+')

MEMORY_BYTES=$(echo "$DDSIM_OUT" | grep -oP 'Peak memory usage:\s*\K[0-9]+' || echo "NA")

if [ "$MEMORY_BYTES" != "NA" ]; then
    MEMORY_KB=$(awk "BEGIN {print int($MEMORY_BYTES/1024)}")
else
    MEMORY_KB="NA"
fi

# Print in pycobench format
echo "###runtime:$RUNTIME"
echo "###memory:$MEMORY_KB"

exit $EXIT_CODE