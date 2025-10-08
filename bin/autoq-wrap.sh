#!/bin/bash
# autoq-wrap.sh
# Usage: ./autoq-wrap.sh <pre_file> <qasm_file>

PRE_FILE="$1"
QASM_FILE="$2"
AUTOQ_EXE="/../simulators/autoq/build/cli/autoq"

if [[ ! -f "$PRE_FILE" || ! -f "$QASM_FILE" ]]; then
    echo "Usage: $0 <pre_file> <qasm_file>"
    exit 1
fi

TIME_OUTPUT=$( { /usr/bin/time -f "%e %M" "$AUTOQ_EXE" "$PRE_FILE" "$QASM_FILE" 1>/dev/null; } 2>&1 )

# Extract runtime and memory
RUNTIME=$(echo "$TIME_OUTPUT" | awk '{print $1}')
MEMORY_KB=$(echo "$TIME_OUTPUT" | awk '{print $2}')

echo "###runtime:$RUNTIME"
echo "###memory:$MEMORY_KB"
