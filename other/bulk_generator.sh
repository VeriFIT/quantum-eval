#!/bin/bash
set -euo pipefail

# Path to your Grover generator Python script
GEN_SCRIPT="./generator_GR_noloop.py"   # change if needed
OUT_DIR="LP-Grover"

mkdir -p "$OUT_DIR"

for n in $(seq 30 50); do
    OUT_FILE="$OUT_DIR/NL_$n.qasm"
    echo "Generating $OUT_FILE ..."
    python3 "$GEN_SCRIPT" "$n" > "$OUT_FILE"
done

echo "Done generating LP-Grover circuits (n = 30-50)"
