#!/bin/bash

COMBINED_FILE="../bench/no-measure_SL-all.input"

# empty
> "$COMBINED_FILE"

INPUT_FILES=("../bench/no-measure_BernsteinVazirani.input" "../bench/no-measure_Feynman.input" "../bench/no-measure_MCToffoli.input" "../bench/no-measure/ModifiedRevLib.input" "../bench/no-measure_MOGrover.input" "../bench/no-measure_Random.input" "../bench/no-measure_RevLib.input")

for f in "${INPUT_FILES[@]}"; do
    cat "$f" >> "$COMBINED_FILE"
done

# remove duplicate lines
sort -u "$COMBINED_FILE" -o "$COMBINED_FILE"

echo "Combined ${#INPUT_FILES[@]} files into $COMBINED_FILE"
