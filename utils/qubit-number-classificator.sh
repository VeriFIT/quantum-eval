#!/bin/bash

CIRCUITS_ROOT="../circuits"

# Loop over second-level subfolders (measure/no-measure/.../benchmark)
find "${CIRCUITS_ROOT}" -mindepth 2 -maxdepth 2 -type d | while read -r folder; do
    parent_name=$(basename "$(dirname "$folder")")   # measure / no-measure
    folder_name=$(basename "$folder")               # Feynman / BernsteinVazirani
    OUTPUT_FILE="${parent_name}_${folder_name}-classification.csv"

    echo "name;num_qubits" > "$OUTPUT_FILE"

    # Loop over .qasm files in this subfolder
    find "$folder" -type f -name "*.qasm" | while read -r qasm_file; do
        num_qubits=$(grep -oP '(qreg|qubit)\s*(\w+\[|\[)\K[0-9]+' "$qasm_file" | head -n 1)
        num_qubits=${num_qubits:-0}

        echo "${qasm_file};${num_qubits}" >> "$OUTPUT_FILE"
    done

    echo "CSV file generated: $OUTPUT_FILE"
done
