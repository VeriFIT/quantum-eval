#!/usr/bin/env python3
import os
import sys

def generate_input_file(base_dir="../circuits/dirac-grover", output_file="dirac-grover.input"):
    lines = []

    for subdir in sorted(os.listdir(base_dir)):
        full_subdir = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue

        circuit_dir = os.path.join(full_subdir, "circuit")
        pre_dir = os.path.join(full_subdir, "pre")

        if not os.path.isdir(circuit_dir) or not os.path.isdir(pre_dir):
            print(f"Skipping {subdir} (missing circuit/ or pre/)")
            continue

        circuits = sorted(
            [os.path.join(circuit_dir, f) for f in os.listdir(circuit_dir) if f.endswith(".qasm")]
        )

        prefiles = sorted(
            [os.path.join(pre_dir, f) for f in os.listdir(pre_dir) if f.endswith(".hsl")]
        )
        for circuit in circuits:
            for prefile in prefiles:
                lines.append(f"{circuit};{prefile}")


    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {output_file} with {len(lines)} combinations.")


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "../circuits/dirac-grover"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dirac-grover.input"
    generate_input_file(base_dir, output_file)
