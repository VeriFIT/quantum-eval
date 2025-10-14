#!/usr/bin/env python3
import os
import sys

def generate_dirac_file(num_qubits: int, zeros: int, const_name="c1", outdir="."):
    s_len = num_qubits - zeros
    suffix = "0" * zeros
    num_vectors = 2 ** s_len

    if s_len == 0:
        pattern = f"{{{const_name} |{suffix}>}}"
    else:
        pattern = f"{{{const_name} |s{suffix}> : |s|={s_len}}}"

    contents = "\n".join([
        "Constants",
        f"{const_name} := 1",
        "Extended Dirac",
        pattern
    ])

    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"dirac_{num_qubits}q_{num_vectors}v.hsl")
    with open(filename, "w") as f:
        f.write(contents + "\n")

    print(f"Generated {filename}")


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_dirac_files.py <num_qubits> [output_dir]")
        sys.exit(1)

    num_qubits = int(sys.argv[1])
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."

    for zeros in range(1, num_qubits + 1):
        generate_dirac_file(num_qubits, zeros, outdir=outdir)


if __name__ == "__main__":
    main()
