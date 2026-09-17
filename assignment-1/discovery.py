import csv
import io
import zipfile
from pathlib import Path

import yaml

root = Path("datasets/student-bundle/core/raw")

print("=== Source sizes ===")
for path in sorted(root.rglob("*")):
    if path.is_file():
        print(f"{path.relative_to(root)}\t{path.stat().st_size} bytes")

print("\n=== Syndrome CSV sample ===")
syndromes_zip = root / "source=qec_syndromes" / "syndromes_dataset.zip"
with zipfile.ZipFile(syndromes_zip) as z:
    csv_name = next(name for name in z.namelist() if name.endswith(".csv"))
    text = z.read(csv_name).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))

    print("CSV file:", csv_name)
    print("row count:", len(rows))
    print("columns:", list(rows[0].keys()))
    print("first row:", rows[0])
    print("data types:", {k: type(v).__name__ for k, v in rows[0].items()})

print("\n=== Google experiment discovery ===")
google_zip = root / "source=google_qec" / "google-surface-code-curated.zip"
with zipfile.ZipFile(google_zip) as z:
    experiments = sorted(
        {name.split("/", 1)[0] for name in z.namelist() if "/" in name and name.endswith("properties.yml")}
    )
    print("experiments:", experiments)

    exp = experiments[0]
    props = yaml.safe_load(z.read(f"{exp}/properties.yml"))
    print("properties keys:", list(props.keys()))
    print("properties sample:", {k: props[k] for k in list(props)[:10]})

    for file_name in [
        f"{exp}/measurements.b8",
        f"{exp}/sweep.b8",
        f"{exp}/detection_events.b8",
        f"{exp}/obs_flips_actual.01",
    ]:
        if file_name in z.namelist():
            data = z.read(file_name)
            print(file_name, "size:", len(data), "bytes")
            if file_name.endswith(".01"):
                print("line count:", len(data.splitlines()))
                print("first lines:", data.splitlines()[:5])
            else:
                print("first bytes:", data[:16])

print("\n=== QASMBench discovery ===")
qasm_zip = root / "source=qasmbench" / "qasmbench-qec.zip"
with zipfile.ZipFile(qasm_zip) as z:
    qasm_files = sorted(name for name in z.namelist() if name.endswith(".qasm"))
    benchmarks = sorted({Path(name).parent.name for name in qasm_files if name.startswith("small/")})
    print("benchmark folders:", benchmarks)
    print("qasm files:", qasm_files[:10])
    print("total qasm files:", len(qasm_files))

    sample_name = qasm_files[0]
    sample_text = z.read(sample_name).decode("utf-8")
    print("sample file:", sample_name)
    print("sample first 20 lines:")
    for line in sample_text.splitlines()[:20]:
        print(line)

    # Basic QASM features worth recording for later analysis.
    qasm_keywords = {
        "OPENQASM": "openqasm version",
        "qreg": "quantum register",
        "creg": "classical register",
        "cx": "CNOT",
        "measure": "measurement",
        "if": "conditional operation",
        "x": "X gate",
        "z": "Z gate",
    }
    found = {k: sample_text.count(v) for k, v in {"OPENQASM": "OPENQASM", "qreg": "qreg", "creg": "creg", "cx": "cx", "measure": "measure", "if": "if", "x": "x", "z": "z"}.items()}
    print("sample keyword counts:", found)

    print("QASMBench notes:")
    print("- these are OpenQASM circuits, not training rows")
    print("- focus on benchmark name, circuit variant, register sizes, and parity-check structure")
    print("- possible IDs: benchmark folder, circuit file, qasm member path, register names")


    # TODO: Write down discoveries and decisions in decision_log.md --> discuss the next steps with the team 

    