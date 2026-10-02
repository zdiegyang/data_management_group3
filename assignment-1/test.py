from __future__ import annotations
import ast
import csv
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter" / "src"))

from quantum_lake_student.formats import iter_b8_records, parse_01_records


def validate_syndrome_row(row: dict[str, str]) -> None:
    syndrome = ast.literal_eval(row["syndromes"])
    label = int(row["labels"])
    quantity = int(row["quantity"])

    assert isinstance(syndrome, tuple), "syndrome should parse as a tuple"
    assert len(syndrome) == 4, "expected 4 rounds"
    assert all(isinstance(round_values, tuple) for round_values in syndrome)
    assert all(len(round_values) == 4 for round_values in syndrome), (
        "expected 4 syndrome checks per round"
    )
    assert all(
        bit in (0, 1)
        for round_values in syndrome
        for bit in round_values
    ), "syndrome values must be binary"
    assert label in (0, 1), "label must be binary"
    assert quantity > 0, "quantity must be positive"

    print("Validated syndrome sample:")
    print("  file:", row.get("__source__", "<unknown>"))
    print("  syndrome:", syndrome)
    print("  label:", label)
    print("  quantity:", quantity)


def validate_b8_sample(data: bytes, *, bits_per_record: int = 8, file_name: str = "<b8>") -> None:
    records = list(iter_b8_records(data, bits_per_record=bits_per_record))
    assert records, "expected at least one b8 record"
    assert all(len(record) == bits_per_record for record in records), (
        "each record should have the expected bit width"
    )
    assert all(bit in (0, 1) for record in records for bit in record), (
        "b8 bits must be binary"
    )
    print("Validated b8 sample:")
    print("  file:", file_name)
    print("  first records:", records[:3])


def validate_01_sample(data: bytes, file_name: str = "<01>") -> None:
    values = parse_01_records(data)
    assert values, "expected at least one 01 value"
    assert all(value in (0, 1) for value in values), "01 values must be 0 or 1"
    print("Validated 01 sample:")
    print("  file:", file_name)
    print("  first values:", values[:10])


def main() -> None:
    syndromes_path = (
        ROOT / "datasets/student-bundle/core/raw/source=qec_syndromes/syndromes_dataset.zip"
    )
    with zipfile.ZipFile(syndromes_path) as archive:
        csv_name = next(
            name for name in archive.namelist()
            if name.endswith(".csv") and not name.endswith("/")
        )
        with archive.open(csv_name, "r") as raw_file:
            reader = csv.DictReader(line.decode("utf-8") for line in raw_file)
            row = next(reader)
        row["__source__"] = csv_name
        validate_syndrome_row(row)

    google_path = (
        ROOT / "datasets/student-bundle/core/raw/source=google_qec/google-surface-code-curated.zip"
    )
    with zipfile.ZipFile(google_path) as archive:
        experiment = next(
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and name.endswith("/measurements.b8")
        )
        measurements_name = f"{experiment}/measurements.b8"
        measurements_data = archive.read(measurements_name)[:16]
        validate_b8_sample(measurements_data, bits_per_record=8, file_name=measurements_name)

        actual_name = f"{experiment}/obs_flips_actual.01"
        actual_data = archive.read(actual_name)
        sample_lines = actual_data.splitlines()[:10]
        sample_01 = b"\n".join(sample_lines)
        validate_01_sample(sample_01, file_name=actual_name)

    print("Checked archives:")
    print("  syndrome CSV:", syndromes_path)
    print("  google experiment zip:", google_path)


if __name__ == "__main__":
    main()

