from pathlib import Path

import pyarrow.parquet as pq

from quantum_lake_student.stages import prepare_data

def _silver_snapshot(base_dir: Path):
    silver_dir = base_dir / "silver"

    tables = {
        "syndrome_observation": silver_dir / "qec_syndromes/syndrome_observation.parquet",
        "google_experiment": silver_dir / "google_qec/experiment.parquet",
        "google_shot": silver_dir / "google_qec/shot.parquet",
        "circuit": silver_dir / "qasmbench/circuit.parquet",
        "stabilizer_check": silver_dir / "qasmbench/stabilizer_check.parquet",
        "conditional_correction": silver_dir / "qasmbench/conditional_correction.parquet",
    }

    snapshot = {}

    for name, path in tables.items():
        table = pq.read_table(path)

        snapshot[name] = {
            "rows": table.num_rows,
            "source_record_ids": sorted(
                table.column("source_record_id").to_pylist()
            ),
        }

    return snapshot

def test_prepare_data_is_repeatable():
    base_dir = Path(__file__).resolve().parents[1]

    prepare_data.run("repeatability-run-1")
    first = _silver_snapshot(base_dir)

    prepare_data.run("repeatability-run-2")
    second = _silver_snapshot(base_dir)

    assert first == second