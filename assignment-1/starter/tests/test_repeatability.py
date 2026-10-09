from pathlib import Path

import pyarrow.parquet as pq
import io
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.stages import prepare_data

def _read_table_from_minio(client, bucket: str, object_name: str):
    """Fetch parquet object from MinIO and load it into a PyArrow Table in memory."""
    response = client.get_object(bucket, object_name)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    return pq.read_table(io.BytesIO(data))


def _silver_snapshot(settings: Settings):
    client = minio_client(settings)
    bucket = settings.s3_bucket

    objects = {
        "syndrome_observation": "silver/qec_syndromes/syndrome_observation.parquet",
        "google_experiment": "silver/google_qec/experiment.parquet",
        "google_shot": "silver/google_qec/shot.parquet",
        "circuit": "silver/qasmbench/circuit.parquet",
        "stabilizer_check": "silver/qasmbench/stabilizer_check.parquet",
        "conditional_correction": "silver/qasmbench/conditional_correction.parquet",
    }

    snapshot = {}

    for name, object_name in objects.items():
        table = _read_table_from_minio(client, bucket, object_name)

        snapshot[name] = {
            "rows": table.num_rows,
            "source_record_ids": sorted(
                table.column("source_record_id").to_pylist()
            ),
        }

    return snapshot


def test_prepare_data_is_repeatable():
    settings = Settings.from_environment()

    prepare_data.run("repeatability-run-1", settings=settings)
    first = _silver_snapshot(settings)

    prepare_data.run("repeatability-run-2", settings=settings)
    second = _silver_snapshot(settings)

    assert first == second