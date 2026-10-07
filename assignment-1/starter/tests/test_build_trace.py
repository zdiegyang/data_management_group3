"""Tests for results/part1/trace_examples.json (make trace)."""

import json

import psycopg
import pytest

from quantum_lake_student import build_trace
from quantum_lake_student.config import Settings

HOPS = ["1_prediction", "2_ml_row", "3_gold", "4_silver_row", "5_source_trace", "6_bronze"]


def _ready(settings):
    """Gold must be loaded and Part II must have written predictions."""
    if not build_trace.PREDICTIONS.exists():
        return False
    try:
        with psycopg.connect(settings.postgres_dsn) as conn:
            return conn.execute("SELECT to_regclass('gold.v_ml_example_gold_record') IS NOT NULL").fetchone()[0]
    except psycopg.Error:
        return False


@pytest.fixture(scope="module")
def traces():
    settings = Settings.from_environment()
    if not _ready(settings):
        pytest.skip("needs `make run` and `make train` first")
    build_trace.run("trace-test-1", settings)
    first = (build_trace.RESULTS / "trace_examples.json").read_text(encoding="utf-8")
    build_trace.run("trace-test-2", settings)
    second = (build_trace.RESULTS / "trace_examples.json").read_text(encoding="utf-8")
    return first, second, json.loads(second)


def test_trace_file_is_identical_on_every_run(traces):
    first, second, _ = traces
    assert first == second


def test_each_trace_starts_from_a_test_prediction_and_reaches_bronze(traces):
    *_, data = traces
    assert set(data) == {"syndrome_prediction_trace", "google_prediction_trace"}
    for trace in data.values():
        assert [key for key in sorted(trace) if key[0].isdigit()] == HOPS
        prediction = trace["1_prediction"]["row"]
        assert prediction["split"] == "test"
        assert trace["2_ml_row"]["row"]["example_id"] == prediction["example_id"]
        source_record_id = trace["4_silver_row"]["row"]["source_record_id"]
        assert all(row["source_record_id"] == source_record_id for row in trace["5_source_trace"]["rows"])
        assert all(item["matches_verified_input_hash"] for item in trace["6_bronze"]["objects"])


def test_google_trace_lists_the_shot_and_all_eight_companion_members(traces):
    *_, data = traces
    google = data["google_prediction_trace"]
    assert {link["gold_table"] for link in google["3_gold"]["links"]} == {"google_shot", "decoder_prediction"}
    assert len(google["3_gold"]["records"]["decoder_prediction"]) == 4
    members = {member["archive_member"].split("/")[-1] for member in google["6_bronze"]["members"]}
    assert members == {
        "measurements.b8", "sweep.b8", "detection_events.b8", "obs_flips_actual.01",
        "obs_flips_predicted_by_belief_matching.01", "obs_flips_predicted_by_correlated_matching.01",
        "obs_flips_predicted_by_pymatching.01", "obs_flips_predicted_by_tensor_network_contraction.01",
    }


def test_a_hash_that_differs_from_the_verified_input_stops_the_trace():
    settings = Settings.from_environment()
    if not _ready(settings):
        pytest.skip("needs `make run` and `make train` first")
    import pyarrow.parquet as pq

    spec = build_trace.TRACES["syndrome_prediction_trace"]
    tables = {key: build_trace._read_lake_table(settings, key) for key in (spec["ml_key"], spec["silver_key"])}
    with psycopg.connect(settings.postgres_dsn) as conn:
        with pytest.raises(build_trace.TraceError, match="hash"):
            build_trace.build_trace(
                spec, conn, pq.read_table(build_trace.PREDICTIONS),
                build_trace._read_lake_table(settings, build_trace.TRACE_KEY), tables,
                verified_hashes={},  # nothing verified -> must refuse
            )


def test_packed_bytes_are_written_as_hex():
    assert build_trace._plain(b"\x01\xff") == "01ff"
    assert build_trace._plain(5) == 5
