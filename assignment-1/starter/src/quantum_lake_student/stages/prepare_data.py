"""Parse, check, and connect the supplied QEC data into Silver tables and source traces.

Implements Stage 2 of the course platform:
- Syndromes CSV parsing, validation, and 16-byte packed formatting.
- Google QEC experiments metadata extraction from properties.yml.
- Google QEC shots parsing with 8-member companion alignment and bit-level invariant checks.
- Extensible hook for QASMBench parsing (prepared for Diego to complete).
- Full source lineage tracing with 'valid' (yes/no) column.
- Machine-readable data issues logging to results/part1/data_issues.parquet.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any
import zipfile
import yaml
import re 

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.formats import b8_record_bytes, parse_01_records
from quantum_lake_student.models import StageResult, stable_record_hash
from quantum_lake_student.tracing import SOURCE_TRACE_SCHEMA, save_source_traces


# ==============================================================================
# SCHEMA DEFINITIONS (Strictly matching silver-tables.md & brief.md)
# ==============================================================================

SYNDROME_OBSERVATION_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("experiment_id", pa.string()),
    ("physical_fault_rate", pa.float64()),
    ("syndrome_bits", pa.binary()),
    ("round_count", pa.int32()),
    ("check_count", pa.int32()),
    ("logical_error_label", pa.bool_()),
    ("quantity", pa.int64()),
])

GOOGLE_EXPERIMENT_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("experiment_id", pa.string()),
    ("basis", pa.string()),
    ("distance", pa.int32()),
    ("rounds", pa.int32()),
    ("shots", pa.int64()),
    ("center_row", pa.int32()),
    ("center_col", pa.int32()),
    ("measurement_count", pa.int32()),
    ("detector_count", pa.int32()),
])

GOOGLE_SHOT_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("experiment_id", pa.string()),
    ("shot_index", pa.int64()),
    ("measurement_bits", pa.binary()),
    ("sweep_bits", pa.binary()),
    ("detector_bits", pa.binary()),
    ("detector_event_count", pa.int32()),
    ("actual_observable_flip", pa.bool_()),
    ("belief_matching_prediction", pa.bool_()),
    ("correlated_matching_prediction", pa.bool_()),
    ("pymatching_prediction", pa.bool_()),
    ("tensor_network_contraction_prediction", pa.bool_()),
])

DATA_ISSUES_SCHEMA = pa.schema([
    ("issue_id", pa.string()),
    ("run_id", pa.string()),
    ("source_record_id", pa.string()),
    ("rule_id", pa.string()),
    ("severity", pa.string()),
    ("observed_value", pa.string()),
    ("action", pa.string()),
    ("reason", pa.string()),
])


def _stable_issue_id(source_record_id: str, rule_id: str, observed_value: str) -> str:
    return f"issue_{stable_record_hash({
        'source_record_id': source_record_id,
        'rule_id': rule_id,
        'observed_value': observed_value,
    })}"


def _append_qasm_issue(
    issues: list[dict[str, Any]],
    run_id: str,
    source_record_id: str,
    rule_id: str,
    observed_value: str,
    reason: str,
) -> None:
    issues.append({
        "issue_id": _stable_issue_id(source_record_id, rule_id, observed_value),
        "run_id": run_id,
        "source_record_id": source_record_id,
        "rule_id": rule_id,
        "severity": "error",
        "observed_value": observed_value,
        "action": "excluded",
        "reason": reason,
    })


class MissingCompanionFileError(RuntimeError):
    """A Google experiment lacks a required companion file; the run must stop."""


# Every Silver rule: (source, what is checked, unit, action on failure).
# Brief: "Record the outcome of every check." Failures also go to data_issues.
CHECK_CATALOG: dict[str, tuple[str, str, str, str]] = {
    "RULE_SYN_FILENAME_PFR": ("qec_syndromes", "physical fault rate parses from the file name", "CSV file", "exclude file"),
    "RULE_SYN_EMPTY_FILE": ("qec_syndromes", "CSV file has a header row", "CSV file", "exclude file"),
    "RULE_SYN_HEADER_DOCUMENTED": ("qec_syndromes", "CSV header compared with the columns documented in README.txt", "CSV file", "record observation"),
    "RULE_SYN_HEADER": ("qec_syndromes", "header is labels,syndromes,quantity (or the documented label,...)", "CSV file", "log warning"),
    "RULE_SYN_ROW_LEN": ("qec_syndromes", "row has exactly 3 fields", "CSV row", "exclude row"),
    "RULE_SYN_LABEL_DOMAIN": ("qec_syndromes", "label is 0 or 1", "CSV row", "exclude row"),
    "RULE_SYN_SHAPE_DOMAIN": ("qec_syndromes", "syndrome is 4 rounds x 4 checks of binary values", "CSV row", "exclude row"),
    "RULE_SYN_QUANTITY_POSITIVE": ("qec_syndromes", "quantity is an integer greater than zero", "CSV row", "exclude row"),
    "RULE_SYN_WEIGHTED_TOTAL": ("qec_syndromes", "accepted quantities of a file sum to the nb-<N> count in its name", "CSV file", "log error"),
    "RULE_SYN_SAME_SYNDROME_BOTH_LABELS": ("qec_syndromes", "a syndrome occurring with both labels in one file is valid and kept", "distinct (file, syndrome)", "record observation"),
    "RULE_EXP_PROPERTIES": ("google_qec", "properties.yml parses with valid basis, distance, rounds and shots", "experiment", "exclude experiment"),
    "RULE_GOOGLE_NAME_PROPERTIES": ("google_qec", "directory name agrees with properties.yml", "experiment", "exclude experiment and its shots"),
    "RULE_GOOGLE_MISSING_COMPANION": ("google_qec", "every required companion file is present", "experiment", "stop run"),
    "RULE_GOOGLE_01_BINARY": ("google_qec", "every .01 line is 0 or 1", ".01 file", "stop run"),
    "RULE_GOOGLE_COMPANION_STRIDE_MISMATCH": ("google_qec", "b8 lengths equal shots x ceil(bits/8) and .01 line counts equal shots", "experiment", "exclude experiment"),
    "RULE_GOOGLE_PADDING_BITS": ("google_qec", "unused padding bits of every b8 record are zero", "shot", "exclude shot"),
    "RULE_QASM_ARCHIVE_UNAVAILABLE": ("qasmbench", "QASMBench archive can be read", "archive", "exclude source"),
    "RULE_QASM_CIRCUIT_PARSE": ("qasmbench", "circuit registers and operation counts parse", "QASM file", "exclude circuit"),
    "RULE_QASM_STABILIZER_PARSE": ("qasmbench", "parity checks parse", "QASM file", "exclude file's checks"),
    "RULE_QASM_CORRECTION_PARSE": ("qasmbench", "conditional corrections parse", "QASM file", "exclude file's corrections"),
}


def _record_check(
    checks: dict[str, dict[str, Any]] | None,
    rule_id: str,
    checked: int = 1,
    observation: str | None = None,
    failed: int = 0,
) -> None:
    """Count how many records a rule evaluated, plus any expected observation."""
    if checks is None:
        return
    entry = checks.setdefault(rule_id, {"checked": 0, "observations": [], "failed": 0})
    entry["checked"] += checked
    entry["failed"] += failed
    if observation and observation not in entry["observations"]:
        entry["observations"].append(observation)


def _check_outcomes(
    checks: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Outcome of every catalogued rule: how many records it checked, passed and failed."""
    failed_by_rule: dict[str, int] = {}
    for issue in issues:
        failed_by_rule[issue["rule_id"]] = failed_by_rule.get(issue["rule_id"], 0) + 1
    outcomes = []
    for rule_id, (source, description, unit, on_failure) in CHECK_CATALOG.items():
        entry = checks.get(rule_id, {"checked": 0, "observations": []})
        failed = failed_by_rule.get(rule_id, 0)
        if failed:
            outcome = "failed"
        elif entry["observations"]:
            outcome = "observed"
        elif entry["checked"]:
            outcome = "passed"
        else:
            outcome = "not evaluated"
        outcomes.append({
            "rule_id": rule_id,
            "source": source,
            "check": description,
            "unit": unit,
            "on_failure": on_failure,
            "checked": entry["checked"],
            "failed": failed,
            "passed": max(entry["checked"] - failed, 0),
            "outcome": outcome,
            "observations": entry["observations"],
        })
    return outcomes


NOMINAL_SAMPLES = re.compile(r"_nb-(\d+)([KMG]?)", re.IGNORECASE)
SAMPLE_SUFFIX = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}


def _check_weighted_total(
    member: str,
    file_weight: int,
    run_id: str,
    issues: list[dict[str, Any]],
    checks: dict[str, dict[str, Any]] | None,
) -> None:
    """Rule: the accepted quantities of one CSV sum to the nb-<N> count in its name."""
    _record_check(checks, "RULE_SYN_WEIGHTED_TOTAL")
    match = NOMINAL_SAMPLES.search(member)
    if match is None:
        declared = None
        reason = "File name has no nb-<count> token to reconcile the quantities against"
    else:
        declared = int(match.group(1)) * SAMPLE_SUFFIX[match.group(2).upper()]
        if file_weight == declared:
            return
        reason = f"Accepted quantities sum to {file_weight}, the file name declares {declared}"
    source_record_id = f"qec_syndromes:{member}"
    observed_value = f"sum={file_weight};declared={declared}"
    issues.append({
        "issue_id": _stable_issue_id(source_record_id, "RULE_SYN_WEIGHTED_TOTAL", observed_value),
        "run_id": run_id,
        "source_record_id": source_record_id,
        "rule_id": "RULE_SYN_WEIGHTED_TOTAL",
        "severity": "error",
        "observed_value": observed_value,
        "action": "logged",
        "reason": reason,
    })


DOCUMENTED_COLUMN = re.compile(r"^\s*-\s*(\w+)\s*:", re.MULTILINE)


def _documented_syndrome_columns(archive: zipfile.ZipFile) -> list[str] | None:
    """Column names listed in the syndrome archive README, if it has one."""
    readme = next(
        (name for name in archive.namelist() if Path(name).name.lower().startswith("readme")),
        None,
    )
    if readme is None:
        return None
    return DOCUMENTED_COLUMN.findall(archive.read(readme).decode("utf-8", errors="replace"))


GOOGLE_EXPERIMENT_DIR = re.compile(
    r"^(?P<code>[a-z_]+?)_b(?P<basis>[A-Z])_d(?P<distance>\d+)_r(?P<rounds>\d+)"
    r"_center_(?P<row>\d+)_(?P<col>\d+)$"
)


def _experiment_name_mismatches(exp_dir: str, prop: dict[str, Any]) -> list[str]:
    """Compare `{code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}` (the
    pattern documented in the Google README) with the values in properties.yml."""
    match = GOOGLE_EXPERIMENT_DIR.fullmatch(exp_dir)
    if match is None:
        return ["name does not follow {code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}"]
    declared = {
        "code": str(prop.get("type", "")),
        "basis": str(prop.get("basis")),
        "distance": str(prop.get("distance")),
        "rounds": str(prop.get("rounds")),
        "row": str(prop.get("center_data_qubit_row")),
        "col": str(prop.get("center_data_qubit_col")),
    }
    mismatches = []
    for field, from_name in match.groupdict().items():
        # properties.yml `type` is e.g. surface_code_memory_experiment; the name holds its prefix.
        if field == "code":
            agrees = declared[field].startswith(from_name)
        else:
            agrees = declared[field] == from_name
        if not agrees:
            mismatches.append(f"{field}: name={from_name} properties={declared[field]}")
    return mismatches


def _row_counts(
    rows_read: int,
    rows_accepted: int,
    rows_loaded: int,
    unit: str,
) -> dict[str, int | str]:
    rows_rejected = rows_read - rows_accepted
    if rows_rejected < 0:
        raise ValueError("Accepted row count cannot exceed rows read")
    return {
        "unit": unit,
        "rows_read": rows_read,
        "rows_accepted": rows_accepted,
        "rows_rejected": rows_rejected,
        "rows_loaded": rows_loaded,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _remember_table(
    run_state: dict[str, Any],
    table_path: str,
    counts: dict[str, int | str],
) -> None:
    run_state["row_counts"][table_path] = counts
    run_state["output_table_counts"][table_path] = counts["rows_loaded"]


def _record_failed_run(
    run_id: str,
    result: StageResult | None,
    run_state: dict[str, Any],
    error: Exception,
) -> None:
    base_dir = run_state["base_dir"]
    if result is None:
        result = StageResult(stage="prepare_data", run_id=run_id)
    results_base = base_dir / "results/part1"
    results_base.mkdir(parents=True, exist_ok=True)
    result.finish()

    error_message = str(error)
    issues = run_state.setdefault("issues", [])
    issues.append({
        "issue_id": _stable_issue_id(
            "prepare_data", "RUN_PREPARE_DATA_FAILED", error_message
        ),
        "run_id": run_id,
        "source_record_id": None,
        "rule_id": "RUN_PREPARE_DATA_FAILED",
        "severity": "error",
        "observed_value": f"{type(error).__name__}: {error_message}",
        "action": "aborted",
        "reason": error_message,
    })

    row_counts = run_state.setdefault("row_counts", {})
    output_table_counts = run_state.setdefault("output_table_counts", {})
    issues_path = results_base / "data_issues.parquet"
    try:
        issue_table = pa.Table.from_pandas(
            pd.DataFrame(issues),
            schema=DATA_ISSUES_SCHEMA,
            preserve_index=False,
        )
        pq.write_table(issue_table, issues_path, compression="zstd")
        _remember_table(
            run_state,
            "results/part1/data_issues.parquet",
            _row_counts(len(issues), len(issues), issue_table.num_rows, "issue finding"),
        )
    except Exception:
        logging.exception("Could not persist data issues for failed run %s", run_id)

    result.issue_count = len(issues)
    result.finish()
    row_counts_path = results_base / "row_counts.json"
    try:
        # run.json (status, error, hashes) is written once per run by the CLI.
        _write_json(row_counts_path, {"run_id": run_id, "tables": row_counts})
    except Exception:
        logging.exception("Could not persist failed run row counts for %s", run_id)

    settings = run_state.get("settings")
    if settings and settings.lake_backend == "minio":
        try:
            client = minio_client(settings)
            client.fput_object(settings.s3_bucket, "results/part1/data_issues.parquet", str(issues_path))
            client.fput_object(settings.s3_bucket, "results/part1/row_counts.json", str(row_counts_path))
        except Exception:
            logging.exception("Could not upload failed run metadata for %s", run_id)

DECODER_NAMES = [
    "belief_matching",
    "correlated_matching",
    "pymatching",
    "tensor_network_contraction",
]

QASMBENCH_CIRCUIT_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("circuit_id", pa.string()),
    ("benchmark_name", pa.string()),
    ("variant", pa.string()),
    ("register_declarations", pa.string()),
    ("qubit_count", pa.int32()),
    ("operation_count", pa.int32()),
    ("measurement_count", pa.int32()),
    ("two_qubit_gate_count", pa.int32()),
])

QASMBENCH_STABILIZER_CHECK_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("circuit_id", pa.string()),
    ("check_id", pa.string()),
    ("ancilla_qubit", pa.string()),
    ("data_qubits", pa.list_(pa.string())),
    ("syndrome_bit", pa.string()),
])

QASMBENCH_CONDITIONAL_CORRECTION_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("circuit_id", pa.string()),
    ("condition_register", pa.string()),
    ("condition_value", pa.int64()),
    ("gate", pa.string()),
    ("target_qubit", pa.string()),
])


# ==============================================================================
# BRONZE HELPER: FETCH AND HASH ARCHIVES
# ==============================================================================

def get_bronze_archive(
    bronze_object_name: str,
    settings: Settings,
    base_dir: Path,
) -> tuple[bytes, str]:
    """Fetch raw Bronze archive bytes and return (bytes, sha256_hash).

    Tries MinIO first if configured, then falls back to local workspace mounts.
    """
    bronze_bytes: bytes | None = None

    if settings.lake_backend == "minio":
        try:
            client = minio_client(settings)
            response = client.get_object(settings.s3_bucket, bronze_object_name)
            bronze_bytes = response.read()
            response.close()
            response.release_conn()
        except Exception:
            bronze_bytes = None

    if bronze_bytes is None:
        candidates = [
            Path("/course-data/raw") / bronze_object_name.replace("bronze/", ""),
            base_dir.parent / "datasets/student-bundle/core/raw" / bronze_object_name.replace("bronze/", ""),
            base_dir / "datasets/student-bundle/core/raw" / bronze_object_name.replace("bronze/", ""),
            Path("datasets/student-bundle/core/raw") / bronze_object_name.replace("bronze/", ""),
        ]
        for p in candidates:
            if p.exists():
                bronze_bytes = p.read_bytes()
                break

    if bronze_bytes is None:
        raise FileNotFoundError(
            f"Could not locate Bronze archive '{bronze_object_name}' in MinIO or local paths."
        )

    sha256 = hashlib.sha256(bronze_bytes).hexdigest()
    return bronze_bytes, sha256


# ==============================================================================
# 1. SYNDROME OBSERVATIONS PARSER
# ==============================================================================

def prepare_syndrome_observations(
    bronze_bytes: bytes,
    bronze_object_name: str,
    bronze_sha256: str,
    run_id: str,
    issues: list[dict[str, Any]],
    checks: dict[str, dict[str, Any]] | None = None,
) -> tuple[pa.Table, dict[str, list[Any]], int]:
    """Parse simulated syndrome CSV files, validate invariants, and extract traces."""
    silver_rows = []
    trace_cols: dict[str, list[Any]] = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }
    input_records_count = 0

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as z:
        csv_members = sorted([name for name in z.namelist() if name.endswith(".csv") and not name.startswith("__MACOSX")])
        documented_columns = _documented_syndrome_columns(z)
        for member in csv_members:
            stem = Path(member).stem
            _record_check(checks, "RULE_SYN_FILENAME_PFR")
            # Parse physical fault rate from filename: d-3_pfr-0.001000_nb-10M.csv
            try:
                pfr_str = member.split("_pfr-")[1].split("_")[0]
                pfr = float(pfr_str)
            except Exception as e:
                issues.append({
                    "issue_id": _stable_issue_id(
                        f"qec_syndromes:{member}", "RULE_SYN_FILENAME_PFR", member
                    ),
                    "run_id": run_id,
                    "source_record_id": f"qec_syndromes:{member}",
                    "rule_id": "RULE_SYN_FILENAME_PFR",
                    "severity": "error",
                    "observed_value": member,
                    "action": "excluded",
                    "reason": f"Could not parse physical_fault_rate from filename: {e}",
                })
                continue

            exp_id = stem
            file_weight = 0

            with z.open(member) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                _record_check(checks, "RULE_SYN_EMPTY_FILE")
                try:
                    header = next(reader)
                except StopIteration:
                    issues.append({
                        "issue_id": _stable_issue_id(
                            f"qec_syndromes:{member}", "RULE_SYN_EMPTY_FILE", member
                        ),
                        "run_id": run_id,
                        "source_record_id": f"qec_syndromes:{member}",
                        "rule_id": "RULE_SYN_EMPTY_FILE",
                        "severity": "error",
                        "observed_value": member,
                        "action": "excluded",
                        "reason": "CSV file is empty",
                    })
                    continue

                # The README documents a column `label`; the files use `labels`.
                # Expected observation: columns are read by position, not by name.
                header_observation = None
                if documented_columns and header != documented_columns:
                    header_observation = (
                        f"README documents columns {documented_columns}; CSV header is {header}"
                    )
                _record_check(checks, "RULE_SYN_HEADER_DOCUMENTED", observation=header_observation)

                _record_check(checks, "RULE_SYN_HEADER")
                if header not in (["labels", "syndromes", "quantity"], ["label", "syndromes", "quantity"]):
                    issues.append({
                        "issue_id": _stable_issue_id(
                            f"qec_syndromes:{member}:header",
                            "RULE_SYN_HEADER",
                            str(header),
                        ),
                        "run_id": run_id,
                        "source_record_id": f"qec_syndromes:{member}:header",
                        "rule_id": "RULE_SYN_HEADER",
                        "severity": "warning",
                        "observed_value": str(header),
                        "action": "logged",
                        "reason": f"Non-standard header: {header}",
                    })

                for row_idx, row in enumerate(reader):
                    input_records_count += 1
                    source_record_id = f"qec_syndromes:{member}:row:{row_idx}"
                    record_locator = f"row:{row_idx}"

                    _record_check(checks, "RULE_SYN_ROW_LEN")
                    if len(row) != 3:
                        issues.append({
                            "issue_id": _stable_issue_id(
                                source_record_id, "RULE_SYN_ROW_LEN", str(row)
                            ),
                            "run_id": run_id,
                            "source_record_id": source_record_id,
                            "rule_id": "RULE_SYN_ROW_LEN",
                            "severity": "error",
                            "observed_value": str(row),
                            "action": "excluded",
                            "reason": f"Expected 3 columns, found {len(row)}",
                        })
                        trace_cols["source_record_id"].append(source_record_id)
                        trace_cols["source_name"].append("qec_syndromes")
                        trace_cols["bronze_object"].append(bronze_object_name)
                        trace_cols["archive_member"].append(member)
                        trace_cols["record_locator"].append(record_locator)
                        trace_cols["input_sha256"].append(bronze_sha256)
                        trace_cols["valid"].append("no")
                        continue

                    # Validate label
                    _record_check(checks, "RULE_SYN_LABEL_DOMAIN")
                    try:
                        label_val = int(row[0].strip())
                        if label_val not in (0, 1):
                            raise ValueError(f"Label not binary: {label_val}")
                        logical_error_label = bool(label_val)
                    except Exception as e:
                        issues.append({
                            "issue_id": _stable_issue_id(
                                source_record_id, "RULE_SYN_LABEL_DOMAIN", row[0]
                            ),
                            "run_id": run_id,
                            "source_record_id": source_record_id,
                            "rule_id": "RULE_SYN_LABEL_DOMAIN",
                            "severity": "error",
                            "observed_value": row[0],
                            "action": "excluded",
                            "reason": str(e),
                        })
                        trace_cols["source_record_id"].append(source_record_id)
                        trace_cols["source_name"].append("qec_syndromes")
                        trace_cols["bronze_object"].append(bronze_object_name)
                        trace_cols["archive_member"].append(member)
                        trace_cols["record_locator"].append(record_locator)
                        trace_cols["input_sha256"].append(bronze_sha256)
                        trace_cols["valid"].append("no")
                        continue

                    # Validate 4x4 syndrome sequence
                    _record_check(checks, "RULE_SYN_SHAPE_DOMAIN")
                    try:
                        syndrome_tuple = ast.literal_eval(row[1].strip())
                        if len(syndrome_tuple) != 4 or not all(len(r) == 4 for r in syndrome_tuple):
                            raise ValueError("Syndrome must have shape 4 rounds x 4 checks")
                        flat_bits = [bit for r in syndrome_tuple for bit in r]
                        if len(flat_bits) != 16 or any(b not in (0, 1) for b in flat_bits):
                            raise ValueError("Syndrome bits must all be 0 or 1")
                        syndrome_bytes = bytes(flat_bits)
                    except Exception as e:
                        _record_check(checks, "RULE_SYN_SHAPE_DOMAIN", checked=0, failed=1)
                        issues.append({
                            "issue_id": _stable_issue_id(
                                source_record_id, "RULE_SYN_SHAPE_DOMAIN", row[1]
                            ),
                            "run_id": run_id,
                            "source_record_id": source_record_id,
                            "rule_id": "RULE_SYN_SHAPE_DOMAIN",
                            "severity": "error",
                            "observed_value": row[1],
                            "action": "excluded",
                            "reason": str(e),
                        })
                        trace_cols["source_record_id"].append(source_record_id)
                        trace_cols["source_name"].append("qec_syndromes")
                        trace_cols["bronze_object"].append(bronze_object_name)
                        trace_cols["archive_member"].append(member)
                        trace_cols["record_locator"].append(record_locator)
                        trace_cols["input_sha256"].append(bronze_sha256)
                        trace_cols["valid"].append("no")
                        continue

                    # Validate quantity
                    _record_check(checks, "RULE_SYN_QUANTITY_POSITIVE")
                    try:
                        quantity = int(row[2].strip())
                        if quantity <= 0:
                            raise ValueError("Quantity must be greater than zero")
                    except Exception as e:
                        _record_check(checks, "RULE_SYN_QUANTITY_POSITIVE", checked=0, failed=1)
                        issues.append({
                            "issue_id": _stable_issue_id(
                                source_record_id, "RULE_SYN_QUANTITY_POSITIVE", row[2]
                            ),
                            "run_id": run_id,
                            "source_record_id": source_record_id,
                            "rule_id": "RULE_SYN_QUANTITY_POSITIVE",
                            "severity": "error",
                            "observed_value": row[2],
                            "action": "excluded",
                            "reason": str(e),
                        })
                        trace_cols["source_record_id"].append(source_record_id)
                        trace_cols["source_name"].append("qec_syndromes")
                        trace_cols["bronze_object"].append(bronze_object_name)
                        trace_cols["archive_member"].append(member)
                        trace_cols["record_locator"].append(record_locator)
                        trace_cols["input_sha256"].append(bronze_sha256)
                        trace_cols["valid"].append("no")
                        continue

                    # Accepted valid record
                    silver_rows.append({
                        "source_record_id": source_record_id,
                        "experiment_id": exp_id,
                        "physical_fault_rate": pfr,
                        "syndrome_bits": syndrome_bytes,
                        "round_count": 4,
                        "check_count": 4,
                        "logical_error_label": logical_error_label,
                        "quantity": quantity,
                    })

                    trace_cols["source_record_id"].append(source_record_id)
                    trace_cols["source_name"].append("qec_syndromes")
                    trace_cols["bronze_object"].append(bronze_object_name)
                    trace_cols["archive_member"].append(member)
                    trace_cols["record_locator"].append(record_locator)
                    trace_cols["input_sha256"].append(bronze_sha256)
                    trace_cols["valid"].append("yes")
                    file_weight += quantity

            _check_weighted_total(member, file_weight, run_id, issues, checks)

    # from_pylist keeps the schema even when every row was rejected.
    table_silver = pa.Table.from_pylist(silver_rows, schema=SYNDROME_OBSERVATION_SCHEMA)
    return table_silver, trace_cols, input_records_count


# ==============================================================================
# 2. GOOGLE QEC EXPERIMENTS PARSER
# ==============================================================================

def prepare_google_experiments(
    bronze_bytes: bytes,
    bronze_object_name: str,
    bronze_sha256: str,
    run_id: str,
    issues: list[dict[str, Any]],
    checks: dict[str, dict[str, Any]] | None = None,
) -> tuple[pa.Table, dict[str, list[Any]], int]:
    """Parse Google QEC experiment directories and properties.yml."""
    experiment_rows = []
    trace_cols: dict[str, list[Any]] = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }
    input_records_count = 0

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as z:
        yaml_members = sorted([name for name in z.namelist() if name.endswith("properties.yml") and not name.startswith("__MACOSX")])
        for ym in yaml_members:
            input_records_count += 1
            exp_dir = ym.split("/")[0]
            source_record_id = f"google_qec:{exp_dir}:properties.yml"
            _record_check(checks, "RULE_EXP_PROPERTIES")

            try:
                raw_yaml = z.read(ym).decode("utf-8")
                prop = yaml.safe_load(raw_yaml)

                basis = str(prop["basis"])
                distance = int(prop["distance"])
                rounds = int(prop["rounds"])
                shots = int(prop["shots"])
                center_row = int(prop["center_data_qubit_row"])
                center_col = int(prop["center_data_qubit_col"])
                meas_count = int(prop["circuit_measurements"])
                det_count = int(prop["circuit_detectors"])

                if basis not in ("X", "Z") or distance not in (3, 5) or rounds <= 0 or shots <= 0:
                    raise ValueError(f"Invalid experiment property values in {ym}")

                _record_check(checks, "RULE_GOOGLE_NAME_PROPERTIES")
                mismatches = _experiment_name_mismatches(exp_dir, prop)
                if mismatches:
                    observed_value = "; ".join(mismatches)
                    issues.append({
                        "issue_id": _stable_issue_id(
                            source_record_id, "RULE_GOOGLE_NAME_PROPERTIES", observed_value
                        ),
                        "run_id": run_id,
                        "source_record_id": source_record_id,
                        "rule_id": "RULE_GOOGLE_NAME_PROPERTIES",
                        "severity": "error",
                        "observed_value": observed_value,
                        "action": "excluded",
                        "reason": "Experiment directory name disagrees with properties.yml",
                    })
                    trace_cols["source_record_id"].append(source_record_id)
                    trace_cols["source_name"].append("google_qec")
                    trace_cols["bronze_object"].append(bronze_object_name)
                    trace_cols["archive_member"].append(ym)
                    trace_cols["record_locator"].append("properties.yml")
                    trace_cols["input_sha256"].append(bronze_sha256)
                    trace_cols["valid"].append("no")
                    continue

                experiment_rows.append({
                    "source_record_id": source_record_id,
                    "experiment_id": exp_dir,
                    "basis": basis,
                    "distance": distance,
                    "rounds": rounds,
                    "shots": shots,
                    "center_row": center_row,
                    "center_col": center_col,
                    "measurement_count": meas_count,
                    "detector_count": det_count,
                })

                trace_cols["source_record_id"].append(source_record_id)
                trace_cols["source_name"].append("google_qec")
                trace_cols["bronze_object"].append(bronze_object_name)
                trace_cols["archive_member"].append(ym)
                trace_cols["record_locator"].append("properties.yml")
                trace_cols["input_sha256"].append(bronze_sha256)
                trace_cols["valid"].append("yes")

            except Exception as e:
                issues.append({
                    "issue_id": _stable_issue_id(
                        source_record_id, "RULE_EXP_PROPERTIES", ym
                    ),
                    "run_id": run_id,
                    "source_record_id": source_record_id,
                    "rule_id": "RULE_EXP_PROPERTIES",
                    "severity": "error",
                    "observed_value": ym,
                    "action": "excluded",
                    "reason": str(e),
                })
                trace_cols["source_record_id"].append(source_record_id)
                trace_cols["source_name"].append("google_qec")
                trace_cols["bronze_object"].append(bronze_object_name)
                trace_cols["archive_member"].append(ym)
                trace_cols["record_locator"].append("properties.yml")
                trace_cols["input_sha256"].append(bronze_sha256)
                trace_cols["valid"].append("no")

    # from_pylist keeps the schema even when every experiment was rejected.
    table_experiment = pa.Table.from_pylist(experiment_rows, schema=GOOGLE_EXPERIMENT_SCHEMA)
    return table_experiment, trace_cols, input_records_count


# ==============================================================================
# 3. GOOGLE QEC SHOTS PARSER WITH MULTI-MEMBER COMPANION TRACING
# ==============================================================================

def prepare_google_shots(
    bronze_bytes: bytes,
    bronze_object_name: str,
    bronze_sha256: str,
    run_id: str,
    issues: list[dict[str, Any]],
    checks: dict[str, dict[str, Any]] | None = None,
    excluded_experiments: set[str] | None = None,
) -> tuple[pa.Table, dict[str, list[Any]], int]:
    """Parse aligned hardware shots across 8 companion member files.

    Generates multi-member companion lineage traces for every shot, documenting
    exact member byte slices and lines with valid='yes'/'no'.
    """
    source_record_ids = []
    experiment_ids = []
    shot_indices = []
    measurement_bits_list = []
    sweep_bits_list = []
    detector_bits_list = []
    detector_event_counts = []
    actual_flips_list = []
    bm_preds = []
    cm_preds = []
    pym_preds = []
    tnc_preds = []

    trace_cols: dict[str, list[Any]] = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }

    input_records_count = 0

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as z:
        member_set = set(z.namelist())
        yaml_members = sorted([name for name in member_set if name.endswith("properties.yml") and not name.startswith("__MACOSX")])

        for ym in yaml_members:
            exp = ym.split("/")[0]
            prop = yaml.safe_load(z.read(ym).decode("utf-8"))

            shots = int(prop["shots"])
            num_meas = int(prop["circuit_measurements"])
            num_det = int(prop["circuit_detectors"])
            num_sweep = int(prop.get("circuit_sweep_bits", 0))
            input_records_count += shots

            # Shots of an experiment rejected in the experiment table are not
            # published either; the issue is recorded once, at experiment level.
            if excluded_experiments and exp in excluded_experiments:
                continue

            meas_stride = b8_record_bytes(num_meas)
            det_stride = b8_record_bytes(num_det)
            sweep_stride = b8_record_bytes(num_sweep) if num_sweep > 0 else 0

            # Mandatory companion files
            det_file = f"{exp}/detection_events.b8"
            meas_file = f"{exp}/measurements.b8"
            sweep_file = f"{exp}/sweep.b8"
            # sweep.b8 is required whenever properties.yml declares sweep bits.
            has_sweep = num_sweep > 0
            actual_file = f"{exp}/obs_flips_actual.01"

            decoder_files = {
                dec: f"{exp}/obs_flips_predicted_by_{dec}.01"
                for dec in DECODER_NAMES
            }

            # Check presence of mandatory companion files
            required_companions = [det_file, meas_file, actual_file, *decoder_files.values()]
            if has_sweep:
                required_companions.append(sweep_file)
            missing_companions = [f for f in required_companions if f not in member_set]
            _record_check(checks, "RULE_GOOGLE_MISSING_COMPANION")
            if missing_companions:
                issues.append({
                    "issue_id": _stable_issue_id(
                        f"google_qec:{exp}",
                        "RULE_GOOGLE_MISSING_COMPANION",
                        str(missing_companions),
                    ),
                    "run_id": run_id,
                    "source_record_id": f"google_qec:{exp}",
                    "rule_id": "RULE_GOOGLE_MISSING_COMPANION",
                    "severity": "error",
                    "observed_value": str(missing_companions),
                    "action": "stopped run",
                    "reason": f"Experiment {exp} missing required companion files",
                })
                # Brief: "a missing required companion file ... must stop the run."
                raise MissingCompanionFileError(
                    f"Experiment {exp} is missing required companion files: {missing_companions}"
                )

            # Read all companion streams into memory
            det_raw = z.read(det_file)
            meas_raw = z.read(meas_file)
            sweep_raw = z.read(sweep_file) if has_sweep else b""
            _record_check(checks, "RULE_GOOGLE_01_BINARY", checked=1 + len(decoder_files))
            act_raw = parse_01_records(z.read(actual_file))

            dec_streams = {}
            for dec, dfpath in decoder_files.items():
                dec_streams[dec] = parse_01_records(z.read(dfpath))

            # Pre-validate companion file sizes / line counts
            _record_check(checks, "RULE_GOOGLE_COMPANION_STRIDE_MISMATCH")
            if (
                len(det_raw) != shots * det_stride
                or len(meas_raw) != shots * meas_stride
                or (has_sweep and len(sweep_raw) != shots * sweep_stride)
                or len(act_raw) != shots
                or any(len(dec_streams[d]) != shots for d in DECODER_NAMES)
            ):
                issues.append({
                    "issue_id": _stable_issue_id(
                        f"google_qec:{exp}",
                        "RULE_GOOGLE_COMPANION_STRIDE_MISMATCH",
                        f"det:{len(det_raw)}, meas:{len(meas_raw)}, act:{len(act_raw)}",
                    ),
                    "run_id": run_id,
                    "source_record_id": f"google_qec:{exp}",
                    "rule_id": "RULE_GOOGLE_COMPANION_STRIDE_MISMATCH",
                    "severity": "error",
                    "observed_value": f"det:{len(det_raw)}, meas:{len(meas_raw)}, act:{len(act_raw)}",
                    "action": "excluded",
                    "reason": "Companion file byte length or line count does not match declared shots",
                })
                continue

            # Assemble aligned shots
            for i in range(shots):
                shot_id = f"google_qec:{exp}:shot:{i}"

                m_chunk = meas_raw[i * meas_stride : (i + 1) * meas_stride]
                d_chunk = det_raw[i * det_stride : (i + 1) * det_stride]
                sw_chunk = sweep_raw[i * sweep_stride : (i + 1) * sweep_stride] if has_sweep else b""

                # Verify unused padding bits are strictly zero
                det_rem = num_det % 8
                meas_rem = num_meas % 8
                is_valid = True
                invalid_reason = ""

                invalid_value = ""
                _record_check(checks, "RULE_GOOGLE_PADDING_BITS")
                padded_records = [
                    ("detection_events.b8", d_chunk, det_rem),
                    ("measurements.b8", m_chunk, meas_rem),
                    ("sweep.b8", sw_chunk, num_sweep % 8 if has_sweep else 0),
                ]
                for file_name, chunk, used_bits in padded_records:
                    if used_bits and (chunk[-1] >> used_bits) != 0:
                        is_valid = False
                        invalid_reason = f"Non-zero padding bits in {file_name}"
                        invalid_value = f"{file_name}:last_byte={chunk[-1]}"
                        break

                event_cnt = int.from_bytes(d_chunk, "little").bit_count()

                bm = bool(dec_streams["belief_matching"][i])
                cm = bool(dec_streams["correlated_matching"][i])
                pym = bool(dec_streams["pymatching"][i])
                tnc = bool(dec_streams["tensor_network_contraction"][i])
                actual_val = bool(act_raw[i])

                trace_status = "yes" if is_valid else "no"

                # Multi-member companion tracing: Every shot records all its constituent companion members
                companion_members = [
                    (det_file, f"shot:{i}"),
                    (meas_file, f"shot:{i}"),
                    (actual_file, f"line:{i}"),
                    (decoder_files["belief_matching"], f"line:{i}"),
                    (decoder_files["correlated_matching"], f"line:{i}"),
                    (decoder_files["pymatching"], f"line:{i}"),
                    (decoder_files["tensor_network_contraction"], f"line:{i}"),
                ]
                if has_sweep:
                    companion_members.append((sweep_file, f"shot:{i}"))

                for member_path, locator in companion_members:
                    trace_cols["source_record_id"].append(shot_id)
                    trace_cols["source_name"].append("google_qec")
                    trace_cols["bronze_object"].append(bronze_object_name)
                    trace_cols["archive_member"].append(member_path)
                    trace_cols["record_locator"].append(locator)
                    trace_cols["input_sha256"].append(bronze_sha256)
                    trace_cols["valid"].append(trace_status)

                if not is_valid:
                    issues.append({
                        "issue_id": _stable_issue_id(
                            shot_id,
                            "RULE_GOOGLE_PADDING_BITS",
                            invalid_value,
                        ),
                        "run_id": run_id,
                        "source_record_id": shot_id,
                        "rule_id": "RULE_GOOGLE_PADDING_BITS",
                        "severity": "error",
                        "observed_value": invalid_value,
                        "action": "excluded",
                        "reason": invalid_reason,
                    })
                    continue

                # Append valid row to Silver columnar arrays
                source_record_ids.append(shot_id)
                experiment_ids.append(exp)
                shot_indices.append(i)
                measurement_bits_list.append(m_chunk)
                sweep_bits_list.append(sw_chunk)
                detector_bits_list.append(d_chunk)
                detector_event_counts.append(event_cnt)
                actual_flips_list.append(actual_val)
                bm_preds.append(bm)
                cm_preds.append(cm)
                pym_preds.append(pym)
                tnc_preds.append(tnc)

    table_shot = pa.Table.from_arrays(
        [
            pa.array(source_record_ids, type=pa.string()),
            pa.array(experiment_ids, type=pa.string()),
            pa.array(shot_indices, type=pa.int64()),
            pa.array(measurement_bits_list, type=pa.binary()),
            pa.array(sweep_bits_list, type=pa.binary()),
            pa.array(detector_bits_list, type=pa.binary()),
            pa.array(detector_event_counts, type=pa.int32()),
            pa.array(actual_flips_list, type=pa.bool_()),
            pa.array(bm_preds, type=pa.bool_()),
            pa.array(cm_preds, type=pa.bool_()),
            pa.array(pym_preds, type=pa.bool_()),
            pa.array(tnc_preds, type=pa.bool_()),
        ],
        schema=GOOGLE_SHOT_SCHEMA,
    )

    return table_shot, trace_cols, input_records_count


# ==============================================================================
# 4. QASMBENCH HOOK 
# ==============================================================================

def prepare_qasmbench(
    bronze_bytes: bytes | None,
    bronze_object_name: str,
    bronze_sha256: str | None,
    run_id: str,
    base_dir: Path,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, pa.Table], dict[str, list[Any]], dict[str, dict[str, int | str]]]:
    """
    Produces the tables from the QASMBench sources. 
    """
    silver_qasm_tables: dict[str, pa.Table] = {}
    trace_cols: dict[str, list[Any]] = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }
    # execute if tables are not present using code from the notebooks 

    # compose circuit table and traces:  
    silver_qasm_tables['circuit'], circuit_trace_cols, circuit_rows_read, circuit_rows_rejected = _create_qasmbench_circuit_table(
        bronze_bytes, bronze_object_name, bronze_sha256, run_id, issues
    )
    _extend_trace_cols(trace_cols, circuit_trace_cols)

    # compose stabilizer checks table and traces:  
    silver_qasm_tables['stabilizer_check'], sc_trace_cols, sc_rows_read, sc_rows_rejected = _create_qasmbench_stabilizer_checks_table(
        bronze_bytes, bronze_object_name, bronze_sha256, run_id, issues
    )
    _extend_trace_cols(trace_cols, sc_trace_cols)

    # compose conditional corrections table and traces: 
    silver_qasm_tables['conditional_correction'], cc_trace_cols, cc_rows_read, cc_rows_rejected = _create_qasmbench_conditional_correction_table(
        bronze_bytes, bronze_object_name, bronze_sha256, run_id, issues
    )
    _extend_trace_cols(trace_cols, cc_trace_cols)

    source_counts = {
        "circuit": (circuit_rows_read, circuit_rows_rejected),
        "stabilizer_check": (sc_rows_read, sc_rows_rejected),
        "conditional_correction": (cc_rows_read, cc_rows_rejected),
    }
    row_units = {
        "circuit": "circuit row",
        "stabilizer_check": "stabilizer check row",
        "conditional_correction": "conditional correction row",
    }
    row_counts = {}
    for name, table in silver_qasm_tables.items():
        source_files_read, source_files_rejected = source_counts[name]
        child_rows = table.num_rows
        row_counts[name] = _row_counts(
            child_rows,
            child_rows,
            child_rows,
            row_units[name],
        )
        row_counts[name]["source_files_read"] = source_files_read
        row_counts[name]["source_files_rejected"] = source_files_rejected
    
    return silver_qasm_tables, trace_cols, row_counts


def _extend_trace_cols(target_trace_cols, incoming_trace_cols): 
    """Helper function for extending tracer cols with incoming tracer values from a newly created table"""
    for key in target_trace_cols: 
        target_trace_cols[key].extend(incoming_trace_cols[key])

def _create_qasmbench_circuit_table(bronze_bytes, bronze_object_name, bronze_sha256, run_id, issues):
    """Parse QASM circuit members into the silver circuit table and lineage traces."""
    if bronze_bytes is None:
        empty_table = pa.Table.from_pylist([], schema=QASMBENCH_CIRCUIT_SCHEMA)
        empty_trace = {
            "source_record_id": [],
            "source_name": [],
            "bronze_object": [],
            "archive_member": [],
            "record_locator": [],
            "input_sha256": [],
            "valid": [],
        }
        return empty_table, empty_trace, 0, 0

    circuit_records = []
    rows_read = 0
    rows_rejected = 0
    trace_cols = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as archive:
        qasm_members = sorted(name for name in archive.namelist() if name.endswith(".qasm"))
        for member in qasm_members:
            rows_read += 1
            try:
                record = _parse_circuit(member, archive.read(member).decode("utf-8"))
            except Exception as exc:
                rows_rejected += 1
                _append_qasm_issue(
                    issues,
                    run_id,
                    f"qasmbench:{member}",
                    "RULE_QASM_CIRCUIT_PARSE",
                    member,
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            circuit_records.append(record)

            trace_cols["source_record_id"].append(record["source_record_id"])
            trace_cols["source_name"].append("qasmbench")
            trace_cols["bronze_object"].append(bronze_object_name)
            trace_cols["archive_member"].append(member)
            trace_cols["record_locator"].append("qasm_program")
            trace_cols["input_sha256"].append(bronze_sha256)
            trace_cols["valid"].append("yes")

            print(
                f"  {record['benchmark_name']}/{record['variant']}: "
                f"{record['qubit_count']} qubits, {record['measurement_count']} measurements, "
                f"{record['two_qubit_gate_count']} two-qubit gates"
            )

    table = pa.Table.from_pylist(circuit_records, schema=QASMBENCH_CIRCUIT_SCHEMA)
    return table, trace_cols, rows_read, rows_rejected


def _create_qasmbench_stabilizer_checks_table(bronze_bytes, bronze_object_name, bronze_sha256, run_id, issues): 
    """Parse QASMbench bronze objects and produce the stabilizer-check table."""
    if bronze_bytes is None:
        empty_table = pa.Table.from_pylist([], schema=QASMBENCH_STABILIZER_CHECK_SCHEMA)
        empty_trace = {
            "source_record_id": [],
            "source_name": [],
            "bronze_object": [],
            "archive_member": [],
            "record_locator": [],
            "input_sha256": [],
            "valid": [],
        }
        return empty_table, empty_trace, 0, 0

    sc_records = []
    rows_read = 0
    rows_rejected = 0
    trace_cols = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }
    input_sha256 = bronze_sha256 or "unknown"

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as archive:
        qasm_members = sorted(name for name in archive.namelist() if name.endswith(".qasm"))
        for member in qasm_members:
            rows_read += 1
            try:
                records = _parse_stabilizer_checks(member, archive.read(member).decode("utf-8"))
            except Exception as exc:
                rows_rejected += 1
                _append_qasm_issue(
                    issues,
                    run_id,
                    f"qasmbench:{member}",
                    "RULE_QASM_STABILIZER_PARSE",
                    member,
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            for rec in records:
                sc_records.append({
                    "source_record_id": rec["source_record_id"],
                    "circuit_id": rec["circuit_id"],
                    "check_id": rec["check_id"],
                    "ancilla_qubit": rec["ancilla_qubit"],
                    "data_qubits": rec["data_qubits"],
                    "syndrome_bit": rec["syndrome_bit"],
                })
                trace_cols["source_record_id"].append(rec["source_record_id"])
                trace_cols["source_name"].append("qasmbench")
                trace_cols["bronze_object"].append(bronze_object_name)
                trace_cols["archive_member"].append(member)
                trace_cols["record_locator"].append(rec["record_locator"])
                trace_cols["input_sha256"].append(input_sha256)
                trace_cols["valid"].append("yes")

            print(f"  {Path(member).parent.name}/{Path(member).stem}: {len(records)} stabilizer checks")

    table = pa.Table.from_pylist(sc_records, schema=QASMBENCH_STABILIZER_CHECK_SCHEMA)
    return table, trace_cols, rows_read, rows_rejected


def _create_qasmbench_conditional_correction_table(bronze_bytes, bronze_object_name, bronze_sha256, run_id, issues): 
    """Parse QASMbench bronze objects and produce the conditional-correction table."""
    if bronze_bytes is None:
        empty_table = pa.Table.from_pylist([], schema=QASMBENCH_CONDITIONAL_CORRECTION_SCHEMA)
        empty_trace = {
            "source_record_id": [],
            "source_name": [],
            "bronze_object": [],
            "archive_member": [],
            "record_locator": [],
            "input_sha256": [],
            "valid": [],
        }
        return empty_table, empty_trace, 0, 0

    cc_records = []
    rows_read = 0
    rows_rejected = 0
    trace_cols = {
        "source_record_id": [],
        "source_name": [],
        "bronze_object": [],
        "archive_member": [],
        "record_locator": [],
        "input_sha256": [],
        "valid": [],
    }
    input_sha256 = bronze_sha256 or "unknown"

    with zipfile.ZipFile(io.BytesIO(bronze_bytes)) as archive:
        qasm_members = sorted(name for name in archive.namelist() if name.endswith(".qasm"))
        for member in qasm_members:
            rows_read += 1
            try:
                records = _parse_conditional_corrections(member, archive.read(member).decode("utf-8"))
            except Exception as exc:
                rows_rejected += 1
                _append_qasm_issue(
                    issues,
                    run_id,
                    f"qasmbench:{member}",
                    "RULE_QASM_CORRECTION_PARSE",
                    member,
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            for rec in records:
                cc_records.append({
                    "source_record_id": rec["source_record_id"],
                    "circuit_id": rec["circuit_id"],
                    "condition_register": rec["condition_register"],
                    "condition_value": rec["condition_value"],
                    "gate": rec["gate"],
                    "target_qubit": rec["target_qubit"],
                })
                trace_cols["source_record_id"].append(rec["source_record_id"])
                trace_cols["source_name"].append("qasmbench")
                trace_cols["bronze_object"].append(bronze_object_name)
                trace_cols["archive_member"].append(rec.get("archive_member", member))
                trace_cols["record_locator"].append(rec.get("record_locator", "statement"))
                trace_cols["input_sha256"].append(input_sha256)
                trace_cols["valid"].append("yes")

            print(f"  {Path(member).parent.name}/{Path(member).stem}: {len(records)} conditional corrections")

    table = pa.Table.from_pylist(cc_records, schema=QASMBENCH_CONDITIONAL_CORRECTION_SCHEMA)
    return table, trace_cols, rows_read, rows_rejected



# Constants for computing the QASMBench circuit table 

TWO_QUBIT_GATES = {"cx", "cnot", "cz", "ch", "cy", "swap", "cu1", "crz"}
REGISTER_DECLARATION = re.compile(r"^(qreg|creg)\s+(\w+)\s*\[\s*(\d+)\s*\]$")
GATE_HEADER = re.compile(r"gate\s+(\w+)\s*([^{}]*)\{(?P<body>.*?)\}", re.DOTALL)

# Helper functions for computing the QASMbench circuit table 

def _operand_width(operand, registers):
    match = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", operand.strip())
    assert match, f"Unsupported operand: {operand}"
    name, index = match.groups()
    assert name in registers, f"Unknown register: {name}"
    return 1 if index is not None else registers[name]

def _expanded_operation_count(operands, registers):
    widths = [_operand_width(operand, registers) for operand in operands]
    assert widths and len(set(widths)) == 1, f"Register widths do not align: {operands}"
    return widths[0]

def _body_two_qubit_count(body):
    count = 0
    for statement in body.split(";"):
        tokens = statement.strip().split(None, 1)
        if tokens and tokens[0].lower() in TWO_QUBIT_GATES:
            count += 1
    return count


def _body_operation_count(body):
    count = 0
    for statement in body.split(";"):
        statement = " ".join(statement.split())
        if not statement or statement.startswith(("OPENQASM", "include", "barrier", "opaque", "measure")):
            continue
        if len(statement.split(None, 1)) == 2:
            count += 1
    return count

def _parse_circuit(member, text):
    cleaned = re.sub(r"//.*", "", text)
    gate_definitions = {}
    for match in GATE_HEADER.finditer(cleaned):
        gate_definitions[match.group(1)] = {
            "formal_count": len([item for item in match.group(2).split(",") if item.strip()]),
            "two_qubit_count": _body_two_qubit_count(match.group("body")),
            "operation_count": _body_operation_count(match.group("body")),
        }
    executable_text = GATE_HEADER.sub("", cleaned)

    registers = {}
    qreg_sizes = {}
    register_lines = []
    measurement_count = 0
    operation_count = 0
    two_qubit_gate_count = 0
    for statement in executable_text.replace("{", " ").replace("}", " ").split(";"):
        statement = " ".join(statement.split())
        if not statement or statement.startswith(("OPENQASM", "include", "barrier", "opaque")):
            continue
        declaration = REGISTER_DECLARATION.fullmatch(statement)
        if declaration:
            kind, name, size = declaration.groups()
            registers[name] = int(size)
            if kind == "qreg":
                qreg_sizes[name] = int(size)
            register_lines.append(f"{kind} {name}[{size}]")
            continue
        if statement.startswith("measure "):
            left = statement[len("measure "):].split("->", 1)[0].strip()
            measurement_count += _operand_width(left, registers)
            continue
        if statement.startswith("if("):
            statement = statement.split(")", 1)[1].strip()
        tokens = statement.split(None, 1)
        if len(tokens) != 2:
            continue
        gate_name, operand_text = tokens
        operands = [item.strip() for item in operand_text.split(",")]
        gate_key = gate_name.lower()
        if gate_key in TWO_QUBIT_GATES:
            expanded_count = _expanded_operation_count(operands, registers)
            operation_count += expanded_count
            two_qubit_gate_count += expanded_count
        elif gate_name in gate_definitions:
            widths = [_operand_width(item, registers) for item in operands]
            assert len(widths) == gate_definitions[gate_name]["formal_count"]
            assert len(set(widths)) == 1, f"Custom gate widths do not align: {statement}"
            operation_count += gate_definitions[gate_name]["operation_count"] * widths[0]
            two_qubit_gate_count += gate_definitions[gate_name]["two_qubit_count"] * widths[0]
        else:
            operation_count += _expanded_operation_count(operands, registers)

    benchmark_name = Path(member).parent.name
    filename = Path(member).stem
    variant = "transpiled" if filename.endswith("_transpiled") else "source"
    return {
        "source_record_id": f"qasmbench:{member}",
        "circuit_id": f"qasmbench:{member[:-5]}",
        "benchmark_name": benchmark_name,
        "variant": variant,
        "register_declarations": "; ".join(register_lines),
        "qubit_count": sum(qreg_sizes.values()),
        "operation_count": operation_count,
        "measurement_count": measurement_count,
        "two_qubit_gate_count": two_qubit_gate_count,
    }


# Constants for Stabilizer Checks table 

CNOT_STATEMENT = re.compile(r"^cx\s+([^,]+)\s*,\s*(.+)$", re.IGNORECASE)
REGISTER_DECLARATION = re.compile(r"^(qreg|creg)\s+(\w+)\s*\[\s*(\d+)\s*\]$")
GATE_HEADER = re.compile(r"gate\s+(\w+)\s*([^{}]*)\{(?P<body>.*?)\}", re.DOTALL)


# Helper functions for Stabilizer Checks builder 

def _normalize_operand(operand):
    return "".join(operand.strip().split())

def _parse_registers(text):
    registers = {}
    for statement in text.split(";"):
        statement = " ".join(statement.split())
        match = REGISTER_DECLARATION.fullmatch(statement)
        if match:
            kind, name, size = match.groups()
            registers[name] = (kind, int(size))
    return registers

def _expand_operand(operand, registers):
    operand = _normalize_operand(operand)
    match = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", operand)
    assert match, f"Unsupported operand: {operand}"
    name, index = match.groups()
    assert name in registers, f"Unknown register: {name}"
    kind, size = registers[name]
    if index is not None:
        assert int(index) < size, f"Register index out of range: {operand}"
        return [f"{name}[{index}]"]
    return [f"{name}[{position}]" for position in range(size)]

def _expand_pairs(left, right, registers):
    left_values = _expand_operand(left, registers)
    right_values = _expand_operand(right, registers)
    assert len(left_values) == len(right_values), f"CNOT registers do not align: {left}, {right}"
    return list(zip(left_values, right_values))


def _mask_gate_definitions(text):
    return GATE_HEADER.sub(
        lambda match: "".join(
            char if char in "\r\n;" else " " for char in match.group(0)
        ),
        text,
    )

def _parse_stabilizer_checks(member, text):
    cleaned = re.sub(r"//.*", "", text)
    registers = _parse_registers(cleaned)
    measurement_map = {}
    measurement_statement_map = {}
    for statement_index, statement in enumerate(cleaned.split(";")):
        statement = " ".join(statement.split())
        if not statement.startswith("measure "):
            continue
        measured, destination = [part.strip() for part in statement[len("measure "):].split("->", 1)]
        measured_values = _expand_operand(measured, registers)
        destination_values = _expand_operand(destination, registers)
        assert len(measured_values) == len(destination_values)
        measurement_map.update(zip(measured_values, destination_values))
        measurement_statement_map.update(
            (qubit, statement_index) for qubit in measured_values
        )

    gate_definitions = {}
    for match in GATE_HEADER.finditer(cleaned):
        formal_names = [item.strip() for item in match.group(2).split(",") if item.strip()]
        body_pairs = []
        body_offset = match.start("body")
        current_offset = 0
        body = match.group("body")
        for raw_statement in body.split(";"):
            leading_whitespace = len(raw_statement) - len(raw_statement.lstrip())
            statement = " ".join(raw_statement.split())
            body_match = CNOT_STATEMENT.fullmatch(statement)
            if body_match:
                body_statement_index = cleaned.count(
                    ";", 0, body_offset + current_offset + leading_whitespace
                )
                body_pairs.append((
                    body_match.group(1).strip(),
                    body_match.group(2).strip(),
                    body_statement_index,
                ))
            current_offset += len(raw_statement) + 1
        gate_definitions[match.group(1)] = (formal_names, body_pairs)

    executable_text = _mask_gate_definitions(cleaned)
    interactions = []
    for statement_index, statement in enumerate(executable_text.split(";")):
        statement = " ".join(statement.split())
        if not statement or statement.startswith(("OPENQASM", "include", "barrier", "opaque", "measure", "qreg", "creg")):
            continue
        tokens = statement.split(None, 1)
        if len(tokens) != 2:
            continue
        operation, operand_text = tokens
        if operation.lower() == "cx":
            left, right = [part.strip() for part in operand_text.split(",", 1)]
            interactions.extend(
                (data_qubit, ancilla_qubit, {statement_index})
                for data_qubit, ancilla_qubit in _expand_pairs(left, right, registers)
            )
        elif operation in gate_definitions:
            formal_names, body_pairs = gate_definitions[operation]
            actual_operands = [part.strip() for part in operand_text.split(",")]
            assert len(formal_names) == len(actual_operands), f"Custom gate arguments do not align: {statement}"
            substitutions = dict(zip(formal_names, actual_operands))
            for left, right, definition_statement_index in body_pairs:
                mapped_left = substitutions[left]
                mapped_right = substitutions[right]
                interactions.extend(
                    (data_qubit, ancilla_qubit, {statement_index, definition_statement_index})
                    for data_qubit, ancilla_qubit in _expand_pairs(
                        mapped_left, mapped_right, registers
                    )
                )

    by_ancilla = {}
    statement_indices_by_ancilla = {}
    for data_qubit, ancilla_qubit, statement_indices in interactions:
        by_ancilla.setdefault(ancilla_qubit, set()).add(data_qubit)
        statement_indices_by_ancilla.setdefault(ancilla_qubit, set()).update(statement_indices)

    benchmark_name = Path(member).parent.name
    variant = "transpiled" if Path(member).stem.endswith("_transpiled") else "source"
    circuit_id = f"qasmbench:{member[:-5]}"
    records = []
    for check_number, ancilla_qubit in enumerate(sorted(by_ancilla)):
        data_qubits = sorted(by_ancilla[ancilla_qubit])
        syndrome_bit = measurement_map.get(ancilla_qubit)
        if len(data_qubits) < 2 or syndrome_bit is None:
            continue
        statement_indices = statement_indices_by_ancilla[ancilla_qubit]
        statement_indices.add(measurement_statement_map[ancilla_qubit])
        check_id = f"{circuit_id}:check:{check_number}"
        records.append({
            "source_record_id": f"qasmbench:{member}:check:{check_number}",
            "circuit_id": circuit_id,
            "check_id": check_id,
            "ancilla_qubit": ancilla_qubit,
            "data_qubits": data_qubits,
            "syndrome_bit": syndrome_bit,
            "record_locator": "statements:" + ",".join(
                str(index) for index in sorted(statement_indices)
            ),
        })
    return records

# Constants for Conditional Corrections 

CONDITIONAL_CORRECTION = re.compile(
    r"^if\s*\(\s*([A-Za-z_]\w*)\s*==\s*(-?\d+)\s*\)\s*([A-Za-z_]\w*)\s+([^;]+)$",
    re.IGNORECASE,
)

# Helper function for creating Conditional Corrections 

def _parse_conditional_corrections(member, text):
    cleaned = re.sub(r"//.*", "", text)
    circuit_id = f"qasmbench:{member[:-5]}"
    records = []
    for statement_index, statement in enumerate(cleaned.split(";")):
        statement = " ".join(statement.split())
        if not statement:
            continue
        match = CONDITIONAL_CORRECTION.fullmatch(statement)
        if not match:
            continue
        condition_register, condition_value, gate, target_qubit = match.groups()
        records.append({
            "source_record_id": f"qasmbench:{member}:statement:{statement_index}",
            "circuit_id": circuit_id,
            "condition_register": condition_register,
            "condition_value": int(condition_value),
            "gate": gate,
            "target_qubit": "".join(target_qubit.split()),
            "archive_member": member,
            "record_locator": f"statement:{statement_index}",
        })
    return records




# ==============================================================================
# MAIN STAGE 2 RUNNER
# ==============================================================================

def run(run_id: str, settings: Settings | None = None) -> StageResult:
    run_state: dict[str, Any] = {
        "base_dir": Path(__file__).resolve().parents[3],
        "input_hashes": {},
        "row_counts": {},
        "output_table_counts": {},
    }
    try:
        return _execute_run(run_id, settings, run_state)
    except Exception as exc:
        try:
            _record_failed_run(run_id, run_state.get("result"), run_state, exc)
        except Exception:
            logging.exception("Could not record failed prepare_data run %s", run_id)
        raise


def _record_same_syndrome_both_labels(
    table_syn: pa.Table,
    checks: dict[str, dict[str, Any]],
) -> None:
    """Brief: check "the valid case where one syndrome occurs with both labels"."""
    labels_by_syndrome: dict[tuple[str, bytes], set[bool]] = {}
    for experiment_id, bits, label in zip(
        table_syn.column("experiment_id").to_pylist(),
        table_syn.column("syndrome_bits").to_pylist(),
        table_syn.column("logical_error_label").to_pylist(),
    ):
        labels_by_syndrome.setdefault((experiment_id, bits), set()).add(label)
    both = sum(1 for labels in labels_by_syndrome.values() if len(labels) == 2)
    _record_check(
        checks,
        "RULE_SYN_SAME_SYNDROME_BOTH_LABELS",
        checked=len(labels_by_syndrome),
        observation=f"{both} syndromes occur with both labels within one file; all rows kept",
    )


def _execute_run(
    run_id: str,
    settings: Settings | None,
    run_state: dict[str, Any],
) -> StageResult:
    """Execute Stage 2: Parse, validate, clean, and write Silver tables and traces."""
    if settings is None:
        settings = Settings.from_environment()

    result = StageResult(stage="prepare_data", run_id=run_id)
    base_dir = Path(__file__).resolve().parents[3]  # starter/ root
    run_state["result"] = result
    run_state["settings"] = settings

    silver_base = base_dir / "silver"
    results_base = base_dir / "results/part1"
    silver_base.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)

    issues: list[dict[str, Any]] = []
    run_state["issues"] = issues
    checks: dict[str, dict[str, Any]] = {}
    total_inputs = 0
    total_outputs = 0

    # --------------------------------------------------------------------------
    # 1. Process Syndromes
    # --------------------------------------------------------------------------
    syn_obj = "bronze/source=qec_syndromes/syndromes_dataset.zip"
    syn_bytes, syn_sha = get_bronze_archive(syn_obj, settings, base_dir)
    run_state["input_hashes"][syn_obj] = syn_sha
    table_syn, trace_syn, in_syn = prepare_syndrome_observations(
        syn_bytes, syn_obj, syn_sha, run_id, issues, checks
    )
    _record_same_syndrome_both_labels(table_syn, checks)
    total_inputs += in_syn
    total_outputs += table_syn.num_rows

    syn_out_dir = silver_base / "qec_syndromes"
    syn_out_dir.mkdir(parents=True, exist_ok=True)
    syn_path = syn_out_dir / "syndrome_observation.parquet"
    pq.write_table(table_syn, syn_path, compression="zstd")
    _remember_table(
        run_state,
        "silver/qec_syndromes/syndrome_observation.parquet",
        _row_counts(in_syn, table_syn.num_rows, table_syn.num_rows, "CSV data row"),
    )

    # --------------------------------------------------------------------------
    # 2. Process Google QEC Experiments
    # --------------------------------------------------------------------------
    google_obj = "bronze/source=google_qec/google-surface-code-curated.zip"
    google_bytes, google_sha = get_bronze_archive(google_obj, settings, base_dir)
    run_state["input_hashes"][google_obj] = google_sha
    table_exp, trace_exp, in_exp = prepare_google_experiments(
        google_bytes, google_obj, google_sha, run_id, issues, checks
    )
    accepted_experiments = set(table_exp.column("experiment_id").to_pylist())
    excluded_experiments = {
        record_id.split(":")[1]
        for record_id in trace_exp["source_record_id"]
    } - accepted_experiments
    total_inputs += in_exp
    total_outputs += table_exp.num_rows

    google_out_dir = silver_base / "google_qec"
    google_out_dir.mkdir(parents=True, exist_ok=True)
    exp_path = google_out_dir / "experiment.parquet"
    pq.write_table(table_exp, exp_path, compression="zstd")
    _remember_table(
        run_state,
        "silver/google_qec/experiment.parquet",
        _row_counts(in_exp, table_exp.num_rows, table_exp.num_rows, "experiment properties file"),
    )

    # --------------------------------------------------------------------------
    # 3. Process Google QEC Shots
    # --------------------------------------------------------------------------
    table_shot, trace_shot, in_shot = prepare_google_shots(
        google_bytes, google_obj, google_sha, run_id, issues, checks, excluded_experiments
    )
    total_inputs += in_shot
    total_outputs += table_shot.num_rows

    shot_path = google_out_dir / "shot.parquet"
    pq.write_table(table_shot, shot_path, compression="zstd")
    _remember_table(
        run_state,
        "silver/google_qec/shot.parquet",
        _row_counts(in_shot, table_shot.num_rows, table_shot.num_rows, "declared shot"),
    )

    # --------------------------------------------------------------------------
    # 4. Process QASMBench 
    # --------------------------------------------------------------------------
    qasm_obj = "bronze/source=qasmbench/qasmbench-qec.zip"
    try:
        qasm_bytes, qasm_sha = get_bronze_archive(qasm_obj, settings, base_dir)
        with zipfile.ZipFile(io.BytesIO(qasm_bytes)) as archive:
            archive.namelist()
    except Exception as exc:
        qasm_bytes, qasm_sha = None, None
        _append_qasm_issue(
            issues,
            run_id,
            "qasmbench:archive",
            "RULE_QASM_ARCHIVE_UNAVAILABLE",
            qasm_obj,
            f"{type(exc).__name__}: {exc}",
        )
    run_state["input_hashes"][qasm_obj] = qasm_sha

    qasm_tables, trace_qasm, qasm_row_counts = prepare_qasmbench(
        qasm_bytes, qasm_obj, qasm_sha, run_id, base_dir, issues
    )
    # QASMBench rules are evaluated once per .qasm file in each table builder.
    _record_check(checks, "RULE_QASM_ARCHIVE_UNAVAILABLE")
    for rule_id, table_name in [
        ("RULE_QASM_CIRCUIT_PARSE", "circuit"),
        ("RULE_QASM_STABILIZER_PARSE", "stabilizer_check"),
        ("RULE_QASM_CORRECTION_PARSE", "conditional_correction"),
    ]:
        _record_check(checks, rule_id, checked=qasm_row_counts[table_name]["source_files_read"])
    total_inputs += sum(counts["rows_read"] for counts in qasm_row_counts.values())
    for q_name, q_table in qasm_tables.items():
        q_dir = silver_base / "qasmbench"
        q_dir.mkdir(parents=True, exist_ok=True)
        q_path = q_dir / f"{q_name}.parquet"
        pq.write_table(q_table, q_path, compression="zstd")
        _remember_table(
            run_state,
            f"silver/qasmbench/{q_name}.parquet",
            qasm_row_counts[q_name],
        )
        total_outputs += q_table.num_rows

    # --------------------------------------------------------------------------
    # 5. Save Lineage Traces (Combined across all sources)
    # --------------------------------------------------------------------------
    trace_path = results_base / "source_trace.parquet"
    # Merge columnar trace dictionaries
    all_trace_cols = {
        k: (
            trace_syn[k]
            + trace_exp[k]
            + trace_shot[k]
            + trace_qasm[k]
        )
        for k in trace_syn
    }
    trace_rows_loaded = save_source_traces(all_trace_cols, "all", trace_path, settings=settings)
    _remember_table(
        run_state,
        "results/part1/source_trace.parquet",
        _row_counts(trace_rows_loaded, trace_rows_loaded, trace_rows_loaded, "trace row"),
    )

    # --------------------------------------------------------------------------
    # 6. Save Data Issues Table
    # --------------------------------------------------------------------------
    issues_path = results_base / "data_issues.parquet"
    if issues:
        df_issues = pd.DataFrame(issues)
        table_issues = pa.Table.from_pandas(df_issues, schema=DATA_ISSUES_SCHEMA, preserve_index=False)
    else:
        table_issues = pa.Table.from_pylist([], schema=DATA_ISSUES_SCHEMA)

    pq.write_table(table_issues, issues_path, compression="zstd")
    _remember_table(
        run_state,
        "results/part1/data_issues.parquet",
        _row_counts(len(issues), len(issues), table_issues.num_rows, "issue finding"),
    )

    row_counts = {
        "silver/qec_syndromes/syndrome_observation.parquet": _row_counts(
            in_syn, table_syn.num_rows, table_syn.num_rows, "CSV data row"
        ),
        "silver/google_qec/experiment.parquet": _row_counts(
            in_exp, table_exp.num_rows, table_exp.num_rows, "experiment properties file"
        ),
        "silver/google_qec/shot.parquet": _row_counts(
            in_shot, table_shot.num_rows, table_shot.num_rows, "declared shot"
        ),
    }
    row_counts.update({
        f"silver/qasmbench/{table_name}.parquet": counts
        for table_name, counts in qasm_row_counts.items()
    })
    row_counts["results/part1/source_trace.parquet"] = _row_counts(
        trace_rows_loaded, trace_rows_loaded, trace_rows_loaded, "trace row"
    )
    row_counts["results/part1/data_issues.parquet"] = _row_counts(
        len(issues), len(issues), len(issues), "issue finding"
    )

    output_table_counts = {
        table_name: counts["rows_loaded"]
        for table_name, counts in row_counts.items()
    }
    result.input_count = total_inputs
    result.output_count = total_outputs
    result.issue_count = len(issues)
    result.finish()

    check_outcomes = _check_outcomes(checks, issues)
    row_counts_path = results_base / "row_counts.json"
    check_outcomes_path = results_base / "check_outcomes.json"
    _write_json(row_counts_path, {"run_id": run_id, "tables": row_counts})
    _write_json(check_outcomes_path, {"run_id": run_id, "checks": check_outcomes})
    # run.json is written once per run by the CLI from these details.
    result.details["output_table_counts"] = output_table_counts
    result.details["checks"] = check_outcomes

    # --------------------------------------------------------------------------
    # 7. Upload to MinIO (if configured)
    # --------------------------------------------------------------------------
    if settings.lake_backend == "minio":
        try:
            client = minio_client(settings)
            client.fput_object(settings.s3_bucket, "silver/qec_syndromes/syndrome_observation.parquet", str(syn_path))
            client.fput_object(settings.s3_bucket, "silver/google_qec/experiment.parquet", str(exp_path))
            client.fput_object(settings.s3_bucket, "silver/google_qec/shot.parquet", str(shot_path))
            for q_name in qasm_tables:
                client.fput_object(settings.s3_bucket, f"silver/qasmbench/{q_name}.parquet", str(silver_base / "qasmbench" / f"{q_name}.parquet"))
            client.fput_object(settings.s3_bucket, "results/part1/data_issues.parquet", str(issues_path))
            client.fput_object(settings.s3_bucket, "results/part1/row_counts.json", str(row_counts_path))
            client.fput_object(settings.s3_bucket, "results/part1/check_outcomes.json", str(check_outcomes_path))
        except Exception:
            logging.exception("MinIO upload failed during prepare_data run %s", run_id)
            raise

    return result
