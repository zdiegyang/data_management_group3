"""Step 7 of getting-started: ONE record per source from bronze -> silver -> gold -> ml, then traced back.

Run from starter/:  python -m quantum_lake_student.slice_demo

LAKE_BACKEND=minio (default in the workspace container): Bronze is READ from the MinIO bucket with the
MinIO client (byte-range requests, so zips are never fully downloaded/extracted); the slice's Silver/ML
files are WRITTEN back to the bucket under the `slice/` prefix so they never collide with real outputs.
LAKE_BACKEND=local: same code, but reads ../datasets/.../raw and writes ./lake/slice/.
This is a throw-away vertical slice to prove the layers connect; the real stages replace it.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.formats import b8_record_bytes, parse_01_records
from quantum_lake_student.ml import google_data_split, syndrome_data_split, unpack_little_endian_bits

# Paths are resolved from THIS file, not from the current directory, so it works from anywhere.
PROJECT = Path(__file__).resolve().parents[2]              # .../starter  (== /workspace in the container)
SQL = PROJECT / "sql"
LOCAL_RAW = next((p for p in (PROJECT.parent / "datasets/student-bundle/core/raw",   # on your laptop
                              Path("/course-data/raw"))                                 # in the container
                  if p.exists()), PROJECT.parent / "datasets/student-bundle/core/raw")  # only for LAKE_BACKEND=local
SLICE = "slice"                                            # output prefix, keeps us away from real zones

SYN_KEY = "bronze/source=qec_syndromes/syndromes_dataset.zip"
GOOGLE_KEY = "bronze/source=google_qec/google-surface-code-curated.zip"

# ---- which single records we follow --------------------------------------------------------
SYN_MEMBER, SYN_ROW = "d-3_pfr-0.000500_nb-10M.csv", 2      # 0-based data row in that CSV
G_EXP, G_SHOT = "surface_code_bX_d3_r25_center_3_5", 0
DECODERS = ["belief_matching", "correlated_matching", "pymatching", "tensor_network_contraction"]


# ============================== object-store access ==========================================
class MinioRangeFile(io.RawIOBase):
    """Read-only, seekable file object over one MinIO object using HTTP range GETs.

    zipfile needs seek()+read(); this lets it read the zip's central directory and ONE member without
    downloading the whole archive. Counts requests/bytes so you can see how little was transferred.
    """

    def __init__(self, client, bucket: str, key: str):
        st = client.stat_object(bucket, key)
        self._c, self._bucket, self._key = client, bucket, key
        self.size, self.etag, self.version_id = st.size, st.etag, st.version_id
        self._pos, self.requests, self.bytes_fetched = 0, 0, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, b):
        n = min(len(b), self.size - self._pos)
        if n <= 0:
            return 0
        resp = self._c.get_object(self._bucket, self._key, offset=self._pos, length=n,
                                  version_id=self.version_id)   # pin to the version we stat'ed
        try:
            data = resp.read()
        finally:
            resp.close(); resp.release_conn()
        b[:len(data)] = data
        self._pos += len(data)
        self.requests += 1; self.bytes_fetched += len(data)
        return len(data)


def _sha256_stream(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


class Lake:
    """Tiny facade so the rest of the script is identical for MinIO and local backends."""

    def __init__(self, s: Settings):
        self.s = s
        self.minio = s.lake_backend == "minio"
        self.client = minio_client(s) if self.minio else None
        self._ranged: list[tuple[str, MinioRangeFile]] = []

    # ---- Bronze (read-only) ----
    def open_zip(self, key: str) -> zipfile.ZipFile:
        if not self.minio:
            return zipfile.ZipFile(LOCAL_RAW / key.removeprefix("bronze/"))
        raw = MinioRangeFile(self.client, self.s.s3_bucket, key)
        self._ranged.append((key, raw))
        return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=256 * 1024))

    def sha256(self, key: str) -> str:
        """sha256 of a whole Bronze object (streamed; used for the trace's input_sha256)."""
        if not self.minio:
            with open(LOCAL_RAW / key.removeprefix("bronze/"), "rb") as f:
                return _sha256_stream(iter(lambda: f.read(1 << 20), b""))
        resp = self.client.get_object(self.s.s3_bucket, key)
        try:
            return _sha256_stream(resp.stream(1 << 20))
        finally:
            resp.close(); resp.release_conn()

    def bronze_report(self) -> None:
        for key, raw in self._ranged:
            print(f"  {key}: read {raw.bytes_fetched:,} of {raw.size:,} bytes "
                  f"({100 * raw.bytes_fetched / raw.size:.1f}%) in {raw.requests} range request(s); "
                  f"etag={raw.etag[:12]} version={raw.version_id}")

    # ---- Silver / ML zones ----
    def uri(self, rel: str) -> str:
        return f"s3://{self.s.s3_bucket}/{SLICE}/{rel}" if self.minio else str(self.s.local_lake_root / SLICE / rel)

    def put_table(self, table: pa.Table, rel: str) -> str:
        """Write a Parquet file to the zone; returns the sha256 of the bytes written."""
        buf = io.BytesIO(); pq.write_table(table, buf); data = buf.getvalue()
        if self.minio:
            self.client.put_object(self.s.s3_bucket, f"{SLICE}/{rel}", io.BytesIO(data), len(data),
                                   content_type="application/octet-stream")
        else:
            p = self.s.local_lake_root / SLICE / rel
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def get_table(self, rel: str) -> pa.Table:
        if self.minio:
            resp = self.client.get_object(self.s.s3_bucket, f"{SLICE}/{rel}")
            try:
                return pq.read_table(io.BytesIO(resp.read()))
            finally:
                resp.close(); resp.release_conn()
        return pq.read_table(self.s.local_lake_root / SLICE / rel)


# ============================== BRONZE (read only, never modified) ===========================
def bronze_syndrome_row(lake: Lake) -> dict:
    with lake.open_zip(SYN_KEY) as z, z.open(SYN_MEMBER) as f:
        # stream the CSV and stop at the row we need instead of loading all of it
        reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
        for i, row in enumerate(reader):
            if i == SYN_ROW:
                return row
    raise IndexError(f"{SYN_MEMBER} has no data row {SYN_ROW}")


def bronze_google_shot(lake: Lake) -> dict:
    """Read only the bytes of shot G_SHOT from each companion member (no full extraction)."""
    with lake.open_zip(GOOGLE_KEY) as z:
        props = yaml.safe_load(z.read(f"{G_EXP}/properties.yml"))

        def b8(name: str, bits: int) -> bytes:
            n = b8_record_bytes(bits)
            with z.open(f"{G_EXP}/{name}") as f:
                f.read(G_SHOT * n)                    # skip earlier shots (0 bytes for shot 0)
                return f.read(n)

        def bit01(name: str) -> int:
            with z.open(f"{G_EXP}/{name}") as f:
                for i, line in enumerate(f):
                    if i == G_SHOT:
                        return parse_01_records(line)[0]
            raise IndexError(name)

        return {
            "props": props,
            "measurement": b8("measurements.b8", props["circuit_measurements"]),
            "sweep": b8("sweep.b8", props["circuit_sweep_bits"]),
            "detector": b8("detection_events.b8", props["circuit_detectors"]),
            "actual": bit01("obs_flips_actual.01"),
            "pred": {d: bit01(f"obs_flips_predicted_by_{d}.01") for d in DECODERS},
        }


# ============================== SILVER (typed Parquet, contract schemas) =====================
SYN_SCHEMA = pa.schema([
    ("source_record_id", pa.string()), ("experiment_id", pa.string()),
    ("physical_fault_rate", pa.float64()), ("syndrome_bits", pa.binary()),
    ("round_count", pa.int32()), ("check_count", pa.int32()),
    ("logical_error_label", pa.bool_()), ("quantity", pa.int64()),
])
SYN_SILVER = "silver/qec_syndromes/syndrome_observation.parquet"
G_EXP_SILVER, G_SHOT_SILVER = "silver/google_qec/experiment.parquet", "silver/google_qec/shot.parquet"


def silver_syndrome(raw: dict, lake: Lake) -> None:
    grid = ast.literal_eval(raw["syndromes"])
    assert len(grid) == 4 and all(len(r) == 4 for r in grid), "not a 4x4 syndrome"
    flat = bytes(bit for rnd in grid for bit in rnd)           # round first, then check
    assert set(flat) <= {0, 1}
    qty = int(raw["quantity"]); assert qty > 0
    pfr = float(SYN_MEMBER.split("_pfr-")[1].split("_")[0])
    row = {
        "source_record_id": f"qec_syndromes:{SYN_MEMBER}:row:{SYN_ROW}",
        "experiment_id": SYN_MEMBER.removesuffix(".csv"),
        "physical_fault_rate": pfr, "syndrome_bits": flat, "round_count": 4, "check_count": 4,
        "logical_error_label": raw["labels"] == "1", "quantity": qty,
    }
    lake.put_table(pa.Table.from_pylist([row], schema=SYN_SCHEMA), SYN_SILVER)


def silver_google(raw: dict, lake: Lake) -> None:
    p = raw["props"]
    exp = {"source_record_id": f"google_qec:{G_EXP}:properties.yml", "experiment_id": G_EXP,
           "basis": p["basis"], "distance": p["distance"], "rounds": p["rounds"], "shots": p["shots"],
           "center_row": p["center_data_qubit_row"], "center_col": p["center_data_qubit_col"],
           "measurement_count": p["circuit_measurements"], "detector_count": p["circuit_detectors"]}
    events = sum(unpack_little_endian_bits(raw["detector"], p["circuit_detectors"]))
    shot = {"source_record_id": f"google_qec:{G_EXP}:shot:{G_SHOT}", "experiment_id": G_EXP,
            "shot_index": G_SHOT, "measurement_bits": raw["measurement"], "sweep_bits": raw["sweep"],
            "detector_bits": raw["detector"], "detector_event_count": events,
            "actual_observable_flip": bool(raw["actual"]),
            **{f"{d}_prediction": bool(v) for d, v in raw["pred"].items()}}
    lake.put_table(pa.Table.from_pylist([exp]), G_EXP_SILVER)
    lake.put_table(pa.Table.from_pylist([shot]), G_SHOT_SILVER)


# ============================== TRACE (Silver row -> Bronze bytes) ===========================
def trace_rows(lake: Lake) -> list[dict]:
    syn_hash, g_hash = lake.sha256(SYN_KEY), lake.sha256(GOOGLE_KEY)
    rows = [{"source_record_id": f"qec_syndromes:{SYN_MEMBER}:row:{SYN_ROW}", "source_name": "qec_syndromes",
             "bronze_object": SYN_KEY, "archive_member": SYN_MEMBER,
             "record_locator": f"csv_row:{SYN_ROW}", "input_sha256": syn_hash}]
    members = ["measurements.b8", "sweep.b8", "detection_events.b8", "obs_flips_actual.01",
               *[f"obs_flips_predicted_by_{d}.01" for d in DECODERS]]
    for m in members:  # one logical shot -> several trace rows (one per companion file)
        rows.append({"source_record_id": f"google_qec:{G_EXP}:shot:{G_SHOT}", "source_name": "google_qec",
                     "bronze_object": GOOGLE_KEY, "archive_member": f"{G_EXP}/{m}",
                     "record_locator": f"shot:{G_SHOT}", "input_sha256": g_hash})
    return rows


# ============================== GOLD (one transaction, idempotent) ===========================
def load_gold(conn: psycopg.Connection, lake: Lake) -> None:
    syn = lake.get_table(SYN_SILVER).to_pylist()[0]           # Gold is fed from SILVER, not Bronze
    gexp = lake.get_table(G_EXP_SILVER).to_pylist()[0]
    gshot = lake.get_table(G_SHOT_SILVER).to_pylist()[0]
    with conn.transaction():                               # all-or-nothing
        conn.execute((SQL / "gold_slice.sql").read_text())
        conn.execute("INSERT INTO gold.sim_experiment VALUES (%s,%s,3) ON CONFLICT DO NOTHING",
                     (syn["experiment_id"], syn["physical_fault_rate"]))
        conn.execute("INSERT INTO gold.syndrome_observation VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                     (syn["source_record_id"], syn["experiment_id"], syn["syndrome_bits"],
                      syn["logical_error_label"], syn["quantity"]))
        conn.execute("INSERT INTO gold.google_experiment VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                     (gexp["experiment_id"], gexp["basis"], gexp["distance"], gexp["rounds"], gexp["shots"],
                      gexp["center_row"], gexp["center_col"], gexp["detector_count"]))
        conn.execute("INSERT INTO gold.google_shot VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                     (gshot["source_record_id"], gshot["experiment_id"], gshot["shot_index"],
                      gshot["detector_bits"], gshot["detector_event_count"], gshot["actual_observable_flip"]))
        for d in DECODERS:
            conn.execute("INSERT INTO gold.decoder_prediction VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                         (gshot["source_record_id"], d, gshot[f"{d}_prediction"]))
        for f in ("ml_syndrome_example.sql", "ml_google_example.sql"):
            conn.execute((SQL / f).read_text())


# ============================== ML (SQL view + thin export) ==================================
SYN_COLS = ["example_id", "experiment_id", "physical_fault_rate", "syndrome_bits", "round_count",
            "check_count", "logical_error_label", "sample_weight"]
G_COLS = ["example_id", "experiment_id", "shot_index", "distance", "rounds", "center_row", "center_col",
          "detector_count", "detector_event_count", "detector_bits",
          *[f"{d}_prediction" for d in DECODERS], "actual_observable_flip"]
ML_SYN, ML_GOOGLE = "ml/ml_syndrome_decoder_example.parquet", "ml/ml_google_decoder_example.parquet"


def export_ml(conn: psycopg.Connection, lake: Lake) -> dict[str, str]:
    cur = conn.execute(f"SELECT {', '.join(SYN_COLS)} FROM gold.v_ml_syndrome_decoder_example ORDER BY example_id")
    syn = [dict(zip(SYN_COLS, r)) for r in cur]
    for r in syn:
        r["syndrome_bits"] = bytes(r["syndrome_bits"])
        r["data_split"] = syndrome_data_split(r["physical_fault_rate"])       # supplied helper
        assert len(r["syndrome_bits"]) == 16 and set(r["syndrome_bits"]) <= {0, 1}
    cur = conn.execute(f"SELECT {', '.join(G_COLS)} FROM gold.v_ml_google_decoder_example ORDER BY example_id")
    g = [dict(zip(G_COLS, r)) for r in cur]
    for r in g:
        r["detector_bits"] = bytes(r["detector_bits"])
        r["data_split"] = google_data_split(r["shot_index"])                  # supplied helper
        bits = unpack_little_endian_bits(r["detector_bits"], r["detector_count"])  # checks byte length
        assert sum(bits) == r["detector_event_count"], "event count != set bits"
    for rows in (syn, g):
        assert len({r["example_id"] for r in rows}) == len(rows), "example_id not unique"
    return {ML_SYN: lake.put_table(pa.Table.from_pylist(syn), ML_SYN),
            ML_GOOGLE: lake.put_table(pa.Table.from_pylist(g), ML_GOOGLE)}


# ============================== TRACE BACK: ML row -> Gold -> Silver -> Bronze ===============
def trace_back(conn: psycopg.Connection, lake: Lake, ml_rel: str, view: str, trace: list[dict]) -> None:
    ex = lake.get_table(ml_rel).to_pylist()[0]
    sid, = conn.execute(f"SELECT source_record_id FROM gold.{view} WHERE example_id=%s", (ex["example_id"],)).fetchone()
    print(f"  ML example_id {ex['example_id'][:12]}...  (split={ex['data_split']})")
    print(f"  -> Gold/Silver source_record_id: {sid}")
    for t in (t for t in trace if t["source_record_id"] == sid):
        print(f"  -> Bronze: {t['bronze_object']} :: {t['archive_member']} @ {t['record_locator']}  sha256={t['input_sha256'][:12]}...")


def main() -> None:
    s = Settings.from_environment()
    lake = Lake(s)
    print(f"backend={s.lake_backend}" + (f"  endpoint={s.s3_endpoint}  bucket={s.s3_bucket}" if lake.minio else ""))
    print("BRONZE -> SILVER")
    syn_raw, g_raw = bronze_syndrome_row(lake), bronze_google_shot(lake)
    print("  syndrome CSV row:", syn_raw)
    lake.bronze_report()
    silver_syndrome(syn_raw, lake); silver_google(g_raw, lake)
    trace = trace_rows(lake)
    lake.put_table(pa.Table.from_pylist(trace), "silver/_slice_source_trace.parquet")
    print(f"  wrote 3 Silver files + {len(trace)} trace rows to {lake.uri('silver/')}")
    with psycopg.connect(s.postgres_dsn) as conn:
        print("SILVER -> GOLD"); load_gold(conn, lake); print("  loaded in one transaction")
        print("GOLD -> ML"); hashes = export_ml(conn, lake)
        for rel, h in hashes.items():
            print(f"  {lake.uri(rel)}: 1 row(s), sha256={h[:12]}...")
        print("TRACE BACK (syndrome)"); trace_back(conn, lake, ML_SYN, "v_ml_syndrome_decoder_example", trace)
        print("TRACE BACK (google)");   trace_back(conn, lake, ML_GOOGLE, "v_ml_google_decoder_example", trace)


if __name__ == "__main__":
    main()
