# Decision log — Assignment 1 (QEC data pipeline for AI)

Plain-Markdown log of discovery findings and decisions, per `getting-started.md`
step 5. Machine-readable issues go in `results/part1/data_issues.parquet` once
the pipeline exists; this file is the human-readable "why" behind that data,
written as we go, not filled in retroactively.

Format per entry: **Date · Topic** — what we found, what we decided, why.

---

## 2026-09-17 · Bronze release shape

- **Found:** Bronze holds exactly 3 objects, one zip per source, matching
  `getting-started.md`. Member counts: `google_qec` 76 files (5 experiment
  dirs × 15 files + README), `qasmbench` 19 files (3 circuits × source +
  transpiled + README + PNGs, plus shared LICENSE/NOTICE/qelib1.inc),
  `qec_syndromes` 8 files (7 CSVs + README).
- **Decision:** Inventory step lists archive members via `zipfile` on
  in-memory bytes, never extracts to local disk. Matches "list ... without
  extracting everything."
- **Why:** Keeps Bronze bytes untouched and avoids re-downloading/extracting
  14.6 MB repeatedly during iteration.

## 2026-09-17 · `qec_syndromes` totals reconcile

- **Found:** All 7 CSVs sum their `quantity` column to exactly 10,000,000,
  matching the `nb-10M` filename token. Total across all 7 files: 75,598 rows
  representing 70,000,000 weighted shots — matches the brief exactly. Every
  parsed `syndromes` value is a 4×4 nested tuple (4 rounds × 4 checks), no
  exceptions.
- **Decision:** Treat `quantity` strictly as a `sample_weight`, never expand it
  into repeated rows. Flatten `syndrome_bits` round-first then check
  (`round0_check0, round0_check1, ..., round3_check3`) to get 16 bits, per
  `silver-tables.md` table 1.
- **Why:** Matches the Silver contract exactly; expanding rows would blow up
  storage for no benefit and contradicts "Do not materialize 70 million
  individual rows" in `data-sources.md`.

## 2026-09-17 · `syndrome_bits` is NOT a safe unique key, and the ambiguity grows with fault rate

- **Found:** `(labels, syndromes)` pairs are duplicate-free **within every one
  of the 7 files** (0 duplicates each, checked per file, not pooled). But the
  same `syndromes` pattern legitimately occurs under *both* labels, and how
  often depends heavily on the physical fault rate of that file:

  | file (pfr) | rows | patterns seen under both labels |
  | --- | --- | --- |
  | 0.000010 | 68 | 0 |
  | 0.000050 | 215 | 13 |
  | 0.000100 | 491 | 46 |
  | 0.000500 | 1,407 | 179 |
  | 0.001000 | 2,854 | 467 |
  | 0.005000 | 20,887 | 6,244 |
  | 0.010000 | 49,676 | 18,210 |

  At the lowest fault rate a syndrome pattern almost always implies one label;
  at the highest fault rate over a third of rows share a pattern with the
  opposite label somewhere else in the same file.
- **Decision:** Primary key for `syndrome_observation` in Gold will still be a
  generated `source_record_id` (SHA-256 of `{source_name, bronze_object,
  archive_member, record_locator}` via the supplied `stable_record_hash`), even
  though `(experiment_id, syndrome_bits, logical_error_label)` is also a valid
  natural key in this release (empirically unique per file). The surrogate id
  is about Bronze traceability, not just deduplication, so both can be true.
- **Why:** A key must hold even if a future release adds rows that break the
  within-file uniqueness we happen to observe today. `data-sources.md`
  explicitly asks "What is the correct key if syndrome alone is not unique?" —
  this is the answer. Separately, the fault-rate-dependent ambiguity is worth a
  line in the design report: it's a real reason `syndrome_bits` alone is a
  weaker ML feature at high fault rates (Part II Task A).

## 2026-09-17 · README documents `label`, the actual column is `labels`

- **Found:** `syndromes_dataset.zip/README.txt` describes the column as
  `label` (singular): *"label: binary label (0: no error, 1: error)"*. The
  actual CSV header, confirmed on every file, is `labels` (plural).
- **Decision:** Read the column by its real header (`labels`), not by
  hardcoding a name from the README. Note the mismatch so nobody "fixes" code
  to match the README and breaks the parser.
- **Why:** Cheap to get wrong, easy to catch once — worth writing down so the
  team doesn't rediscover it independently.

## 2026-09-17 · No missing/empty values in `qec_syndromes`

- **Found:** Checked every field of all 75,598 rows across all 7 files for
  empty/whitespace-only values. None found.
- **Decision:** No null-handling logic needed for this source in Silver
  cleaning; a future release that introduces blanks should be caught by
  keeping this check in the pipeline rather than assuming it still holds.
- **Why:** Confirms the source is as clean as it looks before we build
  cleaning logic that would otherwise silently do nothing.

## 2026-09-17 · Google `b8`/`01` companion files line up

- **Found:** For all 5 experiment directories, `measurements.b8`,
  `detection_events.b8`, and `sweep.b8` byte lengths equal
  `shots × ceil(bits_per_record / 8)` exactly (spot-checked against
  `properties.yml`'s `circuit_measurements` / `circuit_detectors` /
  `circuit_sweep_bits`). Every `obs_flips_actual.01` and all four
  `obs_flips_predicted_by_*.01` files have exactly one line per shot, values
  in `{0, 1}` only. Total shots across all 5 experiments: 250,000 (4×d3 +
  1×d5), matching the brief.
- **Decision:** Treat a "shot" as the unit that must validate across *all six*
  companion files (measurements, sweep, detectors, actual, 4 predictions)
  before it is accepted into Silver `shot.parquet`. Reject (log to
  `data_issues.parquet`) any experiment where a companion file's length
  doesn't reconcile — none found so far, but the check stays in the pipeline.
- **Why:** `silver-tables.md` table 3 requires "Validate all companion-file
  lengths before publishing the row."

## 2026-09-17 · QASMBench: only one circuit has explicit corrections

- **Found:** Of the 3 circuit families, only `qec_sm_n5` contains
  `if(...)` conditional statements (3 of them) and a custom `gate syndrome`
  definition with separate data (`q[3]`) and ancilla (`a[2]`) registers.
  `error_correctiond3_n5` and `qec_en_n5` have zero `if` statements in this
  curated excerpt.
- **Decision:** Build and test the `stabilizer_check` / `conditional_correction`
  parsers against `qec_sm_n5` first (it exercises every required column), then
  confirm the other two circuits degrade gracefully to "0 stabilizer checks
  found" / "0 corrections found" rather than erroring.
- **Why:** `data-sources.md` calls `qec_sm_n5.qasm` "the clearest parity-check
  example" — confirmed by inspection, so it's the right development target.

## 2026-09-17 · No row-level QASMBench ↔ Google/syndrome join exists

- **Found:** Zero overlap between Google experiment directory names
  (`surface_code_bX_d{3,5}_r25_center_*`) and QASMBench benchmark names
  (`error_correctiond3_n5`, `qec_en_n5`, `qec_sm_n5`). QASMBench circuits carry
  no `distance`, `shots`, or any other field that could match a Google
  experiment or a syndrome-file physical fault rate.
- **Decision:** Do not attempt any Gold foreign key between
  `qasmbench.circuit` and `google_qec.experiment` or `qec_syndromes`. Keep
  this evidence (this entry + the notebook §6 output) for the "rejected
  relationship" required in the Part I design report.
- **Why:** `ASSIGNMENT_SPEC.md` explicitly requires preserving evidence that
  this join does *not* exist, not just silently omitting it.

## 2026-09-17 · One Google shot 0 sample, `surface_code_bX_d3_r25_center_3_5`

- **Found:** shot 0's measurement and detector records decode cleanly against
  `properties.yml` (209 measurement bits / 200 detector bits, byte-aligned);
  it has 27 fired detectors and `actual_observable_flip = 1`. Across all
  50,000 shots in this experiment, `detector_event_count` ranges 2–76 (of 200
  possible).
- **Decision:** Use this shot as the running example when building the
  `shot.parquet` reader/tests, since it's a real, verified, non-trivial case
  (nonzero detector events) rather than an all-zero edge case.
- **Why:** A concrete, checked example makes it easy to unit-test the Silver
  Google-shot transform against a known-correct value.

---

## Open questions (carried into next session)

1. Should `experiment_id` for `qec_syndromes` be a readable string like
   `qec_syndromes:d3:pfr=0.000010`, or should it also go through
   `stable_record_hash` for consistency with `source_record_id`? Leaning
   readable-string since it doesn't need to trace to one physical bronze
   record, only to a distinct fault-rate group.
2. Detector-storage strategy for Gold (`ASSIGNMENT_SPEC.md` §5 says we choose
   and measure one): one row per fired detector vs. a packed/summary column?
   Need a size estimate before deciding — 250,000 shots × up to 600 bits is
   small enough to consider either.
3. Exact `example_id` hash inputs for `ml_syndrome_decoder_example` — spec
   says "repeatable hash of experiment, syndrome, and label"
   (`required-ml-tables.md`); confirm this differs from `source_record_id` on
   purpose (different columns hashed) before implementing the export query.

## Six required Silver tables — one sentence each

(Per `getting-started.md` step 6: "write one sentence describing what a row
represents in every planned Gold relation." Restating the Silver row meanings
here first, since Gold is designed to hold the same grain unless we find a
reason to change it.)

1. **`qec_syndromes/syndrome_observation.parquet`** — one row is one distinct
   `(syndrome pattern, logical-error label)` combination observed within one
   physical-fault-rate simulation run, carrying how many of the 10M shots in
   that run produced exactly that combination.
2. **`google_qec/experiment.parquet`** — one row is one hardware experiment
   directory (one fixed code distance, basis, round count, and processor
   location), i.e. the container for 50,000 shots.
3. **`google_qec/shot.parquet`** — one row is one single execution of an
   experiment's circuit on hardware, with its raw measurements, derived
   detector events, actual logical outcome, and all four decoders'
   predictions aligned together.
4. **`qasmbench/circuit.parquet`** — one row is one parsed circuit variant
   (one benchmark family, either its original or transpiled form), described
   by its register sizes and operation counts.
5. **`qasmbench/stabilizer_check.parquet`** — one row is one parity/stabilizer
   check defined in a circuit: which ancilla measures which data qubits into
   which classical syndrome bit.
6. **`qasmbench/conditional_correction.parquet`** — one row is one
   syndrome-controlled recovery operation in a circuit: which classical
   condition value triggers which gate on which qubit.
