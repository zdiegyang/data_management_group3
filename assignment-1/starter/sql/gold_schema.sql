-- Gold: student-designed relational QEC model.
-- Idempotent (drop + recreate every run, see note below); safe on every pipeline invocation.
-- One-sentence row meaning is documented on every table via COMMENT ON TABLE.

-- Gold is fully derived from Silver, so each run drops and recreates it from
-- scratch rather than ALTERing an existing (possibly stale) schema. This whole
-- file runs inside load_postgres.py's one transaction together with the data
-- load, so if anything later fails, PostgreSQL's transactional DDL rolls this
-- DROP/CREATE back too -- a failed run leaves the previous Gold schema intact.
DROP SCHEMA IF EXISTS gold CASCADE;
CREATE SCHEMA gold;

-- =============================================================================
-- qec_syndromes: one simulated fault-rate experiment and its aggregated rows.
-- =============================================================================

-- 75,598 syndrome_observation rows collapse to 31,941 distinct
-- patterns, and 14,187 of those (44%) recur across more than one fault-rate
-- experiment -- the same pattern is not unique to one experiment, so it does
-- not belong flattened into the observation row. round_count/check_count
-- describe the pattern's shape, not the observation, so they moved here too.
CREATE TABLE gold.syndrome_pattern (
    syndrome_bits   bytea PRIMARY KEY CHECK (octet_length(syndrome_bits) = 16),
    round_count     int NOT NULL CHECK (round_count = 4),
    check_count     int NOT NULL CHECK (check_count = 4),
    hamming_weight  int NOT NULL CHECK (hamming_weight >= 0)  -- how many of the 16 bits are set
);
COMMENT ON TABLE gold.syndrome_pattern IS
    'One row = one distinct 16-bit syndrome pattern, deduplicated across every '
    'experiment that observed it.';

-- round_count -> rounds, matching google_experiment.rounds for
-- the same concept. distance, parsed from the experiment_id (the
-- "d-3" segment of e.g. "d-3_pfr-0.000500_nb-10M"), mirroring
-- google_experiment.distance. All 7 real experiment_id values parse to
-- distance = 3. check_count is kept: there is no equivalent column on the
-- Google side to align it with.
CREATE TABLE gold.sim_experiment (
    experiment_id        text PRIMARY KEY,
    physical_fault_rate  double precision NOT NULL UNIQUE CHECK (physical_fault_rate > 0),
    distance             int NOT NULL CHECK (distance > 0),
    rounds               int NOT NULL CHECK (rounds = 4),
    check_count           int NOT NULL CHECK (check_count = 4)
);
COMMENT ON TABLE gold.sim_experiment IS
    'One row = one simulated fault-rate experiment (one source CSV file).';

-- This table now only records
-- the (experiment, pattern, label) combination and its weight.
CREATE TABLE gold.syndrome_observation (
    source_record_id     text PRIMARY KEY,
    experiment_id          text NOT NULL REFERENCES gold.sim_experiment,
    syndrome_bits            bytea NOT NULL REFERENCES gold.syndrome_pattern,
    logical_error_label       boolean NOT NULL,
    quantity                   bigint NOT NULL CHECK (quantity > 0),
    UNIQUE (experiment_id, syndrome_bits, logical_error_label)
);
COMMENT ON TABLE gold.syndrome_observation IS
    'One row = one distinct (syndrome pattern, logical-error label) combination observed '
    'within one fault-rate experiment, weighted by how many simulated shots produced it. '
    'The pattern itself lives in syndrome_pattern.';
CREATE INDEX ix_syndrome_observation_experiment
    ON gold.syndrome_observation (experiment_id);
CREATE INDEX ix_syndrome_observation_pattern
    ON gold.syndrome_observation (syndrome_bits);

-- =============================================================================
-- google_qec: hardware experiments, shots, and the four supplied decoders.
-- =============================================================================

CREATE TABLE gold.google_experiment (
    experiment_id      text PRIMARY KEY,
    source_record_id   text NOT NULL,
    basis              text NOT NULL CHECK (basis IN ('X', 'Z')),
    distance           int NOT NULL CHECK (distance IN (3, 5)),
    rounds             int NOT NULL CHECK (rounds > 0),
    shots              bigint NOT NULL CHECK (shots > 0),
    center_row         int NOT NULL,
    center_col         int NOT NULL,
    measurement_count  int NOT NULL CHECK (measurement_count > 0),
    detector_count     int NOT NULL CHECK (detector_count > 0),
    UNIQUE (distance, center_row, center_col)
);
COMMENT ON TABLE gold.google_experiment IS
    'One row = one hardware experiment directory: one fixed code distance, basis, '
    'round count, and processor location, containing many aligned shots.';

-- Detector-storage decision (see design report): packed per shot, not exploded
-- per detector bit or per fired event. See sql/gold_schema.sql comment above
-- CREATE TABLE gold.google_shot for the measured comparison.
CREATE TABLE gold.google_shot (
    source_record_id        text PRIMARY KEY,
    experiment_id            text NOT NULL REFERENCES gold.google_experiment,
    shot_index               bigint NOT NULL CHECK (shot_index >= 0),
    measurement_bits          bytea NOT NULL,
    sweep_bits                bytea NOT NULL,
    detector_bits             bytea NOT NULL,
    detector_event_count      int NOT NULL CHECK (detector_event_count >= 0),
    actual_observable_flip    boolean NOT NULL,
    UNIQUE (experiment_id, shot_index)
);
COMMENT ON TABLE gold.google_shot IS
    'One row = one aligned hardware shot: packed measurement/detector bytes (Stim b8 '
    'format) plus the actual logical outcome. detector_bits stores the full packed '
    'record; detector_event_count is a summary column for aggregate queries. ';
CREATE INDEX ix_google_shot_experiment_shot
    ON gold.google_shot (experiment_id, shot_index);

-- decoders are a lookup table instead of a free-text
-- CHECK list repeated inline -- adding a fifth decoder is one INSERT, not a
-- schema change, and decoder_prediction gets a real foreign key.
CREATE TABLE gold.decoder (
    decoder_name  text PRIMARY KEY
);
COMMENT ON TABLE gold.decoder IS
    'One row = one supplied decoder. A lookup table so decoder_prediction has '
    'a real foreign key instead of an inline CHECK list.';

CREATE TABLE gold.decoder_prediction (
    shot_source_record_id  text NOT NULL REFERENCES gold.google_shot,
    decoder_name            text NOT NULL REFERENCES gold.decoder,
    predicted_flip           boolean NOT NULL,
    PRIMARY KEY (shot_source_record_id, decoder_name)
);
COMMENT ON TABLE gold.decoder_prediction IS
    'One row = one supplied decoder''s prediction for one shot (long form: adding a '
    'decoder adds rows here, not a new column on google_shot).';
CREATE INDEX ix_decoder_prediction_decoder
    ON gold.decoder_prediction (decoder_name);

-- A convenience view so "detector summaries" are queryable without repeating this
-- aggregation in every analysis query (checklist: detector summaries queryable).
CREATE OR REPLACE VIEW gold.v_experiment_detector_summary AS
SELECT experiment_id,
       count(*)                          AS shot_count,
       avg(detector_event_count)         AS avg_detector_events,
       min(detector_event_count)         AS min_detector_events,
       max(detector_event_count)         AS max_detector_events,
       stddev_pop(detector_event_count)  AS stddev_detector_events
FROM gold.google_shot
GROUP BY experiment_id;

-- =============================================================================
-- qasmbench: circuits and their structure. No row-level join to the other two
-- sources exists (rejected relationship, see decision log) -- no foreign key out.
-- =============================================================================

-- NEW (F10): splits "which algorithm" from "which compiled variant". Verified
-- against the real data: 6 circuit rows collapse to 3 distinct benchmarks,
-- each compiled as both a source and a transpiled variant.
CREATE TABLE gold.benchmark (
    benchmark_name  text PRIMARY KEY
);
COMMENT ON TABLE gold.benchmark IS
    'One row = one QASMBench algorithm family, independent of how it was compiled.';

CREATE TABLE gold.circuit (
    circuit_id              text PRIMARY KEY,
    source_record_id         text NOT NULL,
    benchmark_name            text NOT NULL REFERENCES gold.benchmark,
    variant                   text NOT NULL CHECK (variant IN ('source', 'transpiled')),
    register_declarations     text NOT NULL,
    qubit_count               int NOT NULL CHECK (qubit_count > 0),
    measurement_count         int NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count      int NOT NULL CHECK (two_qubit_gate_count >= 0),
    UNIQUE (benchmark_name, variant)
);
COMMENT ON TABLE gold.circuit IS
    'One row = one parsed circuit variant (one benchmark, compiled source or '
    'transpiled), described by its register sizes and operation counts. No FK to '
    'google_experiment or sim_experiment: no shared experiment identifier exists '
    '(investigated and rejected -- see decision log).';

CREATE TABLE gold.stabilizer_check (
    source_record_id  text PRIMARY KEY,
    circuit_id          text NOT NULL REFERENCES gold.circuit,
    check_id             text NOT NULL,
    ancilla_qubit        text NOT NULL,
    syndrome_bit         text NOT NULL,
    UNIQUE (circuit_id, check_id)
);
COMMENT ON TABLE gold.stabilizer_check IS
    'One row = one parity/stabilizer check defined in a circuit: which ancilla '
    'measures into which classical syndrome bit. Participating data qubits are '
    'in check_data_qubit, not a packed array here.';
CREATE INDEX ix_stabilizer_check_circuit ON gold.stabilizer_check (circuit_id);

-- NEW (F10): replaces the Silver data_qubits text[] (always length 2 or 4 in
-- the real data) with one row per participating data qubit -- queryable with
-- ordinary joins and WHERE clauses instead of array functions. position
-- preserves the original Silver list order.
CREATE TABLE gold.check_data_qubit (
    check_source_record_id  text NOT NULL REFERENCES gold.stabilizer_check,
    position                  int NOT NULL CHECK (position >= 0),
    data_qubit                 text NOT NULL,
    PRIMARY KEY (check_source_record_id, position)
);
COMMENT ON TABLE gold.check_data_qubit IS
    'One row = one data qubit participating in one stabilizer check, in its '
    'original Silver list order.';
CREATE INDEX ix_check_data_qubit_qubit ON gold.check_data_qubit (data_qubit);

CREATE TABLE gold.conditional_correction (
    source_record_id     text PRIMARY KEY,
    circuit_id             text NOT NULL REFERENCES gold.circuit,
    condition_register      text NOT NULL,
    condition_value          bigint NOT NULL,
    gate                     text NOT NULL,
    target_qubit             text NOT NULL
);
COMMENT ON TABLE gold.conditional_correction IS
    'One row = one syndrome-controlled recovery operation in a circuit: which '
    'classical condition value triggers which gate on which qubit.';
CREATE INDEX ix_conditional_correction_circuit
    ON gold.conditional_correction (circuit_id);
