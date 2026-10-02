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

CREATE TABLE gold.sim_experiment (
    experiment_id        text PRIMARY KEY,
    physical_fault_rate  double precision NOT NULL UNIQUE CHECK (physical_fault_rate > 0),
    round_count          int NOT NULL CHECK (round_count = 4),
    check_count          int NOT NULL CHECK (check_count = 4)
);
COMMENT ON TABLE gold.sim_experiment IS
    'One row = one simulated distance-three fault-rate experiment (one source CSV file).';

CREATE TABLE gold.syndrome_observation (
    source_record_id     text PRIMARY KEY,
    experiment_id         text NOT NULL REFERENCES gold.sim_experiment,
    syndrome_bits         bytea NOT NULL CHECK (octet_length(syndrome_bits) = 16),
    logical_error_label   boolean NOT NULL,
    quantity              bigint NOT NULL CHECK (quantity > 0),
    UNIQUE (experiment_id, syndrome_bits, logical_error_label)
);
COMMENT ON TABLE gold.syndrome_observation IS
    'One row = one distinct (syndrome pattern, logical-error label) combination observed '
    'within one fault-rate experiment, weighted by how many simulated shots produced it.';
CREATE INDEX ix_syndrome_observation_experiment
    ON gold.syndrome_observation (experiment_id);

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
    'record; detector_event_count is a summary column for aggregate queries. '
    'Exploding to one row per fired detector event would be ~10.45M rows for the same '
    '8.75MB of information this column already holds, with no detector query this '
    'project needs that bit_count()/get_bit() cannot answer on the packed form.';
CREATE INDEX ix_google_shot_experiment_shot
    ON gold.google_shot (experiment_id, shot_index);

CREATE TABLE gold.decoder_prediction (
    shot_source_record_id  text NOT NULL REFERENCES gold.google_shot,
    decoder_name            text NOT NULL CHECK (decoder_name IN (
        'belief_matching', 'correlated_matching', 'pymatching', 'tensor_network_contraction'
    )),
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

CREATE TABLE gold.circuit (
    circuit_id              text PRIMARY KEY,
    source_record_id         text NOT NULL,
    benchmark_name            text NOT NULL,
    variant                   text NOT NULL CHECK (variant IN ('source', 'transpiled')),
    register_declarations     text NOT NULL,
    qubit_count               int NOT NULL CHECK (qubit_count > 0),
    measurement_count         int NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count      int NOT NULL CHECK (two_qubit_gate_count >= 0),
    UNIQUE (benchmark_name, variant)
);
COMMENT ON TABLE gold.circuit IS
    'One row = one parsed circuit variant (one benchmark family, source or '
    'transpiled), described by its register sizes and operation counts. No FK to '
    'google_experiment or sim_experiment: no shared experiment identifier exists '
    '(investigated and rejected -- see decision log).';

CREATE TABLE gold.stabilizer_check (
    source_record_id  text PRIMARY KEY,
    circuit_id          text NOT NULL REFERENCES gold.circuit,
    check_id             text NOT NULL,
    ancilla_qubit        text NOT NULL,
    data_qubits          text[] NOT NULL CHECK (array_length(data_qubits, 1) >= 2),
    syndrome_bit         text NOT NULL,
    UNIQUE (circuit_id, check_id)
);
COMMENT ON TABLE gold.stabilizer_check IS
    'One row = one parity/stabilizer check defined in a circuit: which ancilla '
    'measures which data qubits into which classical syndrome bit.';
CREATE INDEX ix_stabilizer_check_circuit ON gold.stabilizer_check (circuit_id);

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
