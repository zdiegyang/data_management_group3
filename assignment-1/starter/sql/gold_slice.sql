-- Minimal Gold slice: just enough tables to carry ONE syndrome row and ONE Google shot.
-- Idempotent: safe to run repeatedly. Grow these into your real Gold schema later.
CREATE SCHEMA IF NOT EXISTS gold;

-- One row = one simulated fault-rate experiment (one CSV file).
CREATE TABLE IF NOT EXISTS gold.sim_experiment (
    experiment_id        text PRIMARY KEY,
    physical_fault_rate  double precision NOT NULL CHECK (physical_fault_rate > 0),
    distance             int  NOT NULL CHECK (distance > 0)
);

-- One row = one distinct (syndrome, label) combination within one experiment.
CREATE TABLE IF NOT EXISTS gold.syndrome_observation (
    source_record_id     text PRIMARY KEY,
    experiment_id        text NOT NULL REFERENCES gold.sim_experiment,
    syndrome_bits        bytea   NOT NULL CHECK (octet_length(syndrome_bits) = 16),
    logical_error_label  boolean NOT NULL,
    quantity             bigint  NOT NULL CHECK (quantity > 0),
    UNIQUE (experiment_id, syndrome_bits, logical_error_label)
);

-- One row = one Google hardware experiment directory.
CREATE TABLE IF NOT EXISTS gold.google_experiment (
    experiment_id   text PRIMARY KEY,
    basis           text NOT NULL,
    distance        int  NOT NULL CHECK (distance IN (3, 5)),
    rounds          int  NOT NULL,
    shots           bigint NOT NULL CHECK (shots > 0),
    center_row      int  NOT NULL,
    center_col      int  NOT NULL,
    detector_count  int  NOT NULL CHECK (detector_count > 0)
);

-- One row = one aligned hardware shot (detector events + actual outcome).
CREATE TABLE IF NOT EXISTS gold.google_shot (
    source_record_id       text PRIMARY KEY,
    experiment_id          text NOT NULL REFERENCES gold.google_experiment,
    shot_index             bigint NOT NULL CHECK (shot_index >= 0),
    detector_bits          bytea  NOT NULL,
    detector_event_count   int    NOT NULL CHECK (detector_event_count >= 0),
    actual_observable_flip boolean NOT NULL,
    UNIQUE (experiment_id, shot_index)
);
CREATE INDEX IF NOT EXISTS ix_google_shot_exp_shot ON gold.google_shot (experiment_id, shot_index);

-- One row = one decoder's prediction for one shot (long form: adding a decoder adds rows, not columns).
CREATE TABLE IF NOT EXISTS gold.decoder_prediction (
    shot_source_record_id  text NOT NULL REFERENCES gold.google_shot,
    decoder_name           text NOT NULL,
    predicted_flip         boolean NOT NULL,
    PRIMARY KEY (shot_source_record_id, decoder_name)
);
