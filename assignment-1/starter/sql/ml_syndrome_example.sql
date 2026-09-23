-- Gold -> ML view. Everything except data_split comes from Gold; data_split is added by the
-- supplied Python helper (syndrome_data_split) in the export step.
CREATE OR REPLACE VIEW gold.v_ml_syndrome_decoder_example AS
SELECT
    encode(sha256(convert_to(
        o.experiment_id || '|' || encode(o.syndrome_bits, 'hex') || '|' || o.logical_error_label::int::text,
        'UTF8')), 'hex')                      AS example_id,
    o.experiment_id,
    e.physical_fault_rate,
    o.syndrome_bits,
    4::int                                    AS round_count,
    4::int                                    AS check_count,
    o.logical_error_label,
    o.quantity                                AS sample_weight,
    o.source_record_id                        -- lineage only; NOT exported
FROM gold.syndrome_observation o
JOIN gold.sim_experiment e USING (experiment_id);
