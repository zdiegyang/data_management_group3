CREATE OR REPLACE VIEW gold.v_ml_google_decoder_example AS
SELECT
    encode(sha256(convert_to(s.experiment_id || '|' || s.shot_index::text, 'UTF8')), 'hex') AS example_id,
    s.experiment_id,
    s.shot_index,
    e.distance, e.rounds, e.center_row, e.center_col, e.detector_count,
    s.detector_event_count,
    s.detector_bits,
    bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'belief_matching')                AS belief_matching_prediction,
    bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'correlated_matching')            AS correlated_matching_prediction,
    bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'pymatching')                     AS pymatching_prediction,
    bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'tensor_network_contraction')     AS tensor_network_contraction_prediction,
    s.actual_observable_flip,
    s.source_record_id                                                                          -- lineage only; NOT exported
FROM gold.google_shot s
JOIN gold.google_experiment e USING (experiment_id)
JOIN gold.decoder_prediction p ON p.shot_source_record_id = s.source_record_id
GROUP BY s.source_record_id, s.experiment_id, s.shot_index, e.distance, e.rounds, e.center_row,
         e.center_col, e.detector_count, s.detector_event_count, s.detector_bits, s.actual_observable_flip;

