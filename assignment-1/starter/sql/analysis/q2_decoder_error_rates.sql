-- Q2: How do the supplied decoder logical-error rates compare by code distance
-- and distance-three processor location? One row per experiment and decoder.
-- A decoder makes a logical error when its predicted flip differs from the
-- actual flip (computed here, never stored). Joins four Gold tables:
-- google_experiment -> google_shot -> decoder_prediction -> decoder.
SELECT e.distance,
       e.center_row,
       e.center_col,
       d.decoder_name,
       count(*)                                                               AS shots,
       count(*) FILTER (WHERE p.predicted_flip <> s.actual_observable_flip)    AS decoder_errors,
       round(count(*) FILTER (WHERE p.predicted_flip <> s.actual_observable_flip)::numeric
             / count(*), 6)                                                   AS logical_error_rate,
       round(avg(s.actual_observable_flip::int), 6)                           AS actual_flip_rate
FROM gold.google_experiment e
JOIN gold.google_shot s ON s.experiment_id = e.experiment_id
JOIN gold.decoder_prediction p ON p.shot_source_record_id = s.source_record_id
JOIN gold.decoder d ON d.decoder_name = p.decoder_name
GROUP BY e.distance, e.center_row, e.center_col, d.decoder_name
ORDER BY e.distance, e.center_row, e.center_col, d.decoder_name;
