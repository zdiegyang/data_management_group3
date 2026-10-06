-- ML example_id -> Gold record(s). Brief: "every ML example_id can be resolved to
-- the relevant Gold record or records through a relation or view you design".
-- One row = one Gold record that an ML example was built from. A syndrome example
-- has one (its syndrome_observation); a Google example has five (its google_shot
-- and the four decoder_prediction rows aligned to that shot). Gold source_record_id
-- then resolves through results/part1/source_trace.parquet to the Bronze members.
-- Requires ml_syndrome_example.sql and ml_google_example.sql to be created first.
CREATE OR REPLACE VIEW gold.v_ml_example_gold_record AS
SELECT example_id,
       'ml_syndrome_decoder_example'      AS ml_table,
       'syndrome_observation'             AS gold_table,
       source_record_id,
       NULL::text                         AS decoder_name
FROM gold.v_ml_syndrome_decoder_example
UNION ALL
SELECT example_id,
       'ml_google_decoder_example',
       'google_shot',
       source_record_id,
       NULL::text
FROM gold.v_ml_google_decoder_example
UNION ALL
SELECT g.example_id,
       'ml_google_decoder_example',
       'decoder_prediction',
       p.shot_source_record_id,
       p.decoder_name
FROM gold.v_ml_google_decoder_example g
JOIN gold.decoder_prediction p ON p.shot_source_record_id = g.source_record_id;
