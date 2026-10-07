-- Q1: How do weighted syndrome frequency and logical-error labels change with
-- physical fault rate? One row per fault-rate experiment. Every rate is weighted
-- by quantity (the number of simulated shots a row stands for), never by row count.
-- Joins gold.sim_experiment, gold.syndrome_observation and gold.syndrome_pattern.
WITH per_pattern AS (
    SELECT o.experiment_id,
           o.syndrome_bits,
           count(DISTINCT o.logical_error_label) AS label_count
    FROM gold.syndrome_observation o
    GROUP BY o.experiment_id, o.syndrome_bits
)
SELECT e.physical_fault_rate,
       sum(o.quantity)                                                        AS weighted_shots,
       round(sum(o.quantity) FILTER (WHERE o.logical_error_label)::numeric
             / sum(o.quantity), 6)                                            AS weighted_logical_error_rate,
       round(sum(o.quantity) FILTER (WHERE p.hamming_weight = 0)::numeric
             / sum(o.quantity), 6)                                            AS all_zero_syndrome_share,
       round(sum(o.quantity * p.hamming_weight)::numeric / sum(o.quantity), 6) AS weighted_mean_fired_bits,
       (SELECT count(*) FROM per_pattern pp WHERE pp.experiment_id = e.experiment_id)
                                                                              AS distinct_syndromes,
       (SELECT count(*) FROM per_pattern pp
        WHERE pp.experiment_id = e.experiment_id AND pp.label_count = 2)       AS syndromes_with_both_labels
FROM gold.sim_experiment e
JOIN gold.syndrome_observation o ON o.experiment_id = e.experiment_id
JOIN gold.syndrome_pattern p ON p.syndrome_bits = o.syndrome_bits
GROUP BY e.experiment_id, e.physical_fault_rate
ORDER BY e.physical_fault_rate;
