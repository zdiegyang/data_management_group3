-- Q1 detail: weighted syndrome frequency and logical-error rate per fault rate
-- and per number of fired syndrome bits (hamming weight of the 16-bit pattern).
-- Joins gold.sim_experiment, gold.syndrome_observation and gold.syndrome_pattern.
SELECT e.physical_fault_rate,
       p.hamming_weight                                                       AS fired_bits,
       sum(o.quantity)                                                        AS weighted_shots,
       round(sum(o.quantity)::numeric
             / sum(sum(o.quantity)) OVER (PARTITION BY e.experiment_id), 6)   AS weighted_frequency,
       round(coalesce(sum(o.quantity) FILTER (WHERE o.logical_error_label), 0)::numeric
             / sum(o.quantity), 6)                                            AS weighted_logical_error_rate
FROM gold.sim_experiment e
JOIN gold.syndrome_observation o ON o.experiment_id = e.experiment_id
JOIN gold.syndrome_pattern p ON p.syndrome_bits = o.syndrome_bits
GROUP BY e.experiment_id, e.physical_fault_rate, p.hamming_weight
ORDER BY e.physical_fault_rate, p.hamming_weight;
