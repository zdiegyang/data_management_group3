-- Evidence for the rejected relationship. Spec: "Teams must preserve the
-- evidence that no row-level QASMBench-to-experiment join exists."
-- Each row tests one candidate join key between QASMBench circuits and the
-- Google or simulated experiments and counts the matching circuit rows.
-- Every count is 0, so no key links a circuit to an experiment and Gold has no
-- foreign key between them.
SELECT 'circuit_id = google experiment_id'                    AS candidate_key,
       (SELECT count(*) FROM gold.circuit c
        JOIN gold.google_experiment e ON e.experiment_id = c.circuit_id)       AS matching_circuits
UNION ALL
SELECT 'benchmark_name = google experiment_id',
       (SELECT count(*) FROM gold.circuit c
        JOIN gold.google_experiment e ON e.experiment_id = c.benchmark_name)
UNION ALL
SELECT 'circuit_id = simulated experiment_id',
       (SELECT count(*) FROM gold.circuit c
        JOIN gold.sim_experiment e ON e.experiment_id = c.circuit_id)
UNION ALL
SELECT 'benchmark_name = simulated experiment_id',
       (SELECT count(*) FROM gold.circuit c
        JOIN gold.sim_experiment e ON e.experiment_id = c.benchmark_name)
UNION ALL
SELECT 'source_record_id shared with a google experiment',
       (SELECT count(*) FROM gold.circuit c
        JOIN gold.google_experiment e ON e.source_record_id = c.source_record_id)
UNION ALL
-- A distance-d surface-code patch uses 2*d*d - 1 physical qubits (17 for d=3,
-- 49 for d=5); QASMBench circuits declare their qubit count.
SELECT 'qubit_count = qubits of a google patch (2*d*d - 1)',
       (SELECT count(*) FROM gold.circuit c
        JOIN gold.google_experiment e ON c.qubit_count = 2 * e.distance * e.distance - 1)
ORDER BY candidate_key;
