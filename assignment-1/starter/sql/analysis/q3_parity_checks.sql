-- Q3 (part 1): How does the repetition-code circuit map data qubits to
-- parity-check ancillas and syndrome bits? One row per parity check of
-- qec_sm_n5. Joins gold.circuit, gold.stabilizer_check and gold.check_data_qubit.
SELECT c.variant,
       sc.ancilla_qubit,
       string_agg(q.data_qubit, ', ' ORDER BY q.position)                     AS data_qubits,
       sc.syndrome_bit
FROM gold.circuit c
JOIN gold.stabilizer_check sc ON sc.circuit_id = c.circuit_id
JOIN gold.check_data_qubit q ON q.check_source_record_id = sc.source_record_id
WHERE c.benchmark_name = 'qec_sm_n5'
GROUP BY c.variant, sc.ancilla_qubit, sc.syndrome_bit
ORDER BY c.variant, sc.syndrome_bit;
