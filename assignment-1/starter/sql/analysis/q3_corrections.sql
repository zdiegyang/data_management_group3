-- Q3 (part 2): Which parity checks fire for each syndrome value, and which
-- conditional correction does that value trigger? One row per correction of
-- qec_sm_n5.
-- A correction is conditioned on the whole syndrome register (e.g. syn == 3).
-- Bit k of that value is the syndrome bit syn[k], i.e. the result of the check
-- that writes syn[k]. A check "fires" for a value when its bit is 1, so the
-- checks are linked to a correction through the register name and bit index,
-- not paired with every correction of the circuit.
-- Joins gold.circuit, gold.conditional_correction, gold.stabilizer_check and
-- gold.check_data_qubit.
WITH checks AS (
    SELECT sc.circuit_id,
           sc.ancilla_qubit,
           split_part(sc.syndrome_bit, '[', 1)                                AS register,
           substring(sc.syndrome_bit FROM '\[(\d+)\]')::int                   AS bit_index,
           sc.syndrome_bit,
           string_agg(q.data_qubit, ', ' ORDER BY q.position)                 AS data_qubits
    FROM gold.stabilizer_check sc
    JOIN gold.check_data_qubit q ON q.check_source_record_id = sc.source_record_id
    GROUP BY sc.circuit_id, sc.ancilla_qubit, sc.syndrome_bit
)
SELECT c.variant,
       cc.condition_register,
       cc.condition_value,
       string_agg(ch.ancilla_qubit || ' (' || ch.data_qubits || ') -> ' || ch.syndrome_bit, '; '
                  ORDER BY ch.bit_index)
           FILTER (WHERE (cc.condition_value >> ch.bit_index) & 1 = 1)       AS fired_checks,
       string_agg(ch.ancilla_qubit, ', ' ORDER BY ch.bit_index)
           FILTER (WHERE (cc.condition_value >> ch.bit_index) & 1 = 0)       AS silent_checks,
       cc.gate,
       cc.target_qubit
FROM gold.circuit c
JOIN gold.conditional_correction cc ON cc.circuit_id = c.circuit_id
JOIN checks ch ON ch.circuit_id = cc.circuit_id AND ch.register = cc.condition_register
WHERE c.benchmark_name = 'qec_sm_n5'
GROUP BY c.variant, cc.condition_register, cc.condition_value, cc.gate, cc.target_qubit
ORDER BY c.variant, cc.condition_value;
