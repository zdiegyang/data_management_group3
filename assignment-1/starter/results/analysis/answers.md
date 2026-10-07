# Analysis #
The query for each of the question is attached and the result of the query was downloaded as a .csv file in this folder.
The .csv files were obtained by running the queries below in the Adminer SQL command section on `localhost:8080`

## Q1 : How do weighted syndrome frequency and logical-error labels change with physical fault rate? ##
```sql
WITH weighted AS (
    SELECT
        e.experiment_id,
        e.physical_fault_rate,
        p.hamming_weight,
        o.logical_error_label,
        SUM(o.quantity) AS weighted_shots
    FROM gold.syndrome_observation AS o
    JOIN gold.sim_experiment AS e
        ON e.experiment_id = o.experiment_id
    JOIN gold.syndrome_pattern AS p
        ON p.syndrome_bits = o.syndrome_bits
    GROUP BY
        e.experiment_id,
        e.physical_fault_rate,
        p.hamming_weight,
        o.logical_error_label
),
totals AS (
    SELECT
        experiment_id,
        SUM(weighted_shots) AS total_shots
    FROM weighted
    GROUP BY experiment_id
)
SELECT
    w.physical_fault_rate,
    w.hamming_weight,
    SUM(w.weighted_shots) AS weighted_shots,
    ROUND(
        SUM(w.weighted_shots)::numeric / t.total_shots,
        6
    ) AS weighted_frequency,
    ROUND(
        SUM(w.weighted_shots)
            FILTER (WHERE w.logical_error_label)
        ::numeric / SUM(w.weighted_shots),
        6
    ) AS weighted_logical_error_rate
FROM weighted AS w
JOIN totals AS t
    ON t.experiment_id = w.experiment_id
GROUP BY
    w.experiment_id,
    w.physical_fault_rate,
    w.hamming_weight,
    t.total_shots
ORDER BY
    w.physical_fault_rate,
    w.hamming_weight;
```
We grouped the syndrome observations by physical fault rate and Hamming weight in order to show that higher weight syndromes
become more common as the simulated fault rate changes. The tables we joined are the `syndrome_observation`, `syndrome_experiment` and `syndrome_pattern`. 
The data shows an association between physical fault rate, syndrome-pattern frequency and logical-error frequency. The same syndrome 
pattern can also occur with different logical-error labels.
## Q2 : How do the supplied decoder logical-error rates compare by code distance and distance-three processor location? ##
```sql
SELECT
    e.distance,
    e.center_row,
    e.center_col,
    d.decoder_name,
    COUNT(*) AS shots,
    COUNT(*) FILTER (
        WHERE p.predicted_flip <> s.actual_observable_flip
    ) AS decoder_errors,
    ROUND(
        COUNT(*) FILTER (
            WHERE p.predicted_flip <> s.actual_observable_flip
        )::numeric / COUNT(*),
        6
    ) AS logical_error_rate,
    ROUND(
        AVG(s.actual_observable_flip::int),
        6
    ) AS actual_logical_flip_rate
FROM gold.google_experiment AS e
JOIN gold.google_shot AS s
    ON s.experiment_id = e.experiment_id
JOIN gold.decoder_prediction AS p
    ON p.shot_source_record_id = s.source_record_id
JOIN gold.decoder AS d
    ON d.decoder_name = p.decoder_name
GROUP BY
    e.distance,
    e.center_row,
    e.center_col,
    d.decoder_name
ORDER BY
    e.distance,
    e.center_row,
    e.center_col,
    logical_error_rate,
    d.decoder_name;
```
For this question we joined together the tables `google_experiment`, `google_shot`, `decoder_prediction` and `decoder_prediction`. 
The data shows that the decoders have different logic-error rates, and that their preformance varies with code distance and location. 
The distance 3 results show that the decoder performance is not identical across the 4 locations. The observations from the experiments
do not establish that processor location or code distance cause any change in performance for any of the decoders.
## Q3 : How does the repetition-code circuit map data qubits to parity-check ancillas, syndrome bits, and conditional corrections? ##
```sql
SELECT
    c.benchmark_name,
    c.variant,
    sc.check_id,
    STRING_AGG(
        cdq.data_qubit,
        ', ' ORDER BY cdq.position
    ) AS data_qubits,
    sc.ancilla_qubit,
    sc.syndrome_bit,
    cc.condition_register,
    cc.condition_value,
    cc.gate,
    cc.target_qubit
FROM gold.circuit AS c
JOIN gold.stabilizer_check AS sc
    ON sc.circuit_id = c.circuit_id
JOIN gold.check_data_qubit AS cdq
    ON cdq.check_source_record_id = sc.source_record_id
LEFT JOIN gold.conditional_correction AS cc
    ON cc.circuit_id = c.circuit_id
WHERE c.benchmark_name = 'qec_sm_n5'
GROUP BY
    c.benchmark_name,
    c.variant,
    sc.check_id,
    sc.ancilla_qubit,
    sc.syndrome_bit,
    cc.condition_register,
    cc.condition_value,
    cc.gate,
    cc.target_qubit
ORDER BY
    c.variant,
    sc.check_id,
    cc.condition_value;
```
The tables joined are `circuit`, `stabilizer_check`, `check_data_qubit` and `conditional_correction`. The qec_sm_n5 circuit 
uses three data qubits and two parity-check ancillas. One check measures the parity of q[0] and q[1] into ancilla a[0] 
and syndrome bit syn[0]. The other measures q[1] and q[2] into a[1] and syn[1]. The resulting two-bit syndrome controls 
conditional X corrections for syndrome values 1, 2, and 3. Gold represents these relationships through circuit, 
`stabilizer_check`, `check_data_qubit`, and `conditional_correction`, allowing the circuit structure to be queried without simulating it. 