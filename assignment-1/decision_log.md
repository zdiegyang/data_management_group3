# Sample Discoveries on Datasets: 

=== Source sizes ===
source=google_qec/google-surface-code-curated.zip       14638673 bytes
source=qasmbench/qasmbench-qec.zip      144172 bytes
source=qec_syndromes/syndromes_dataset.zip      358017 bytes

=== Syndrome CSV sample ===
CSV file: d-3_pfr-0.000010_nb-10M.csv
row count: 68
columns: ['labels', 'syndromes', 'quantity']
first row: {'labels': '0', 'syndromes': '((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))', 'quantity': '9987291'}
data types: {'labels': 'str', 'syndromes': 'str', 'quantity': 'str'}

=== Google experiment discovery ===
experiments: ['surface_code_bX_d3_r25_center_3_5', 'surface_code_bX_d3_r25_center_5_3', 'surface_code_bX_d3_r25_center_5_7', 'surface_code_bX_d3_r25_center_7_5', 'surface_code_bX_d5_r25_center_5_5']
properties keys: ['type', 'basis', 'rounds', 'distance', 'data_qubits', 'measure_qubits', 'shots', 'center_data_qubit_row', 'center_data_qubit_col', 'circuit_measurements', 'circuit_sweep_bits', 'circuit_detectors', 'circuit_observables', 'circuit_qubits']
properties sample: {'type': 'surface_code_memory_experiment', 'basis': 'X', 'rounds': 25, 'distance': 3, 'data_qubits': 9, 'measure_qubits': 8, 'shots': 50000, 'center_data_qubit_row': 3, 'center_data_qubit_col': 5, 'circuit_measurements': 209}
surface_code_bX_d3_r25_center_3_5/measurements.b8 size: 1350000 bytes
first bytes: b'\x85 \xad\x905\x90_\xb8\x1d\xb8\x1d\xb8]\xf8]\xf8'
surface_code_bX_d3_r25_center_3_5/sweep.b8 size: 100000 bytes
first bytes: b'\xd0\x00\xd0\x00\xd0\x00\xd0\x00\xd0\x00\xd0\x00\xd0\x00\xd0\x00'
surface_code_bX_d3_r25_center_3_5/detection_events.b8 size: 1250000 bytes
first bytes: b'\x00\x80\x82\t\x00\xa0&\x04\x00\x00\x00\x00\x04\x00\x00 '
surface_code_bX_d3_r25_center_3_5/obs_flips_actual.01 size: 100000 bytes
line count: 50000
first lines: [b'1', b'1', b'1', b'0', b'1']

=== QASMBench discovery ===
benchmark folders: ['error_correctiond3_n5', 'qec_en_n5', 'qec_sm_n5']
qasm files: ['small/error_correctiond3_n5/error_correctiond3_n5.qasm', 'small/error_correctiond3_n5/error_correctiond3_n5_transpiled.qasm', 'small/qec_en_n5/qec_en_n5.qasm', 'small/qec_en_n5/qec_en_n5_transpiled.qasm', 'small/qec_sm_n5/qec_sm_n5.qasm', 'small/qec_sm_n5/qec_sm_n5_transpiled.qasm']
total qasm files: 6
sample file: small/error_correctiond3_n5/error_correctiond3_n5.qasm
sample first 20 lines:
// Error correction: distance-three 5-qubit code, from the paper "Benchmarking gate-based quantum computers" by K. Michielsen et al.

OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

h q[0];
h q[1];
id q[2];
h q[3];
h q[4];
cx q[1],q[2];
h q[1];
h q[2];
cx q[1],q[2];
h q[1];
h q[2];
cx q[1],q[2];
sample keyword counts: {'OPENQASM': 1, 'qreg': 1, 'creg': 1, 'cx': 49, 'measure': 5, 'if': 0, 'x': 49, 'z': 0}
QASMBench notes:
- these are OpenQASM circuits, not training rows
- focus on benchmark name, circuit variant, register sizes, and parity-check structure
- possible IDs: benchmark folder, circuit file, qasm member path, register names