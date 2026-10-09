"""Tests for the QASMBench checks against the files' own documentation."""

import io
import zipfile

import pyarrow as pa

from quantum_lake_student.stages import prepare_data

QELIB = "gate x a { U(pi,0,pi) a; }\n// a comment mentioning gate h\ngate h a { U(pi/2,0,pi) a; }\n"
CIRCUIT = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "gate pair a,b { CX a,b; }\n"
    "h q[0];\nsx q[1];\nsx q[0];\npair q[0],q[1];\nif(c==1) x q[0];\nmeasure q -> c;\n"
)
README = "# Application: demo\n- Qubit Count : 2\n- Dual Gate Count : 3\n- Gate Count : 6\n"


def _archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("qelib1.inc", QELIB)
        archive.writestr("small/demo/demo.qasm", CIRCUIT)
        archive.writestr("small/demo/README.md", README)
        archive.writestr("small/demo/demo.png", b"\x89PNG")
    return buffer.getvalue()


def _tables(data_qubits=("q[0]", "q[1]"), ancilla="a[0]"):
    circuit = pa.Table.from_pylist([{
        "benchmark_name": "demo", "variant": "source",
        "qubit_count": 2, "two_qubit_gate_count": 1, "operation_count": 6,
    }])
    checks = pa.Table.from_pylist([{
        "source_record_id": "qasmbench:small/demo/demo.qasm:check:0", "ancilla_qubit": ancilla,
        "data_qubits": list(data_qubits),
    }])
    return {"circuit": circuit, "stabilizer_check": checks}


def _run(tables):
    issues, checks = [], {}
    prepare_data._qasm_documentation_checks(_archive(), "run-1", tables, issues, checks)
    return issues, checks


def test_declared_gates_ignore_comments_and_include_opaque():
    assert prepare_data._declared_gates(QELIB) == {"x", "h"}
    assert prepare_data._declared_gates("opaque magic a;\ngate g a { x a; }") == {"magic", "g"}


def test_used_gates_count_executed_statements_only():
    used = prepare_data._used_gates(CIRCUIT)
    assert used == {"h": 1, "sx": 2, "pair": 1, "x": 1}  # gate body CX is a definition


def test_undeclared_gate_is_a_warning_and_the_circuit_is_kept():
    issues, checks = _run(_tables())
    undeclared = [i for i in issues if i["rule_id"] == "RULE_QASM_UNDECLARED_GATE"]
    assert [(i["observed_value"], i["severity"], i["action"]) for i in undeclared] == [
        ("sx (used 2 times)", "warning", "kept")]
    assert checks["RULE_QASM_UNDECLARED_GATE"]["checked"] == 1


def test_readme_metrics_that_differ_from_the_parsed_circuit_are_warned():
    issues, _ = _run(_tables())
    readme = sorted(i["observed_value"] for i in issues if i["rule_id"] == "RULE_QASM_README_METRICS")
    assert readme == ["Dual Gate Count: README 3, parsed 1"]  # qubit and gate counts agree


def test_parity_check_without_an_ancilla_register_is_noted():
    issues, _ = _run(_tables(data_qubits=("q[0]", "q[1]"), ancilla="q[2]"))
    noted = [i for i in issues if i["rule_id"] == "RULE_QASM_INFERRED_PARITY_CHECK"]
    assert [(i["severity"], i["action"]) for i in noted] == [("info", "kept")]
    explicit, _ = _run(_tables(ancilla="a[0]"))
    assert not [i for i in explicit if i["rule_id"] == "RULE_QASM_INFERRED_PARITY_CHECK"]


def test_non_circuit_members_are_skipped_so_all_members_reconcile():
    issues, checks = _run(_tables())
    skipped = sorted(i["observed_value"] for i in issues if i["rule_id"] == "RULE_QASM_NON_CIRCUIT_MEMBER")
    assert skipped == ["qelib1.inc", "small/demo/README.md", "small/demo/demo.png"]
    assert checks["RULE_QASM_NON_CIRCUIT_MEMBER"]["checked"] == 4  # 1 circuit + 3 skipped


def test_warnings_and_notes_are_not_counted_as_failures():
    issues, checks = _run(_tables(ancilla="q[2]"))
    outcomes = {o["rule_id"]: o for o in prepare_data._check_outcomes(checks, issues)}
    assert outcomes["RULE_QASM_UNDECLARED_GATE"]["outcome"] == "warning"
    assert outcomes["RULE_QASM_UNDECLARED_GATE"]["failed"] == 0
    assert outcomes["RULE_QASM_INFERRED_PARITY_CHECK"]["outcome"] == "observed"
    assert outcomes["RULE_QASM_NON_CIRCUIT_MEMBER"]["notes"] == 3
