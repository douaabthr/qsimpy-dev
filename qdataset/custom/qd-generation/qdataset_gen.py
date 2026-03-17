import os
import re
import csv
import json
import zipfile
import shutil
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

# ---------------------------------------------------------
# 1. PATH & CONFIG
# ---------------------------------------------------------
ZIP_FILE_PATH = "./qdataset/custom/MQTBench_2-130q.zip"
EXTRACT_FOLDER = os.path.join(os.path.dirname(ZIP_FILE_PATH), "temp_qd_generation")

CALIBRATION_FILE = "./qsimpy/resources/backend_calibration_data.json"

TASKS_CSV = "./qdataset/custom/qd-generation/all_tasks.csv"
DETAILS_CSV = "./qdataset/custom/qd-generation/tasks_backend_details.csv"

PATTERN = r"^(.*?)_indep"

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------
with open(CALIBRATION_FILE, "r") as f:
    calibration_data = json.load(f)

TARGET_BACKENDS = list(calibration_data.keys())



provider = FakeProviderForBackendV2()

def get_backend_instance(name):
    try:
        return provider.get_backend(f"fake_{name}")
    except:
        return None

# ---------------------------------------------------------
# 3. FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_detailed_features(qasm_path, calibration_data):
    try:
        init_circuit = QuantumCircuit.from_qasm_file(qasm_path)
    except Exception:
        return None

    features_data = []

    # ✅ ORIGINAL CIRCUIT
    features_data.append({
        "backend": "ideal",
        "width": init_circuit.num_qubits,
        "depth": init_circuit.depth(),
        "gates_errors": dict(init_circuit.count_ops()),
        "critical_path": None,
        "readout_errors": None
    })

    for backend_name in TARGET_BACKENDS:
        if backend_name not in calibration_data:
            continue

        data = calibration_data[backend_name]
        backend_inst = get_backend_instance(backend_name)

        if not backend_inst or backend_inst.num_qubits < init_circuit.num_qubits:
            continue

        transpiled = transpile(init_circuit, backend_inst, optimization_level=3)
        dag = circuit_to_dag(transpiled)

        # A. Gate details (for details file)
        gate_details = {}
        for node in dag.op_nodes():
            if node.name == "barrier":
                continue
            q_indices = [transpiled.find_bit(q).index for q in node.qargs]
            q_key = ",".join(map(str, q_indices))

            error = data.get("gate_metadata", {}).get(node.name, {}).get(q_key, {}).get("error", 0.0)
            gate_id = f"{node.name}_{q_key.replace(',', '_')}"
            gate_details[gate_id] = error

        # B. Clean critical path
        crit_path_dict = {}

        for node in dag.longest_path():
            if isinstance(node, DAGOpNode):
                 # 🚫 Skip barriers
                if node.name == "barrier":
                    continue
                q_indices = [transpiled.find_bit(q).index for q in node.qargs]
                q_str = "_".join(map(str, q_indices))

                duration = data.get("gate_metadata", {}).get(node.name, {}).get(
                    ",".join(map(str, q_indices)), {}
                ).get("duration_ns", 0.0)

                gate_id = f"{node.name}_{q_str}"
                crit_path_dict[gate_id] = duration
        # C. Readout errors
        measured_qubits = {
            transpiled.find_bit(node.qargs[0]).index
            for node in dag.op_nodes() if node.name == 'measure'
        }

        readout_info = {
            str(q): data.get("qubit_properties", {}).get(str(q), {}).get("readout_error", 0.0)
            for q in measured_qubits
        }

        features_data.append({
            "backend": backend_name,
            "width": transpiled.num_qubits,
            "depth": transpiled.depth(),
            "gates_errors": gate_details,
            "critical_path":crit_path_dict,
            "readout_errors": readout_info
        })

    return features_data

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    task_id = 0

    try:
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)

        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)

        os.makedirs(os.path.dirname(TASKS_CSV), exist_ok=True)

        # ---- HEADERS ----
        task_header = [
            "task_id", "algorithm",
            "original_width", "original_depth", "original_gates"
        ]

        for b in TARGET_BACKENDS:
            task_header += [f"{b}_depth", f"{b}_width", f"{b}_gates"]

        details_header = [
            "task_id", "backend",
            "width", "depth",
            "gates_errors",
            "critical_path",
            "readout_errors"
        ]

        qasm_paths = []
        for root, _, files in os.walk(EXTRACT_FOLDER):
            for f in files:
                if f.endswith(".qasm"):
                    qasm_paths.append(os.path.join(root, f))

        with open(TASKS_CSV, "w", newline="") as task_file, \
             open(DETAILS_CSV, "w", newline="") as details_file:

            task_writer = csv.writer(task_file)
            details_writer = csv.writer(details_file)

            task_writer.writerow(task_header)
            details_writer.writerow(details_header)

            for path in sorted(qasm_paths):

                fname = os.path.basename(path)
                match = re.search(PATTERN, fname)
                algo_name = match.group(1) if match else fname

                results = extract_detailed_features(path, calibration_data)
                if not results:
                    continue

                task_id += 1  # ✅ FIXED POSITION

                print(f"[Task {task_id}] Processing: {algo_name}")

                orig = results[0]

                # ---- TASK ROW ----
                task_row = [
                    task_id,
                    algo_name,
                    orig["width"],
                    orig["depth"],
                    str(orig["gates_errors"])  # ✅ clean format
                ]

                res_map = {r["backend"]: r for r in results[1:]}

                for b in TARGET_BACKENDS:
                    if b in res_map:
                        d = res_map[b]

                        # ✅ aggregate gate counts
                        gate_counts = {}
                        for gate_id in d["gates_errors"]:
                            gate_name = gate_id.split("_")[0]
                            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

                        task_row += [
                            d["depth"],
                            d["width"],
                            str(gate_counts)
                        ]
                    else:
                        task_row += [-1, -1, "{}"]

                task_writer.writerow(task_row)

                # ---- DETAILS ROWS ----
                for r in results[1:]:
                    details_row = [
                        task_id,
                        r["backend"],
                        r["width"],
                        r["depth"],
                        str(r["gates_errors"]),
                        r["critical_path"],
                        str(r["readout_errors"])
                    ]
                    details_writer.writerow(details_row)

    finally:
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER)
            print("Cleanup complete.")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()