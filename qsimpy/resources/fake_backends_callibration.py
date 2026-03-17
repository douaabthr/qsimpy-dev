import json
import os
import random
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

# ============================================================
# Random Fallback Distributions (NO FIXED CONSTANTS)
# ============================================================

def fallback_t1():
    return random.uniform(50e-6, 200e-6)

def fallback_t2():
    return random.uniform(30e-6, 180e-6)

def fallback_frequency():
    return random.uniform(4.5e9, 5.5e9)

def fallback_readout_error():
    return random.uniform(0.005, 0.05)

def fallback_gate_error(num_qubits):
    if num_qubits == 1:
        return random.uniform(1e-4, 5e-3)
    else:
        return random.uniform(1e-3, 1e-2)

def fallback_gate_duration(num_qubits):
    if num_qubits == 1:
        return random.uniform(10, 100)      # ns
    else:
        return random.uniform(50, 500)      # ns


# ============================================================
# Provider
# ============================================================

provider = FakeProviderForBackendV2()
ibmq_data = {}

# ============================================================
# Iterate Backends
# ============================================================

for backend in provider.backends():

    short_name = backend.name.replace("fake_", "")

    ibmq_data[short_name] = {
        "system_info": {},
        "qubit_properties": {},
        "gate_metadata": {}
    }

    # -------------------------
    # System Info
    # -------------------------

    ibmq_data[short_name]["system_info"]["num_qubits"] = backend.num_qubits
    ibmq_data[short_name]["system_info"]["backend_version"] = getattr(backend, "backend_version", None)
    # Quantum Volume (Fallback for simulators/fake backends)
    qv = getattr(backend, "quantum_volume", None) 
    if qv is None or qv == 0: 
        qv = random.choice([16, 32, 64, 128, 256]) 
        ibmq_data[short_name]["system_info"]["quantum_volume"] = qv
    # CLOPS (Circuit Layer Operations Per Second) 
    try:
        props = backend.properties()
        clops = getattr(props, "clops", None)
    except Exception:
        clops = None

    # If missing or zero → use random fallback
    if clops is None or clops == 0:
        clops = random.randint(800, 3000)

    ibmq_data[short_name]["system_info"]["clops"] = clops
    # Qubit Properties
    # -------------------------

    for q in range(backend.num_qubits):

        qubit_info = {}

        try:
            qp = backend.qubit_properties(q)

            qubit_info["t1"] = getattr(qp, "t1", None) or fallback_t1()
            qubit_info["t2"] = getattr(qp, "t2", None) or fallback_t2()
            qubit_info["frequency"] = getattr(qp, "frequency", None) or fallback_frequency()

        except Exception:
            qubit_info["t1"] = fallback_t1()
            qubit_info["t2"] = fallback_t2()
            qubit_info["frequency"] = fallback_frequency()

        # Readout Error
        try:
            props = backend.properties()
            ro = props.readout_error(q)
            if ro is None:
                ro = fallback_readout_error()
            qubit_info["readout_error"] = ro
        except Exception:
            qubit_info["readout_error"] = fallback_readout_error()

        ibmq_data[short_name]["qubit_properties"][str(q)] = qubit_info

    # -------------------------
    # Gate Metadata (FIXED FOR RZ = 0 CASE)
    # -------------------------

    target = backend.target

    for inst_name in target.operation_names:

        ibmq_data[short_name]["gate_metadata"][inst_name] = {}

        inst_map = target.get(inst_name)

        if inst_map is None:
            continue

        for qubits, inst_props in inst_map.items():

            if qubits is None:
                continue

            # Default values
            error = None
            duration = None

            if inst_props is not None:
                error = getattr(inst_props, "error", None)
                duration = getattr(inst_props, "duration", None)

            # Convert duration to ns
            if duration is not None:
                duration = duration * 1e9

            # 🔥 IMPORTANT:
            # Treat 0 as missing (fixes rz case)
            if error in [0, None]:
                error = fallback_gate_error(len(qubits))

            if duration in [0, None]:
                duration = fallback_gate_duration(len(qubits))

            qubit_key = ",".join(map(str, qubits))

            ibmq_data[short_name]["gate_metadata"][inst_name][qubit_key] = {
                "error": error,
                "duration_ns": duration
            }

# ============================================================
# Save JSON
# ============================================================

path = "./qsimpy/resources/backend_calibration_data.json"
os.makedirs(os.path.dirname(path), exist_ok=True)

with open(path, "w") as f:
    json.dump(ibmq_data, f, indent=4)

print("Calibration data saved successfully.")