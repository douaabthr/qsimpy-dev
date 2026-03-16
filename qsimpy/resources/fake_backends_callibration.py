import json
import os
import random
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

# -------------------------
# Optional: Reproducibility
# -------------------------
random.seed(42)
np.random.seed(42)

# -------------------------
# Helper Fallback Functions
# -------------------------

def fallback_t1():
    # realistic superconducting qubit range
    return max(20e-6, random.gauss(120e-6, 20e-6))

def fallback_t2():
    return max(10e-6, random.gauss(100e-6, 20e-6))

def fallback_frequency():
    # typical transmon band
    return random.uniform(4.5e9, 5.5e9)

def fallback_clops(num_qubits):
    # scale with backend size
    base = 4000 / max(num_qubits, 1)
    return max(500, int(random.gauss(base, 300)))

def fallback_qv(num_qubits):
    # physically reasonable scaling
    qv = 2 ** (num_qubits // 2)
    return min(qv, 256)

def fallback_readout_error():
    # realistic readout error range for superconducting qubits
    return random.uniform(0.01, 0.05)

# -------------------------
# Provider
# -------------------------

provider = FakeProviderForBackendV2()
ibmq_data = {}

# -------------------------
# Iterate Backends
# -------------------------

for backend in provider.backends():

    short_name = backend.name.replace("fake_", "")

    ibmq_data[short_name] = {
        "system_info": {},
        "qubit_properties": {},
        "gate_metadata": {}
    }

    # -------------------------
    # 1. SYSTEM INFO
    # -------------------------

    ibmq_data[short_name]["system_info"]["num_qubits"] = backend.num_qubits
    ibmq_data[short_name]["system_info"]["sample_name"] = getattr(backend, "sample_name", None)
    ibmq_data[short_name]["system_info"]["backend_version"] = getattr(backend, "backend_version", None)

    # Quantum Volume
    qv = getattr(backend, "quantum_volume", None)
    if not qv or qv == 0:
        qv = fallback_qv(backend.num_qubits)

    ibmq_data[short_name]["system_info"]["quantum_volume"] = qv

    # CLOPS
    try:
        props = backend.properties()
        clops = getattr(props, "clops", None)
    except Exception:
        clops = None

    if not clops or clops == 0:
        clops = fallback_clops(backend.num_qubits)

    ibmq_data[short_name]["system_info"]["clops"] = clops

    # -------------------------
    # 2. QUBIT PROPERTIES
    # -------------------------

    for q in range(backend.num_qubits):

        qubit_info = {}

        # Coherence + Frequency
        try:
            qp = backend.qubit_properties(q)
            qubit_info["t1"] = getattr(qp, "t1", fallback_t1())
            qubit_info["t2"] = getattr(qp, "t2", fallback_t2())
            qubit_info["frequency"] = getattr(qp, "frequency", fallback_frequency())
        except Exception:
            qubit_info["t1"] = fallback_t1()
            qubit_info["t2"] = fallback_t2()
            qubit_info["frequency"] = fallback_frequency()

        # Readout Errors
        try:
            props = backend.properties()
            readout_error = props.readout_error(q)
           

        except Exception:
            readout_error = fallback_readout_error()

        qubit_info["readout_error"] = readout_error

        ibmq_data[short_name]["qubit_properties"][str(q)] = qubit_info

    # -------------------------
    # 3. GATE METADATA
    # -------------------------

    target = backend.target

    for inst_name in target.operation_names:

        ibmq_data[short_name]["gate_metadata"][inst_name] = {}

        inst_map = target.get(inst_name)

        if inst_map is not None and isinstance(inst_map, dict):

            for qubits, inst_props in inst_map.items():

                if qubits is None:
                    continue

                prop_dict = {
                    "error": 0.0,
                    "duration_ns": 0.0
                }

                if inst_props is not None:

                    # Error
                    if hasattr(inst_props, "error") and inst_props.error is not None:
                       
                        prop_dict["error"] = inst_props.error
                    else:
                       
                        prop_dict["error"] = (
                            random.uniform(1e-4, 5e-4)
                            if len(qubits) == 1
                            else random.uniform(1e-3, 5e-3)
                        )

                    # Duration
                    if hasattr(inst_props, "duration") and inst_props.duration is not None:
                        prop_dict["duration_ns"] = inst_props.duration * 1e9
                    else:
                        prop_dict["duration_ns"] = (
                            35.0 if len(qubits) == 1 else 320.0
                        )

                qubit_key = ",".join(map(str, qubits))
                ibmq_data[short_name]["gate_metadata"][inst_name][qubit_key] = prop_dict

# -------------------------
# SAVE FILE
# -------------------------

path = "./qsimpy/resources/backend_calibration_data.json"
os.makedirs(os.path.dirname(path), exist_ok=True)

with open(path, "w") as f:
    json.dump(ibmq_data, f, indent=4)

print(f"Successfully saved data for {len(ibmq_data)} backends to {path}")