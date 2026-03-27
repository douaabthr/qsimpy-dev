def get_avg_gate_error(self, gate_name):
    errors = []

    for key, val in self.error.items():
        if key.startswith(gate_name):
            errors.append(val)

    if len(errors) == 0:
        return 0.01  # fallback

    return sum(errors) / len(errors)

def get_avg_gate_error(self, gate_name):
    errors = []

    for key, val in self.error.items():
        if key.startswith(gate_name):
            errors.append(val)

    if len(errors) == 0:
        return 0.01  # fallback

    return sum(errors) / len(errors)

import numpy as np

def compute_fidelity(self, task):
    """
    Fidelity using backend-specific transpiled circuit
    """
    if self.qnode_name not in task.qtask_data:
        return 0.0

    gate_counts = task.qtask_data[self.qnode_name]["gates"]

    log_fidelity = 0.0

    for gate, count in gate_counts.items():
        avg_error = self.get_avg_gate_error(gate)

        log_fidelity += count * np.log(1 - avg_error + 1e-12)

    return np.exp(log_fidelity)