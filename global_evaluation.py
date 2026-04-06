
import os
import ctypes
try:
    # Change the path below if your anaconda is installed in a different place
    dll_path = r"C:\Users\Imane\anaconda3\envs\qsim\lib\site-packages\torch\lib\c10.dll"
    ctypes.CDLL(dll_path)
    print("Successfully pre-loaded c10.dll")
except Exception as e:
    print(f"Pre-load failed: {e}")
    



import os

from ray.tune.registry import register_env
from ray.rllib.algorithms import Algorithm

from env_creator import qsimpy_env_creator

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HeuristicSolutions:
    def __init__(self, env, num_episodes=50):
        self.env = env
        self.num_episodes = num_episodes
        self.results = {}
        self.rr_index = 0
        self.greedy_index = 0

        self.model = None

    # -------------------- RUN --------------------
    def run(self, control):
        print(f"\nRunning {control}...")
        self.env.round = 1
        data = []

        for episode in tqdm(range(self.num_episodes), desc=f"{control}"):
            obs = self.env.reset()
            self.env.setup_quantum_resources()
            self.rr_index = 0

            terminated = False
            # -------- per episode accumulators --------
            total_completion_time = 0
            total_execution_time = 0
            total_rescheduling = 0
            fidelities = []

            while not terminated:

                # -------- ACTION --------

                if control == "greedy":
                    action = self.greedy(self.greedy_index)
                elif control == "random":
                    action = self.random()
                elif control == "round_robin":
                    action = self.round_robin()
                elif control == "greedy_error":
                    action = self.greedy_error(self.greedy_index)
                elif control == "BestFidelity":	
                    action= self.BestFidelity(self.greedy_index) # est ce que je dois donner greedy_index !!
                elif control== "dqn":
                    action = self.dqn(obs)
                else:
                    raise ValueError(control)

                obs, reward, terminated, _, info = self.env.step(action)
                self.greedy_index = (self.greedy_index + 1) % self.env.n_qnodes

                if reward > 0:
                    self.greedy_index = 0
                # -------- METRICS --------
                    if "scheduled_qtask" in info:
                        task = info["scheduled_qtask"]

                        total_completion_time += (
                            task.waiting_time + task.execution_time
                        )
                        total_execution_time += task.execution_time
                        total_rescheduling += task.rescheduling_count

                    if "fidelity" in info:
                        fidelities.append(info["fidelity"])

            # Finish simulation
            self.env.qsp_env.run()

            # -------- per episode aggregation --------
            avg_fidelity = np.mean(fidelities) if fidelities else 0

            data.append({
                "episode": episode,
                "fidelity": avg_fidelity,                 # mean
                "completion_time": total_completion_time, # sum
                "execution_time": total_execution_time,   # sum
                "rescheduling": total_rescheduling        # sum
            })

        df = pd.DataFrame(data)

        # Save CSV
        output_dir = os.path.join("results", "custom", "global","5_backends")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(
            os.path.join(output_dir, f"{control}_all_metrics.csv"),
            index=False
        )
        self.results[control] = df

    

    
    # -------------------- HEURISTICS --------------------
    def greedy(self, greedy_index):
        greedy_strategy = sorted(self.env.qnodes, key=lambda x: x.next_available_time)
        return self.env.qnodes.index(greedy_strategy[greedy_index])

    def random(self):
        return self.env.action_space.sample()

    def round_robin(self):
        action = self.rr_index % self.env.n_qnodes
        self.rr_index += 1
        return action

    def greedy_error(self, greedy_index, g_error="Readout_assignment_error"):
        greedy_strategy = sorted(
            self.env.qnodes,
            key=lambda x: (x.next_available_time, x.error[g_error])
        )
        return self.env.qnodes.index(greedy_strategy[greedy_index])

    def BestFidelity(self, greedy_index):
		# 1. Get the current task from the environment
        current_task = self.env.current_qtask
		
		# 2. Get the dataset that contains the error rates (passed in env_config)
        dataset = self.env.qtask_dataset
		# --- VERIFICATION START ---
		# This prints the actual file paths stored inside the dataset object
		#print(f"DEBUG: Dataset is using circuit file: {dataset.circuit_file}")
		#print(f"DEBUG: Dataset is using errors file: {dataset.errors_file}")
		#print("****dataset", dataset)
		
		# 3. Calculate fidelity for this task on every available QNode
        node_fidelity_pairs = []
        for node in self.env.qnodes:
            # Check if the node has enough qubits for the task first
            if node.qubit_number >= current_task.qubit_number:
                # Use the compute_fidelity method from your QNode class
                fidelity = node.compute_fidelity(current_task, dataset)
            
                node_fidelity_pairs.append((node, fidelity))
            else:
                # If the task is too big for the node, set fidelity to 0
                node_fidelity_pairs.append((node, -1.0))

        #print("node_fidelity_pairs", node_fidelity_pairs)



        # 4. Sort nodes by fidelity in descending order (highest first)
        # We sort the original qnodes list based on the calculated fidelity
        sorted_strategy = sorted(
            node_fidelity_pairs, 
            key=lambda x: x[1], 
            reverse=True
        )

        # --- PRINT 2: Sorted Strategy ---
        # 5. Return the index of the node corresponding to the greedy_index (for fallback)
        selected_node = sorted_strategy[greedy_index][0] # 0:prendre le premier element dans la paire (node, fidelity)
        # We must return the index of this node within the original self.env.qnodes list

        return self.env.qnodes.index(selected_node)

    # -------------------- DQN --------------------
    def dqn(self,obs):

        formatted_obs = obs if not isinstance(obs,tuple) else obs[0]
        action = self.model.compute_single_action(formatted_obs,explore = False)
        
        return action
    

    # -------------------- PLOT CURVES --------------------
    def plot_curves(self):
        metrics = ["fidelity", "completion_time", "execution_time", "rescheduling"]

        for metric in metrics:
            plt.figure(figsize=(10, 6))

            for name, df in self.results.items():
                plt.plot(
                    df["episode"],
                    df[metric],
                    marker='o',
                    linestyle='-',
                    label=name
                )

            plt.xlabel("Episode")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{metric.replace('_', ' ').title()} over Episodes")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

    # -------------------- PLOT SUMMARY (MEAN + STD) --------------------
    def plot_summary(self):
        metrics = ["fidelity", "completion_time", "execution_time", "rescheduling"]

        for metric in metrics:
            names = []
            means = []
            stds = []

            for name, df in self.results.items():
                names.append(name)
                means.append(df[metric].mean())  # mean of episodes
                stds.append(df[metric].std())

            plt.figure(figsize=(8, 5))
            plt.bar(names, means, yerr=stds, capsize=5)
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"Average {metric.replace('_', ' ').title()}")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.show()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    from env_creator import qsimpy_env_creator

    env_config = {
        "dataset": r"D:\qsimpy-dev\qdataset\custom\datasets\qdataset_50_sub_26.csv",
        "dataset_errors": r"D:\qsimpy-dev\qdataset\custom\qd-generation\tasks_backend_details.csv",
    }

    env = qsimpy_env_creator(env_config)
    heuristics = HeuristicSolutions(env, num_episodes=50)

    algorithms = ["greedy", "random", "round_robin", "greedy_error","BestFidelity","dqn"]

    for algo in algorithms:
        if algo=="dqn":
            register_env("QSimPyEnv", qsimpy_env_creator)

            checkpoint_path = r"D:\qsimpy-dev\results\custom\DQN\imane\DQN_base_fidelity_5000_subset_26_task_5_backends_100_it\DQN_QSimPyEnv_2237e_00000_0_lr=0.0100,n_step=5,num_atoms=10,train_batch_size=78_2026-04-03_18-54-17\checkpoint_000009"

            heuristics.model = Algorithm.from_checkpoint(checkpoint_path)

        heuristics.run(algo)

    # -------- plots --------
    heuristics.plot_curves()
    heuristics.plot_summary()