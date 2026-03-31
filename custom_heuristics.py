import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from env_creator import qsimpy_env_creator
# import wandb
import shutil


# wandb.login()

class HeuristicSolutions:
    def __init__(self, env, num_episodes=5, project_name="qsimpy-heuristics"):
        self.env = env
        self.num_episodes = num_episodes
        self.project_name = project_name
        self.results = []
        self.rr_index = 0
        self.greedy_index = 0

        # Optional: local folder to store dataset copy
        self.local_dataset_copy = "./results/dataset_used.csv"
        if hasattr(self.env, "dataset_path"):
            os.makedirs(os.path.dirname(self.local_dataset_copy), exist_ok=True)
            shutil.copy(self.env.dataset_path, self.local_dataset_copy)
            print(f"Local dataset copy saved to {self.local_dataset_copy}")

    # -------------------- Run heuristics --------------------
    def run(self, control):
        self.results = []
        self.env.round = 1
        print(f"Running heuristic: {control}")

        # -------------------- Init W&B --------------------
        # wandb.init(
        #     project=self.project_name,
        #     name=control,
        #     config={
        #         "algorithm": control,
        #         "episodes": self.num_episodes,
        #         "dataset": os.path.basename(env.qtask_dataset.filename)
        #     }
        # )

        #  # -------------------- Dataset Artifact --------------------
        # dataset_path = env.qtask_dataset.filename

        # artifact = wandb.Artifact(
        #     name="qsimpy_dataset",
        #     type="dataset",
        #     description="Task dataset"
        # )
        # artifact.add_file(dataset_path)

        # # Log + get versioned artifact
        # logged_artifact = wandb.log_artifact(artifact)

        # # 🔥 CRUCIAL: link artifact to run
        # wandb.run.use_artifact(logged_artifact)

        # -------------------- Run episodes --------------------
        for episode in tqdm(range(self.num_episodes), desc="Episodes"):
            arr_temp = {"total_completion_time": 0.0, "rescheduling_count": 0.0}
            terminated = False

            # Reset environment and quantum resources
            self.env.reset()
            self.env.setup_quantum_resources()
            self.rr_index = 0

            while not terminated:
                # Select action
                if control == "greedy":
                    action = self.greedy(self.greedy_index)
                elif control == "random":
                    action = self.random()
                elif control == "round_robin":
                    action = self.round_robin()
                elif control == "greedy_error":
                    action = self.greedy_error(self.greedy_index)
                else:
                    raise ValueError(f"Unknown control: {control}")

                obs, reward, terminated, done, info = self.env.step(action)

                # Round-robin / greedy index update
                self.greedy_index = (self.greedy_index + 1) % self.env.n_qnodes

                if reward > 0:
                    self.greedy_index = 0
                    arr_temp["total_completion_time"] += (
                        info["scheduled_qtask"].waiting_time
                        + info["scheduled_qtask"].execution_time
                    )
                    arr_temp["rescheduling_count"] += info["scheduled_qtask"].rescheduling_count

            self.env.qsp_env.run()
            self.results.append(arr_temp)

        #     # -------------------- Log episode metrics to W&B --------------------
        #     wandb.log({
        #         "episode": episode,
        #         "total_completion_time": arr_temp["total_completion_time"],
        #         "rescheduling_count": arr_temp["rescheduling_count"],
        #         "algorithm": control
        #     })

        # wandb.finish()
        # print(f"Finished W&B run for {control}")

    # -------------------- Heuristic strategies --------------------
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

    # -------------------- Optional: plot results locally --------------------
    def plot_results(self, result_files):
        for file_path, color, label in result_files:
            df = pd.read_csv(file_path)
            plt.plot(df["Episode"], df["Total Completion Time"], ".-", color=color, label=label)
        plt.ylabel("Total Completion Time")
        plt.xlabel("Episode")
        plt.legend(loc=2)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
        plt.show()


# -------------------- Main --------------------
if __name__ == "__main__":

    env_config = {
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        # "dataset": r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\datasets\qdataset_10_sub_15.csv",
        # "dataset_errors": r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\qd-generation\tasks_backend_details.csv",
        "dataset": r"D:\qsimpy-dev\qdataset\custom\datasets\qdataset_10_sub_15.csv",
        "dataset_errors": r"D:\qsimpy-dev\qdataset\custom\qd-generation\tasks_backend_details.csv",

    }

    env = qsimpy_env_creator(env_config)
    heuristics = HeuristicSolutions(env, num_episodes=5)

    # Run all heuristics with W&B logging
    for algo in ["greedy", "random", "round_robin", "greedy_error"]:
        heuristics.run(algo)