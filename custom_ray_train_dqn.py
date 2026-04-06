
# import os
# import ctypes
# try:
#     # Change the path below if your anaconda is installed in a different place
#     dll_path = r"C:\Users\Imane\anaconda3\envs\qsim\lib\site-packages\torch\lib\c10.dll"
#     ctypes.CDLL(dll_path)
#     print("Successfully pre-loaded c10.dll")
# except Exception as e:
#     print(f"Pre-load failed: {e}")

import argparse
import ray
from ray import tune, air, train
from ray.tune.registry import register_env
from env_creator import qsimpy_env_creator
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.framework import try_import_tf
from ray.tune.analysis import ExperimentAnalysis
import os
from ray.air import CheckpointConfig


ray.init(
    ignore_reinit_error=True,
    local_mode=True  # forces all rollouts in main process
)

from ray.rllib.algorithms.callbacks import DefaultCallbacks

tf1, tf, tfv = try_import_tf()
parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=0)

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)   

parser.add_argument(
    "--stop-iters", type=int, default=20, help="Number of iterations to train."
)
parser.add_argument(
    # "--stop-timesteps", type=int, default=22, help="Number of timesteps to train."
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)

class MyCallbacks(DefaultCallbacks):
    def on_episode_step(self, *, episode, **kwargs):
        info = episode._last_infos  

        if info and "fidelity" in info:
            episode.custom_metrics.setdefault("fidelity", []).append(info["fidelity"])
if __name__ == "__main__":
    args = parser.parse_args()

    # ray.init(num_cpus=args.num_cpus or None)
    register_env("QSimPyEnv", qsimpy_env_creator)

    replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

    config = (
        DQNConfig().rollouts(num_rollout_workers=0,
           rollout_fragment_length=50
)
        .framework(framework=args.framework)
        .environment(
            env="QSimPyEnv",
            env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\datasets\qdataset_1000_sub_26.csv",
                "dataset_errors": r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\qd-generation\tasks_backend_details.csv",
            },
            
        )
        # 0 workers → tout se fait sur le process principal
        .training(
            lr=tune.grid_search([0.01]),
            # le réseau s’entraîne sur 78 transitions tirées du replay buffer.
            train_batch_size=tune.grid_search([78]),
            replay_buffer_config=replay_config,
            num_atoms=tune.grid_search(
                [
                    10
                ]
            ),
            n_step=tune.grid_search([5]),
            noisy=True, 
            v_min=-10.0,
            v_max=10.0,
        )
        .callbacks(MyCallbacks)
    )

    stop_config = {
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }
    
    # Get the absolute path of the current directory
    current_directory = os.getcwd()

    # Append the "result" folder to the current directory path
    result_directory = os.path.join(current_directory, "results")

    # Create the storage_path with the "file://" scheme
    storage_path = r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\results\custom"
    # storage_path=r"D:\qsimpy-dev\results\custom"

#  debut de l'entrainement
    results = tune.Tuner(
        "DQN",
        run_config=air.RunConfig(
            stop=stop_config,
            # Save checkpoints every 10 iterations.
            checkpoint_config=CheckpointConfig(checkpoint_frequency=10),
            storage_path=storage_path, 
            name="DQN_QCE_1000"
        ),
        param_space=config.to_dict(),
    ).fit()

    ray.shutdown()