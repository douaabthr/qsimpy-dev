
import os
import ctypes
try:
    # Change the path below if your anaconda is installed in a different place
    dll_path = r"C:\Users\Imane\anaconda3\envs\qsim\lib\site-packages\torch\lib\c10.dll"
    ctypes.CDLL(dll_path)
    print("Successfully pre-loaded c10.dll")
except Exception as e:
    print(f"Pre-load failed: {e}")
    



from env_creator import qsimpy_env_creator

import os


# -------------------------------------------------
# INIT RAY
# -------------------------------------------------

from env_creator import qsimpy_env_creator
from ray.tune.registry import register_env
from ray.rllib.algorithms import Algorithm

register_env("QSimPyEnv", qsimpy_env_creator)

env = qsimpy_env_creator ( 
    env_config = {
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": r"C:\Users\Imane\OneDrive\Bureau\helper\qdataset_1000_sub_26.csv",
        "dataset_errors": r"D:\qsimpy-dev\qdataset\custom\qd-generation\tasks_backend_details.csv",
         
    }
)

checkpoint_path = r"D:\qsimpy-dev\results\custom\fideltyOnly_baseFidelity_5000subset_26task_5backends_100it\DQN_QSimPyEnv_ac07c_00000_0_lr=0.0100,n_step=5,num_atoms=10,train_batch_size=78_2026-04-05_10-49-00\checkpoint_000009"

model = Algorithm.from_checkpoint(checkpoint_path)

num_ep = 1

for ep in range(num_ep):
    obs= env.reset()
    finished = False
    ep_reward=0

    while not finished: 
        formatted_obs = obs if not isinstance(obs,tuple) else obs[0]
        action = model.compute_single_action(formatted_obs,explore = False)
        print(f"Episode {ep}, Action taken: {action}")
        obs, reward, finished, _, info = env.step(action)
        ep_reward += reward

        if finished:
            print(f"Episode{ep} finished with reward {ep_reward} and info {info} ")
            break
print("Evaluation completed.")
env.close()