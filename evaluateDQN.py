from env_creator import qsimpy_env_creator
from ray.tune.registry import register_env
from ray.rllib.algorithms import Algorithm

register_env("QSimPyEnv", qsimpy_env_creator)

env = qsimpy_env_creator ( 
    env_config = {
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\datasets\qdataset_100_sub_26.csv",
        "dataset_errors": r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\qd-generation\tasks_backend_details.csv",

    }
)

checkpoint_path = r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\results\custom\DQN_QCE_1000\DQN_QSimPyEnv_75a24_00000_0_lr=0.0100,n_step=5,num_atoms=10,train_batch_size=78_2026-04-05_15-12-20\checkpoint_000020"

model = Algorithm.from_checkpoint(checkpoint_path)

num_ep = 9

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
        # print("Q-values:", info["q_values"])
        if finished:
            print(f"Episode{ep} finished with reward {ep_reward} and info {info} ")
            break
print("Evaluation completed.")
env.close()




