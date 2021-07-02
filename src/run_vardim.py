from common import *
from seagul.rl.ars import ARSAgent
import copy
import gym
import time
import xarray as xr
import numpy as np
import os
import pybullet_envs

from gym.envs.registration import register

register(id='A1GymEnv-v1' , entry_point='motion_imitation.envs.gym_envs:A1GymEnv', max_episode_steps=1000)

#env_names = ["MinitaurBulletEnv-v0"]#["HalfCheetah-v2"]#, "Hopper-v2", "Walker2d-v2"]
#post_fns = [identity]
#env_names = ["Humanoid-v2"]




env_names = ["HalfCheetah-v2"]#, "Hopper-v2", "Walker2d-v2"]
post_fns = [identity]

#env_names = ['A1GymEnv-v1']
#post_fns = [identity]# variodiv, madodiv]

#torch.set_default_dtype(torch.float64)
num_experiments = len(post_fns)
num_seeds = 10
num_epochs = 750
n_workers = 12
n_delta = 60
n_top = 20
exp_noise =.02
step_size = .025
step_schedule=[.02, .02]
exp_schedule=[.025, .025]


save_dir = "./data17_repl2/"
env_config = {}

assert not os.path.isdir(save_dir)


start = time.time()
for env_name in env_names:
    env = gym.make(env_name, **env_config)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]
    model_dict = {fn.__name__: [] for fn in post_fns}

    rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, num_epochs)),
                           dims=("post", "trial", "epoch"),
                           coords={"post": [fn.__name__ for fn in post_fns]})

    post_rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, num_epochs)),
                                dims=("post", "trial", "epoch"),
                                coords={"post": [fn.__name__ for fn in post_fns]})

    data = xr.Dataset(
        {"rews": rewards,
         "post_rews": post_rewards},
        coords={"post": [fn.__name__ for fn in post_fns]},
        attrs={"model_dict": model_dict, "post_fns": post_fns, "env_name": env_name,
               "hyperparams": {"num_experiments": num_experiments, "num_seeds": num_seeds, "num_epochs": num_epochs,
                               "n_workers": n_workers, "n_delta": n_delta, "n_top": n_top, "exp_noise": exp_noise,
                               "step_schedule":step_schedule, "exp_schedule":exp_schedule
                               },
               "env_config": env_config})

    for post_fn in post_fns:
        for i in range(num_seeds):
            seed = int(np.random.randint(0,2**32-1))
            agent =  ARSAgent(env_name, seed, n_workers=n_workers, n_delta=n_delta,
                              n_top=n_top, exp_noise=exp_noise, step_size=step_size,
                              step_schedule=step_schedule, exp_schedule=exp_schedule,
                              postprocessor=post_fn,
                              env_config=env_config)

            agent.learn(num_epochs)
            
            print(f"{env_name}, {post_fn.__name__}, {i}, {time.time() - start}")
            data.model_dict[post_fn.__name__].append(copy.deepcopy(agent.model))
            data.rews.loc[post_fn.__name__, i, :] = agent.lr_hist
            data.post_rews.loc[post_fn.__name__, i, :] = agent.r_hist
            os.makedirs(f"{save_dir}/{env_name}", exist_ok=True)
            torch.save(agent, f"{save_dir}/{env_name}/agent_{seed}.pkl")

    torch.save(data, f"{save_dir}/{env_name}/data.xr")
