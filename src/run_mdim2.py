import copy
import gym
import torch
import xarray as xr
import numpy as np
import os
import time
from ars import ars
from common import *
from seagul.rl.ars.ars_pipe2 import ARSAgent

env_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
init_names = ["identity", "madodiv", "identity"]

post_fns = [mdim_div2]

torch.set_default_dtype(torch.float64)
num_experiments = len(post_fns)
num_seeds = 10
num_epochs = 250
n_workers = 12
n_delta = 60
n_top = 20
exp_noise = .025
n_postprocess_runs = 5

save_dir = "./data_mdim_npp"
os.makedirs(save_dir)


start = time.time()
env_config = {}
bad_list = []
for env_name, init_name in zip(env_names, init_names):
    init_data = torch.load(f"./data17/{env_name}.xr")
    init_policy_dict = init_data.policy_dict

    env = gym.make(env_name, **env_config)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]
    policy_dict = {fn.__name__: [] for fn in post_fns}
    agent_dict = {fn.__name__: [] for fn in post_fns}

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
        attrs={"policy_dict": policy_dict, "agent_dict": agent_dict, "post_fns": post_fns, "env_name": env_name,
               "hyperparams": {"num_experiments": num_experiments, "num_seeds": num_seeds, "num_epochs": num_epochs,
                               "n_workers": n_workers, "n_delta": n_delta, "n_top": n_top, "exp_noise": exp_noise},
               "env_config": env_config})

    for post_fn in post_fns:
        for i, seed in enumerate(np.random.randint(0,2**32-1,num_seeds)):
            print(f'starting env {env_name} with initial policy {init_name}')
            policy = init_policy_dict[init_name][i]
            try:
                agent = ARSAgent(env_name, policy, seed, n_workers=n_workers, n_delta=n_delta,
                                 n_top=n_top, exp_noise=exp_noise, postprocessor=post_fn,
                                 env_config=env_config, n_postprocess_runs=n_postprocess_runs)

                agent.learn(num_epochs)
            except:
                print("something broke")
                bad_list.append((post_fn.__name__, i))

            print(f"{env_name}, {post_fn.__name__}, {i}, {time.time() - start}")
            data.policy_dict[post_fn.__name__].append(copy.deepcopy(policy))
            data.agent_dict[post_fn.__name__].append(copy.deepcopy(agent))
            data.rews.loc[post_fn.__name__, i, :] = agent.lr_hist
            data.post_rews.loc[post_fn.__name__, i, :] = agent.r_hist

    torch.save(data, f"{save_dir}/{env_name}.xr")
