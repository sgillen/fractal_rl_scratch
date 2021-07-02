import copy
import gym
import torch
import xarray as xr
import numpy as np
import os
import time
#from common import *
from seagul.mesh import target_d_div, target_d_divn
import pybullet_envs

from gym.envs.registration import register
register(id='A1GymEnv-v1' , entry_point='motion_imitation.envs.gym_envs:A1GymEnv', max_episode_steps=1000)

# env_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
# init_names = ["identity", "madodiv", "identity"]
# init_dir = "./data17_repl/"
# save_dir = "./data17_repl_mdim_td/"
# post_fns = [identity, target_d_div, mdim_div]


# env_names = ["HalfCheetah-v2"]
# init_names = ["identity"]
# init_dir = "./data17_repl2/"
# save_dir = "./data17_repl4_mdim_td/"
# post_fns = [identity, target_d_div, mdim_div, better_target_d_div]


env_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
init_names = ["identity", "identity", "identity"]
init_dir = "./data17_repl2/"
save_dir = "./data17_repl4_mdim_td/"
post_fns = [target_d_divn, identity, target_d_div, mdim_div ]



#env_names = ["Humanoid-v2"]
#init_names = ["identity"]
#post_fns = [cdim_div]

#init_dir = "./data_hm0/"
#save_dir = "./data_hm0_cdim/"

# num_experiments = len(post_fns)
# num_seeds = 10
# num_epochs = 750
# n_workers = 12
# n_delta = 240
# n_top = 240
# exp_noise = .0075


#env_names = ['A1GymEnv-v1']
#init_names = ["identity"]
# post_fns = [target_d_div, mdim_div, cdim_div]
# init_dir = "./data_a1f2/"
# save_dir = "./data_a1f2_mdim_t_n/"

assert not os.path.isdir(save_dir)

num_experiments = len(post_fns)
num_seeds = 10
num_epochs = 250
n_workers = 12
n_delta = 60
n_top = 20
exp_noise =.02
step_size = .025
step_schedule=[.02, .02]
exp_schedule=[.025, .025]


start = time.time()
env_config = {}
bad_list = []

for env_name, init_name in zip(env_names, init_names):
    init_data = torch.load(f"{init_dir}/{env_name}/data.xr")
    init_model_dict = init_data.model_dict

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
                               "step_schedule":step_schedule, "exp_schedule":exp_schedule},
               "env_config": env_config})


    for post_fn in post_fns:
        i = 0;
        for save_file in os.scandir(f"{init_dir}/{env_name}"):

            if save_file.name.split(".")[-1] != "pkl":
                print(save_file.name.split(".")[-1])
                continue
            else:
                agent = torch.load(f"{init_dir}/{env_name}/{save_file.name}")
                seed = int(save_file.name.split(".")[-2].split("_")[-1])
                init_epoch = len(agent.r_hist)

                agent.postprocessor = post_fn
                agent.seed = seed
                
            print(f"starting env {env_name} with initial policy {init_name}")
            
            try:
                model, r_hist, lr_hist = agent.learn(num_epochs)
            except:
                print("something broke")
                bad_list.append((post_fn.__name__, i))
                
            print(f"{env_name}, {post_fn.__name__}, {i}, {time.time() - start}")
            data.model_dict[post_fn.__name__].append(copy.deepcopy(agent.model))
            data.rews.loc[post_fn.__name__, i, :] = agent.lr_hist[init_epoch:]
            data.post_rews.loc[post_fn.__name__, i, :] = agent.r_hist[init_epoch:]
            os.makedirs(f"{save_dir}/{env_name}", exist_ok=True)
            torch.save(agent, f"{save_dir}/{env_name}/agent_{seed}.pkl")
            i+=1
            

        torch.save(data, f"{save_dir}/{env_name}/data.xr")
