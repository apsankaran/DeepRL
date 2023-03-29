# Stable Baselines SAC

# import statements

import gym
import numpy as np
import pandas as pd
import os

from stable_baselines3.common.monitor import Monitor
from CustomPPOModel2 import CustomPPO
from CustomCallback import CustomCallback

import matplotlib.pyplot as plt

import pdb

def mkdr(name):
    if not os.path.exists(name):
        os.makedirs(name)

def get_CI(data, confidence = 0.95):

    if (np.array(data) == None).all():
        return {}
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    stats = {}
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    err = z * (std / np.sqrt(n))
    lower = mean - z * (std / np.sqrt(n))
    upper = mean + z * (std / np.sqrt(n))
    stats = {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'err': err,
        'max': np.max(data),
        'min': np.min(data)
    }
    return stats

# create subplots
figure, axis = plt.subplots(2, 1, figsize=(20, 25))
figure.tight_layout(pad=5.0)

for gae_lambda, color in zip([0, 0.5, 1], ['r', 'b', 'g']):

    total_timesteps = 4000000
    eval_freq = 500
    timesteps = [(i+1)*eval_freq for i in range(total_timesteps//eval_freq)][4:]
    ranks_df = pd.DataFrame(columns=timesteps)
    returns_df = pd.DataFrame(columns=timesteps)

    for _ in range(3):

        env = Monitor(gym.make('Ant-v4'))
    
        mkdr('logs')
        mkdr('output')
    
        custom_callback = CustomCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=eval_freq,
                             deterministic=True, render=False)
                             
        n_epochs=10 # default=10 
        model = CustomPPO('MlpPolicy', env, n_epochs=n_epochs, gae_lambda=gae_lambda, verbose=1)
        model.learn(total_timesteps=total_timesteps, callback=custom_callback)
    
        # get results
        env_name = env.unwrapped.spec.id
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        npzfile = np.load(os.getcwd() + "/logs/evaluations.npz")
        np.load = np_load_old
    
        ranks = npzfile['ranks'][4:]
        results = npzfile['results'][4:]
        returns = [np.mean(vals) for vals in results]
        timesteps = npzfile['timesteps'][4:]

        # reset environment
        env.reset()

        ranks_df = pd.concat([ranks_df, pd.DataFrame([ranks], columns=timesteps)], ignore_index=True)
        returns_df = pd.concat([returns_df, pd.DataFrame([returns], columns=timesteps)], ignore_index=True)

    # ranks vs timestep
    df = ranks_df
    indexes = [col for col in df]
    means = [get_CI(df[col])['mean'] for col in df]
    upper_bounds = [get_CI(df[col])['upper'] for col in df]
    lower_bounds = [get_CI(df[col])['lower'] for col in df]
    
    axis[0].plot(indexes, means, color=color, label=str(gae_lambda))
    axis[0].fill_between(indexes, lower_bounds, upper_bounds, color=color, alpha=.1)

    # returns vs timestep
    df = returns_df
    indexes = [col for col in df]
    means = [get_CI(df[col])['mean'] for col in df]
    upper_bounds = [get_CI(df[col])['upper'] for col in df]
    lower_bounds = [get_CI(df[col])['lower'] for col in df]

    axis[1].plot(indexes, means, color=color, label=str(gae_lambda))
    axis[1].fill_between(indexes, lower_bounds, upper_bounds, color=color, alpha=.1)

axis[0].set_title('Rank vs. Timestep')
axis[0].set_xlabel('Timestep')
axis[0].set_ylabel('Rank')
axis[0].legend()

axis[1].set_title('Return vs. Timestep')
axis[1].set_xlabel('Timestep')
axis[1].set_ylabel('Return')
axis[1].legend()

plt.savefig('{}_plots_64_nodes.png'.format(env_name))

print("DONE!")
