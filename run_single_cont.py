from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import pdb
import argparse
import random
import gym
import os

from stable_baselines3.common.monitor import Monitor
from CustomPPOModel2 import CustomPPO
from CustomCallback import CustomCallback

parser = argparse.ArgumentParser()

# saving
parser.add_argument('--outfile', default = None)

# common setup
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--num_timesteps', default = 4000000, type = int)

# variables
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--gae_lambda', default = 0.5, type = float)

FLAGS = parser.parse_args()

def mkdr(name):
    if not os.path.exists(name):
        os.makedirs(name)

def env_setup():
    if FLAGS.env_name == 'Ant-v4':
        env = Monitor(gym.make('Ant-v4'))
    return env

def main():
    
    gae_lambda = FLAGS.gae_lambda
    seed = FLAGS.seed
    env = env_setup()
    
    torch.manual_seed(seed)
    np.random.seed(seed) 

    mkdr('logs')
    mkdr('output')

    custom_callback = CustomCallback(env, best_model_save_path=None,
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

    n_epochs=10 # default=10
    model = CustomPPO('MlpPolicy', env, n_epochs=n_epochs, gae_lambda=gae_lambda, verbose=0)
    model.learn(total_timesteps=FLAGS.num_timesteps, callback=custom_callback)

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
    
    summary = {
        'seed': seed,
        'gae_lambda': gae_lambda,
        'ranks': ranks,
        'returns': returns,
        'timesteps': timesteps
    }

    # print(summary)
    np.save(FLAGS.outfile, summary) 

if __name__ == '__main__':
    main()
