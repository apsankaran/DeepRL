# Stable Baselines SAC

# import statements

import gym
import numpy as np
import os

from CustomSACModel import CustomSAC
from CustomPPOModel import CustomPPO
from stable_baselines3 import PPO

env = gym.make('HalfCheetah-v4')

def mkdr(name):
    if not os.path.exists(name):
        os.makedirs(name)

mkdr('logs')
mkdr('output')

model = CustomPPO('MlpPolicy', env, batch_size=256, n_epochs=16, verbose=1)
model.learn(total_timesteps=200000, log_interval=10)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    print(rewards)
