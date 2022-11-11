# Stable Baselines SAC

# import statements

import gym
import numpy as np
import os

from CustomSACModel import CustomSAC

env = gym.make('Ant-v4')

def mkdr(name):
    if not os.path.exists(name):
        os.makedirs(name)

mkdr('logs')
mkdr('output')

model = CustomSAC('MlpPolicy', env, batch_size=256, gradient_steps=16, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
