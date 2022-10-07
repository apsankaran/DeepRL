# Stable Baselines SAC

# import statements

import gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make('Ant-v2')

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
