# Stable Baselines SAC

# import statements

import gym
import numpy as np
import os

from stable_baselines3.common.monitor import Monitor
from CustomPPOModel2 import CustomPPO
from CustomCallback import CustomCallback

import matplotlib.pyplot as plt

env = Monitor(gym.make('Ant-v4'))

def mkdr(name):
    if not os.path.exists(name):
        os.makedirs(name)

mkdr('logs')
mkdr('output')

custom_callback = CustomCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)


n_epochs=64
gae_lambda=1
model = CustomPPO('MlpPolicy', env, n_epochs=n_epochs, gae_lambda=gae_lambda, verbose=1)
model.learn(total_timesteps=4000000, callback=custom_callback)

# get results
env_name = env.unwrapped.spec.id
npzfile = np.load(os.getcwd() + "/logs/evaluations.npz")

ranks = npzfile['ranks']
results = npzfile['results']
timesteps = npzfile['timesteps']
ep_lengths = npzfile['ep_lengths']

average_rewards = [np.mean(vals) for vals in results]
upper_bounds = [np.mean(vals) + np.std(vals) for vals in results]
lower_bounds = [np.mean(vals) - np.std(vals) for vals in results]
model1_ranks = [vals[0] for vals in ranks]
model2_ranks = [vals[1] for vals in ranks]

# plot results
figure, axis = plt.subplots(3, 1, figsize=(15, 30))
figure.tight_layout(pad=5.0)

# reward vs. timesteps
axis[0].plot(timesteps, average_rewards)
axis[0].fill_between(average_rewards, lower_bounds, upper_bounds, color='b', alpha=.1)
axis[0].set_xlabel("Timestep")
axis[0].set_ylabel("Reward")
axis[0].set_title(env_name)

# rank vs. timesteps for model 1
axis[1].plot(timesteps, model1_ranks)
axis[1].set_xlabel("Timestep")
axis[1].set_ylabel("Rank (Model 1)")
axis[1].set_title(env_name)

# rank vs. timesteps for model 2
axis[2].plot(timesteps, model2_ranks)
axis[2].set_xlabel("Timestep")
axis[2].set_ylabel("Rank (Model 2)")
axis[2].set_title(env_name)

# show/save plots
plt.show()
plt.savefig('{}_lambda-{}_epochs-{}_plots.png'.format(env_name,gae_lambda,n_epochs))

# obs = env.reset()
# while True:
    # action, _states = model.predict(obs)
    # obs, rewards, dones, info = env.step(action)
    # env.render()
    # print(rewards)

print("DONE!")
