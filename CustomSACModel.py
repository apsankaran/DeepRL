### Custom SAC Model

# import statements 

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
import pdb
import random

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

random_seed = 0

th.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

BATCH_SIZE = None
REGULARIZATION_COEFFICIENT = 0.05 # Regularization Coefficient for Custom Loss Functions
phi_matrix_1 = [] # phi matrix for model 1
phi_matrix_2 = [] # phi matrix for model 2
observations_t = None # current state observations
new_observations_t = None # next state observations
online_net = None # online network (not target net)
all_ranks_1 = []
all_ranks_2 = []

# get phi matrix tensor for model and observations+actions
# pass num=1 for first model, num=2 for second model
def get_phi(model, observations_t, num=1):
    layers = [module for module in model.modules() if not isinstance(module, th.nn.Sequential)]
    combined = th.cat((layers[2](observations_t[0]), layers[2](observations_t[1])), dim=1)
    phi_matrix = layers[6+5*(num-1)](layers[5+5*(num-1)](layers[4+5*(num-1)](layers[3+5*(num-1)](combined.float()))))
    return phi_matrix

# default loss function - MSE
def default_loss_function(target_q_values, current_q_values):
    
    loss = th.mean(th.square(target_q_values-current_q_values))
    	
    return loss

# custom loss function - implements explicit DR3 regularizer
# add dot product between each state action and subsequent one’s feature vector to loss
def dr3(target_q_values, current_q_values):
    
    global observations_t, new_observations_t, online_net, BATCHS_SIZE
    
    loss = th.mean(th.square(target_q_values-current_q_values))
        
    curr_states = get_phi(online_net, observations_t, 1)
    next_states = get_phi(online_net, new_observations_t, 1)

    loss += REGULARIZATION_COEFFICIENT * th.sum(th.sum(curr_states * next_states, axis=1)) / BATCH_SIZE

    curr_states = get_phi(online_net, observations_t, 2)
    next_states = get_phi(online_net, new_observations_t, 2)

    loss += REGULARIZATION_COEFFICIENT * th.sum(th.sum(curr_states * next_states, axis=1)) / BATCH_SIZE    

    return loss

# custom loss function - random dot product from phi matrix
# randomly sample two vectors from the phi matrix and add dot product of those vectors to loss
def random_dot(target_q_values, current_q_values):

    global phi_matrix_1, phi_matrix_2

    loss = th.mean(th.square(target_q_values-current_q_values))
    
    phi_matrix = phi_matrix_1
    # Explicit Regularization
    if ((phi_matrix is not None) and (len(phi_matrix) > 1)):
        
        v1 = phi_matrix[random.randrange(len(phi_matrix))]
        v2 = phi_matrix[random.randrange(len(phi_matrix))]
        
        loss += REGULARIZATION_COEFFICIENT * th.dot(th.tensor(v1), th.tensor(v2))
        
    phi_matrix = phi_matrix_2
    # Explicit Regularization
    if ((phi_matrix is not None) and (len(phi_matrix) > 1)):

        v1 = phi_matrix[random.randrange(len(phi_matrix))]
        v2 = phi_matrix[random.randrange(len(phi_matrix))]

        loss += REGULARIZATION_COEFFICIENT * th.dot(th.tensor(v1), th.tensor(v2))

    return loss


# custom loss function - implements regulizer based on min/max singular values in phi matrix
# add difference between max entry in phi matrix ** 2 and min entry in phi matrix ** 2 to loss
def phi_penalty(target_q_values, current_q_values):
    
    global phi_matrix_1, phi_matrix_2
    
    loss = th.mean(th.square(target_q_values-current_q_values))
    
    phi_matrix = phi_matrix_1 
    # Explicit Regularization
    if ((phi_matrix is not None) and (len(phi_matrix) > 0)):
        
        minimum = min([min(value) for value in phi_matrix])
        maximum = max([max(value) for value in phi_matrix])
        
        loss += REGULARIZATION_COEFFICIENT * th.sub(maximum**2, minimum**2)

    phi_matrix = phi_matrix_2
    # Explicit Regularization
    if ((phi_matrix is not None) and (len(phi_matrix) > 0)):

        minimum = min([min(value) for value in phi_matrix])
        maximum = max([max(value) for value in phi_matrix])

        loss += REGULARIZATION_COEFFICIENT * th.sub(maximum**2, minimum**2)

    return loss

# select loss function
loss_function = dr3

outfile = None

class CustomSAC(SAC):

    SACSelf = TypeVar("SACSelf", bound="SAC")
    TOTAL_TIMESTEPS = 10
    def learn(
        self: SACSelf,
        total_timesteps: TOTAL_TIMESTEPS,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
    ) -> SACSelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
	
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:

        # output file path
        global outfile
        if not outfile: 
            outfile = "./logs/loss_{}_regcoeff_{}_seed_{}_gradsteps_{}.txt".format(loss_function.__name__, REGULARIZATION_COEFFICIENT, random_seed, gradient_steps)
            with open(outfile,'w') as f:
                pass
        
        global BATCH_SIZE
        BATCH_SIZE = batch_size
        
        # Switch to train mode (this affects batch norm / dropout)

        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_q_values = th.stack(list(current_q_values), dim=0)
            
            # Set global variables
            global observations_t, new_observations_t, online_net
            observations_t = (replay_data.observations, replay_data.actions)
            new_observations_t = (replay_data.next_observations, next_actions)
            online_net = self.critic

            # Compute critic loss
            critic_loss = loss_function(target_q_values, current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute phi matrix for model 1
            global phi_matrix_1
            phi_matrix_1 = get_phi(self.critic, observations_t, 1)
            phi_matrix_1 = phi_matrix_1.cpu().detach().numpy()
            
            # Compute phi matrix for model 2
            global phi_matrix_2
            phi_matrix_2 = get_phi(self.critic, observations_t, 2)
            phi_matrix_2 = phi_matrix_2.cpu().detach().numpy()

            # Compute Episode Rewards
            def callback_func(self, locals_=None, globals_=None):
                pass

            update_freq = 100
            if self._n_updates + gradient_step == 0 or (self._n_updates + gradient_step + 1) % update_freq == 0:
                self.n_eval_episodes = 10
                episode_rewards, episode_lengths = evaluate_policy (self.policy, self.env, n_eval_episodes=self.n_eval_episodes, render=False, deterministic=True, return_episode_rewards=True, warn=True, callback=callback_func)
                results = (self._n_updates + gradient_step + 1, np.mean(episode_rewards), np.linalg.matrix_rank(phi_matrix_1), np.linalg.matrix_rank(phi_matrix_2))
                with open(outfile, 'a') as f:
                    f.write(str(results) + '\n')
                print(results)
                    
            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
