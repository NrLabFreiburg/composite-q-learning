"""
  This code is based on the original implementation (https://github.com/sfujim/TD3)
  of TD3 (https://arxiv.org/abs/1802.09477).
  
  Supplementary to TMLR 2022 Submission:
                
                Robust and Data-efficient Q-learning by Composite Value-estimation.
"""

import os, sys
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import time

def tt(ndarray):
  return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(state_dim, 400)
    self.fc2 = nn.Linear(400, 300)
    self.fc3 = nn.Linear(300, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    return x

class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, 500)
    self.fc2 = nn.Linear(500, 500)
    self.fc3 = nn.Linear(500, 1)

  def forward(self, xu):
    x, u = xu
    x = F.leaky_relu(self.fc1(torch.cat([x, u], 1)))
    x = F.leaky_relu(self.fc2(x))
    return self.fc3(x)

class TwinCritic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(TwinCritic, self).__init__()
    self._critic1 = Critic(state_dim, action_dim)
    self._critic2 = Critic(state_dim, action_dim)

  def forward(self, xu):
    return self._critic1(xu), self._critic2(xu)

  def Q1(self, xu):
    return self._critic1(xu)

class NStepTD3:
  def __init__(self, state_dim, action_dim):
    self._actor = Actor(state_dim, action_dim)
    self._actor_target = Actor(state_dim, action_dim)

    self._critic = TwinCritic(state_dim, action_dim)
    self._critic_target = TwinCritic(state_dim, action_dim)

    self._actor_target.load_state_dict(self._actor.state_dict())
    self._critic_target.load_state_dict(self._critic.state_dict())

    self._replay_buffer = {"states": [],
                           "actions": [],
                           "next_states": [],
                           "rewards": [],
                           "terminal_flags": [],
                           "size": 0}

    self._batch_size = 200
    self._gamma = 0.99
    self._tau = 0.001 * 5
    self._loss_function = nn.MSELoss()
    self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=0.001)
    self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=0.001)

    self._actor.cuda()
    self._actor_target.cuda()
    self._critic.cuda()
    self._critic_target.cuda()

    self._max_size = 1e6
    self._global_cycle_counter = 0
    self._normal = torch.distributions.normal.Normal(0, 0.2)

    self._n = 10

  def get_action(self, x, noisy=True):
    u = self._actor(tt(x)).cpu().detach().numpy()
    if noisy:
      u += np.random.normal(0, 0.15, u.shape)
    return np.clip(u, -1, 1)

  def train_actor_critic(self):
    batch_indices = np.random.choice(np.arange(self._replay_buffer["size"]), self._batch_size)

    batch_states = tt(np.array([self._replay_buffer["states"][i] for i in batch_indices]))
    batch_actions = tt(np.array([self._replay_buffer["actions"][i] for i in batch_indices]))
    batch_next_states = tt(np.array([self._replay_buffer["next_states"][i] for i in batch_indices]))
    batch_rewards = tt(np.array([self._replay_buffer["rewards"][i] for i in batch_indices])).view(-1, 1)
    batch_tfs = tt(np.array([self._replay_buffer["terminal_flags"][i] for i in batch_indices])).view(-1, 1)

    batch_next_actions = self._actor_target(batch_next_states)
    batch_next_actions += torch.clamp(self._normal.sample(batch_next_actions.size()), -0.5, 0.5).cuda()
    batch_next_actions = torch.clamp(batch_next_actions, -1, 1)

    targets = batch_rewards + (self._gamma**(self._n+1)) * (1-batch_tfs) * torch.min(*self._critic_target([batch_next_states, batch_next_actions]))
    targets = targets.detach()
    
    prediction1, prediction2 = self._critic([batch_states, batch_actions])
    loss = self._loss_function(prediction1, targets) + self._loss_function(prediction2, targets)
    self._critic_optimizer.zero_grad()
    loss.backward()
    self._critic_optimizer.step()

    if self._global_cycle_counter % 2 == 0:
      self._actor_optimizer.zero_grad()
      policy_update = -self._critic.Q1([batch_states, self._actor(batch_states)]).mean()
      policy_update.backward()
      self._actor_optimizer.step()

      soft_update(self._critic_target, self._critic, self._tau)
      soft_update(self._actor_target, self._actor, self._tau)

  def notify_transition(self, state, action, next_state, reward, done):
    self._global_cycle_counter += 1

    self._replay_buffer["states"].append(state)
    self._replay_buffer["actions"].append(action)
    self._replay_buffer["next_states"].append(next_state)
    self._replay_buffer["rewards"].append(reward)
    self._replay_buffer["terminal_flags"].append(1.0 if done else 0.0)
    self._replay_buffer["size"] += 1

    for i in range(1, self._n+1):
      if self._replay_buffer["size"] < (i+1):
        continue
      self._replay_buffer["next_states"][-(i+1)] = next_state
      self._replay_buffer["terminal_flags"][-(i+1)] *= (1.0 if done else 0.0)
      self._replay_buffer["rewards"][-(i+1)] += ((1-self._replay_buffer["terminal_flags"][-i]) * (self._gamma**i) * reward)

    if self._replay_buffer["size"] > self._max_size:
      self._replay_buffer["states"].pop(0)
      self._replay_buffer["actions"].pop(0)
      self._replay_buffer["next_states"].pop(0)
      self._replay_buffer["rewards"].pop(0)
      self._replay_buffer["terminal_flags"].pop(0)
      self._replay_buffer["size"] -= 1
      
    self.train_actor_critic()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--environment', help="Gym Environment", required=True)
  parser.add_argument('-s', '--seed', help="Seed", required=False, type=int)
  args = parser.parse_args()

  if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

  env = gym.make(args.environment)
  global_steps = 400000
  steps = env._max_episode_steps

  controller = NStepTD3(env.observation_space.shape[0], env.action_space.shape[0])
  scale = env.action_space.high[0]

  returns = []
  time_steps = []
  averaged_returns = []

  e = 0
  while e < global_steps:
    x = env.reset()
    for t in range(steps):
      u = controller.get_action(x)
      xp, r, done, _ = env.step(u * scale)
      if t == steps - 1:
        done = False
      controller.notify_transition(x.copy(), u.copy(), xp.copy(), r, done)
      x = xp
      e += 1
      if done:
        break
    # test episode
    x = env.reset()
    ret = 0
    for t in range(steps):
      u = controller.get_action(x, noisy=False)
      xp, r, done, _ = env.step(u * scale)
      ret += r
      x = xp
      if done:
        break
    returns.append(ret)
    averaged_returns.append(np.mean(returns[-100:]))
    time_steps.append(e)
    print('[TD3 -- %s] Step %s: %s' %(args.environment, e, averaged_returns[-1]))

  time_stamp = str(time.time()).replace(".", "")
  np.save("TD3_%s_%s_returns.npy" %(args.environment, time_stamp), averaged_returns)
  np.save("TD3_%s_%s_time_steps.npy" %(args.environment, time_stamp), time_steps)