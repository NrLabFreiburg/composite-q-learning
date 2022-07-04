"""
  Supplementary to TMLR 2022 Submission:
                
                Robust and Data-efficient Q-learning by Composite Value-estimation.
"""

import numpy as np
A = 0
B = 1
C = 2

class Motivation:
  def __init__(self, state_dim):
    self.nS = state_dim
    self.nA = 3
    self.goal = self.nS - 1
    self.trap = self.nS - 2

    self.P = {}
    for i in range(self.nS):
      if i == 0:
        self.P[i] = {A: [(1.0, i+1, -1, False)], B: [(1.0, i, -2, False)], C: [(1.0, i+2, -3, False)]}
      elif i == self.trap-2:
        self.P[i] = {A: [(1.0, i+1, -1, False)], B: [(1.0, i-1, -2, False)], C: [(1.0, i+2, -30, True)]}
      elif i == self.trap-1:
        self.P[i] = {A: [(1.0, i+1, -100, True)], B: [(1.0, i-1, -2, False)], C: [(1.0, i+2, -3, True)]}
      elif i == self.trap or i == self.goal:
        self.P[i] = {A: [(1.0, i, 0, True)], B: [(1.0, i, 0, True)], C: [(1.0, i, 0, True)]}
      else:
        self.P[i] = {A: [(1.0, i+1, -1, False)], B: [(1.0, i-1, -2, False)], C: [(1.0, i+2, -3, False)]}

def generate_data_set(env, episodes, max_steps_per_episode, epsilon=0.3):
  replay_buffer = []
  for e in range(episodes):
    s = 0
    for step in range(max_steps_per_episode):
      a = A
      if s == env.trap-1:
        a = C
      rand = np.random.rand()
      if rand < epsilon:
        a = np.random.choice([B, C])
        if s == env.trap-1:
          a = np.random.choice([A, B])
      p, ns, r, d = env.P[s][a][0]
      replay_buffer.append([s, a, ns, r, d])
      if step + 1 > 1:
        replay_buffer[-2].extend([ns, r, d])
      if step + 1 > 2:
        replay_buffer[-3].extend([ns, r, d])
      if step + 1 > 3:
        replay_buffer[-4].extend([ns, r, d])
      if step + 1 > 4:
        replay_buffer[-5].extend([ns, r, d])
      if d:
        break
      s = ns
  return replay_buffer

def modify_n_step_returns_with_true_model(replay_buffer, env):
  modified_replay_buffer = []
  for e in replay_buffer:
    modified_replay_buffer.append(e[:5])
    s = modified_replay_buffer[-1][2]
    for _ in range(4):
      a = A
      if s == env.trap-1:
        a = C
      p, ns, r, d = env.P[s][a][0]
      modified_replay_buffer[-1].extend([ns, r, d])
      s = ns
      if d:
        break
  return modified_replay_buffer


def q_learning(replay_buffer, updates, alpha=0.001, gamma=1.0):
  s0_q = []
  q = np.zeros((env.nS, env.nA))

  for e in range(updates):
    if (e+1) % 1000 == 0:
      print("%s/%s"%((e+1), updates))
    s, a, ns, r, d = replay_buffer[e%len(replay_buffer)][:5]
    q[s][a] = q[s][a] + alpha * (r + (1-d) * gamma * np.max(q[ns]) - q[s][a])

    s0_q.append((q[0][A], q[0][B]))

  return (s0_q,)

def composite_q_learning(replay_buffer, updates, alpha=0.001, gamma=1.0, alpha_shifted=0.01):
  s0_q = []
  s0_q_0 = []
  s0_q_1 = []
  s0_q_2 = []
  s0_q_3 = []
  q = np.zeros((env.nS, env.nA))
  truncated_q = np.zeros((4, env.nS, env.nA))
  shifted_q = np.zeros((4, env.nS, env.nA))
  for e in range(updates):
    if (e+1) % 1000 == 0:
      print("%s/%s"%((e+1), updates))

    s, a, ns, r, d = replay_buffer[e%len(replay_buffer)][:5]
    old_q = np.copy(q)
    old_truncated_q = np.copy(truncated_q)
    old_shifted_q = np.copy(shifted_q)

    q[s][a] = q[s][a] + alpha * (r + (1-d) * gamma * (old_truncated_q[3][ns][np.argmax(old_q[ns])] + old_shifted_q[3][ns][np.argmax(old_q[ns])]) - q[s][a])
    truncated_q[0][s][a] = truncated_q[0][s][a] + alpha * (r - truncated_q[0][s][a])
    truncated_q[1][s][a] = truncated_q[1][s][a] + alpha * (r + (1-d) * gamma * old_truncated_q[0][ns][np.argmax(old_q[ns])] - truncated_q[1][s][a])
    truncated_q[2][s][a] = truncated_q[2][s][a] + alpha * (r + (1-d) * gamma * old_truncated_q[1][ns][np.argmax(old_q[ns])] - truncated_q[2][s][a])
    truncated_q[3][s][a] = truncated_q[3][s][a] + alpha * (r + (1-d) * gamma * old_truncated_q[2][ns][np.argmax(old_q[ns])] - truncated_q[3][s][a])

    shifted_q[0][s][a] = shifted_q[0][s][a] + alpha_shifted * ((1-d) * gamma * np.max(old_q[ns]) - shifted_q[0][s][a])
    shifted_q[1][s][a] = shifted_q[1][s][a] + alpha_shifted * ((1-d) * gamma * old_shifted_q[0][ns][np.argmax(old_q[ns])] - shifted_q[1][s][a])
    shifted_q[2][s][a] = shifted_q[2][s][a] + alpha_shifted * ((1-d) * gamma * old_shifted_q[1][ns][np.argmax(old_q[ns])] - shifted_q[2][s][a])
    shifted_q[3][s][a] = shifted_q[3][s][a] + alpha_shifted * ((1-d) * gamma * old_shifted_q[2][ns][np.argmax(old_q[ns])] - shifted_q[3][s][a])

    s0_q.append((q[0][A], q[0][B]))
    s0_q_0.append((truncated_q[0][0][A], truncated_q[0][0][B]))
    s0_q_1.append((truncated_q[1][0][A], truncated_q[1][0][B]))
    s0_q_2.append((truncated_q[2][0][A], truncated_q[2][0][B]))
    s0_q_3.append((truncated_q[3][0][A], truncated_q[3][0][B]))

  return s0_q, s0_q_0, s0_q_1, s0_q_2, s0_q_3

def on_policy_multi_step_q_learning(replay_buffer, updates, alpha=0.001, gamma=1.0):
  s0_q = []
  q = np.zeros((env.nS, env.nA))

  for e in range(updates):
    if (e+1) % 1000 == 0:
      print("%s/%s"%((e+1), updates))
    s, a = replay_buffer[e%len(replay_buffer)][:2]
    next_transitions = replay_buffer[e%len(replay_buffer)][2:]
    d = False
    discount = gamma
    target = 0
    for i in range(0, len(next_transitions), 3):
      ns = next_transitions[i]
      target += ((1-d) * discount * next_transitions[i+1])
      d |= next_transitions[i+2]
      discount *= gamma
    target += ((1-d) * discount * np.max(q[ns]))
    q[s][a] = q[s][a] + alpha * (target - q[s][a])

    s0_q.append((q[0][A], q[0][B]))

  return (s0_q,)

def run(f, replay_buffer, updates, alpha, gamma, var_names):
  ret = f(replay_buffer, updates, alpha, gamma)
  for name, v in zip(var_names, ret):
    np.save(name, v)

if __name__ == "__main__":
  env = Motivation(20)
  max_episode_length = 100
  episodes = 1000
  alpha = 0.001
  updates = 5000000
  gamma = 1.0

  replay_buffer = generate_data_set(env, episodes, max_episode_length, epsilon=0.1)
  run(q_learning, replay_buffer, updates, alpha, gamma, ["s0_q"])
  run(composite_q_learning, replay_buffer, updates, alpha, gamma, ["s0_q_c", "s0_q_0", "s0_q_1", "s0_q_2", "s0_q_3"])
  run(on_policy_multi_step_q_learning, replay_buffer, updates, alpha, gamma, ["s0_q_o"])
  
  replay_buffer = modify_n_step_returns_with_true_model(replay_buffer, env)
  run(on_policy_multi_step_q_learning, replay_buffer, updates, alpha, gamma, ["s0_q_mbo"])
