import random
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import make
import pickle

env = make('CartPole-v1')

# Define your state discretization functions
def f(pos):
  s = int(((pos * 20.0)/12.0) + 5)
  s = max(s, 0)
  s = min(s, 9)
  return s
def f2(vel):
  s = int(((vel * 40.0)/11.0) + 5)
  s = max(s, 0)
  s = min(s, 9)
  return s
def f3(apos):
  s = int(((apos * 15.0)/0.419) + 5)
  s = max(s, 0)
  s = min(s, 9)
  return s
def f4(avel):
  s = int(((avel * 20.0)/8.0) + 5)
  s = max(s, 0)
  s = min(s, 9)
  return s

num_episodes = 10000
gamma = 0.99
lamda = 0.97
epsilon = 0.10
number_times = np.ones((10,10,10,10,2))
action_value = np.zeros((10, 10, 10, 10, 2))
r = []

for p in range(num_episodes):
    init = random.uniform(0, 1)
    observation, info = env.reset()
    pos = f(observation[0])
    vel = f2(observation[1])
    apos = f3(observation[2])
    avel = f4(observation[3])
    
    # Epsilon-greedy action selection
    if init > epsilon:
        action = np.argmax(action_value[pos][vel][apos][avel])
    else:
        action = random.choice([0, 1])

    elig_trace = np.zeros((10, 10, 10, 10, 2))
    rwd = 0

    for t in range(500):
        # Adjust alpha over time
        alpha = 0.50 / ((p / 1000) + 1)
        
        # Step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        posdash = f(observation[0])
        veldash = f2(observation[1])
        aposdash = f3(observation[2])
        aveldash = f4(observation[3])
        
        # Epsilon-greedy action selection for next state
        dice = random.uniform(0, 1)
        vals = action_value[posdash][veldash][aposdash][aveldash]
        if dice > epsilon:
            actiondash = np.argmax(vals)
        else:
            actiondash = random.choice([0, 1])
        
        # Compute the TD error
        delta = reward + gamma * action_value[posdash][veldash][aposdash][aveldash][actiondash] - action_value[pos][vel][apos][avel][action]
        
        # Update eligibility trace
        elig_trace[pos][vel][apos][avel][action] += 1
        number_times[pos][vel][apos][avel][action] += 1
        # Update Q-values
        action_value += delta * ( elig_trace / number_times)
        
        # Decay eligibility traces
        elig_trace *= gamma * lamda
        
        pos, vel, apos, avel, action = posdash, veldash, aposdash, aveldash, actiondash
        rwd += reward
        
        if terminated or truncated:
            break
    
    r.append(rwd)

env.close()

# Plot rewards over episodes
print(np.mean(r))
with open('action_value3.pkl', 'wb') as file:
    pickle.dump(action_value, file)

with open('number_times3.pkl', 'wb') as file:
    pickle.dump(number_times, file)
rw = 0
for i in range(100):
    observation, info = env.reset()
    rw = 0
    while True:
        action = np.argmax(action_value[f(observation[0])][f2(observation[1])][f3(observation[2])][f4(observation[3])])
        observation, reward, terminated, truncated, info = env.step(action)
        rw += reward
        if terminated or truncated:
            break
    print(rw)