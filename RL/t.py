import numpy as np
import random
import gymnasium as gym
from moviepy.editor import ImageSequenceClip
from IPython.display import Video
import matplotlib.pyplot as plt
env  = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()
epsilon = 0.97
ep_count = 1
frames = []
action_value = np.zeros((20,30,20,20,2))
gamma = 0.9
lamda = 0.9
l = [0,1]
alpha = 0.5
r = []
for p in range(500):
  observation, info = env.reset()
  action = random.choice(l)
  pos = int(((observation[0] * 25.0)/12.0) + 10)
  vel = int(((observation[1] * 50.0)/11.0) + 15)
  apos = int(((observation[2] * 10)/(0.419)) + 10)
  avel = int(((observation[3] * 25.0)/8.0) + 10)
  elig_trace = np.zeros((20,30,20,20,2))
  rwd = 0
  for t in range(500):
      # print(t)
      alpha = 1.0/(t+1)
      if p == 499:
        frames.append(env.render())
      observation, reward, terminated, truncated, info = env.step(action)
      posdash = int(((observation[0] * 25.0)/12.0) + 10)
      veldash = int(((observation[1] * 50.0)/11.0) + 15)
      aposdash = int(((observation[2] * 10)/(0.419)) + 10)
      aveldash = int(((observation[3] * 25.0)/8.0) + 10)
      dice = random.uniform(0, 1)
      actiondash = 0
      #print(posdash, veldash, aposdash, aveldash)
      vals = action_value[posdash][veldash][aposdash][aveldash]
      if vals[1] > vals[0]:
        actiondash = 1
      if vals[0] == vals[1]:
        if dice > 0.5:
          actiondash = 1
      else:
        if dice > (epsilon+1)/2:
          actiondash = 1 - actiondash
      delta = reward + gamma * action_value[posdash][veldash][posdash][aveldash][actiondash] - action_value[pos][vel][apos][avel][action]
      elig_trace[pos][vel][apos][avel][action] += 1      
      action_value += alpha * delta * elig_trace
      elig_trace *= (gamma * lamda)
      pos , vel , apos, avel, action = posdash, veldash, aposdash, aveldash, actiondash
      rwd += reward
      if terminated or truncated:
        break
  r.append(rwd)
env.close()
x = [i + 1 for i in range(500)]
plt.plot(x, r)
plt.show()
clip = ImageSequenceClip(frames, fps=30)
clip.write_videofile('/home/raghav/Desktop/RL/cartpole.mp4')
Video('/home/raghav/Desktop/RL/cartpole.mp4', embed=True)
