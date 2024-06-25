import numpy as np
import random
import matplotlib.pyplot as plt
### 1
x = np.array(range(1,1001))
plt.figure(figsize=(10, 6))
def reward(a):
    if(a==0):
        return np.random.normal(2,1)
    if(a==1):
        c = np.random.randint(1,3)
        if(c==1):
            return 5
        else: 
            return -6
    if(a==2):
        return np.random.poisson(2)
    if(a==3):
        return np.random.exponential(3)
    if(a==4):
        return reward(np.random.randint(0,4))
### 2
epsilon = 0.1
values = [100, 100, 100, 100, 100]
value_counter = [0,0,0,0,0]
rewards = []
options = [0, 1, 2, 3, 4]
###3
for i in range(1000):
    rwd=0
    for j in range(100):
        choice = -1
        maxval = -100
        for r in values:
            maxval = max(maxval, r)
        indices=[]
        ### 4
        for e in range(5):
            if(values[e]==maxval):
                indices.append(e)
        d = random.uniform(0,1)
        if(d>epsilon):
            choice = random.choice(indices)
        else:
            choice = random.choice(options)
        rew = reward(choice)
        rwd+=rew
        value_counter[choice]+=1
        alpha = 1.0/(value_counter[choice])
        values[choice] += (alpha*(rew - values[choice]))
    
    rewards.append(rwd)
plot_array = np.array(rewards)
plt.plot(x, rewards, label='0.10')
epsilon = 0
values = [100, 100, 100, 100, 100]
value_counter = [0,0,0,0,0]
rewards = []

for i in range(1000):
    rwd=0
    for j in range(100):
        choice = -1
        maxval = -100
        for r in values:
            maxval = max(maxval, r)
        indices=[]
        ### 4
        for e in range(5):
            if(values[e]==maxval):
                indices.append(e)
        d = random.uniform(0,1)
        if(d>epsilon):
            choice = random.choice(indices)
        else:
            choice = random.choice(options)
        rew = reward(choice)
        rwd+=rew
        value_counter[choice]+=1
        alpha = 1.0/(value_counter[choice])
        values[choice] += (alpha*(rew - values[choice]))
    
    rewards.append(rwd)

plot_array = np.array(rewards)
plt.plot(x, rewards, label='0.01')
epsilon = 0.01
values = [100, 100, 100, 100, 100]
value_counter = [0,0,0,0,0]
rewards = []

for i in range(1000):
    rwd=0
    for j in range(100):
        choice = -1
        maxval = -100
        for r in values:
            maxval = max(maxval, r)
        indices=[]
        ### 4
        for e in range(5):
            if(values[e]==maxval):
                indices.append(e)
        d = random.uniform(0,1)
        if(d>epsilon):
            choice = random.choice(indices)
        else:
            choice = random.choice(options)
        rew = reward(choice)
        rwd+=rew
        value_counter[choice]+=1
        alpha = 1.0/(value_counter[choice])
        values[choice] += (alpha*(rew - values[choice]))
    
    rewards.append(rwd)
plot_array = np.array(rewards)
plt.plot(x, rewards, label='0.00')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward at the end of each episode.')
plt.legend()
plt.savefig('initalplot2.png')















