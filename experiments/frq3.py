import gym
from src import MultiArmedBandit, QLearning
import numpy as np
import matplotlib.pyplot as plt

print('Starting example experiment')

env = gym.make('FrozenLake-v1')

# q learning

allarraysQ = np.empty(shape=[0, 100])

for i in range(0, 10):
    agent = QLearning(epsilon=0.01)
    action_values, rewards = agent.fit(env, steps=100000)
    allarraysQ = np.vstack([allarraysQ, rewards])
    print("progress "+str(i))
array10Q1 = np.sum(allarraysQ, axis=0) / 10

allarraysQ = np.empty(shape=[0, 100])
for i in range(0, 10):
    agent = QLearning(epsilon=0.5)
    action_values, rewards = agent.fit(env, steps=100000)
    allarraysQ = np.vstack([allarraysQ, rewards])
    print("progress "+str(i))

array10Q2 = np.sum(allarraysQ, axis=0) / 10

#seems like this collects QLearning info now just graph it like in FRQ2
# plotting
plt.title('Rewards for Q-Learner in FrozenLake-v1 (after 100,000 Steps)')
plt.plot(list(range(0,100)), array10Q1, label="Average of 10 Trials, Epsilon = 0.01")
plt.plot(list(range(0,100)), array10Q2, label="Average of 10 Trials, Epsilon = 0.5")
plt.ylabel("Average Reward over 1000 steps")
plt.xlabel("Reward index (x=1 is first 1000 steps, x=2 is next 1000)")
#average reward over the first s steps
plt.legend()
plt.show()