import gym
from src import MultiArmedBandit, QLearning
import numpy as np
import matplotlib.pyplot as plt

print('Starting example experiment')

env = gym.make('SlotMachines-v0')

# bandit

allarrays = np.empty(shape=[0,100])
for i in range(0,10):
    agent = MultiArmedBandit()
    action_values, rewards = agent.fit(env,steps =100000)
    if i == 0:
        array1mab = rewards
    allarrays= np.vstack([allarrays,rewards])
    print("getting there "+str(i))

array10mab = np.sum(allarrays, axis=0) / 10

first5arrays = allarrays[:-5, :]
array5mab = np.sum(first5arrays, axis=0) / 5

# q learning

allarraysQ = np.empty(shape=[0, 100])
for i in range(0, 10):
    agent = QLearning()
    action_values, rewards = agent.fit(env, steps=100000)
    allarraysQ = np.vstack([allarraysQ, rewards])
    print("making progress "+str(i))
array10Q = np.sum(allarraysQ, axis=0) / 10




# plotting
plt.title('Rewards for Multi-Armed-Bandit')
plt.plot(array1mab, label="First Trial")
plt.plot(array5mab, label="Average of First Five Trials")
plt.plot(array10mab, label="Average of All Trials")
plt.ylabel("Average Reward over 1000 steps")
plt.xlabel("Reward index (x=1 is first 1000 steps, x=2 is next 1000)")
plt.legend()

plt.show()

plt.figure(2)

plt.title('Comparing Rewards for Multi-Armed Bandit and QLearning')
plt.plot(list(range(0,100)), array10mab, label="Multi-Armed Bandit (Average of 10 Trials)")
plt.plot(list(range(0,100)), array10Q, label="QLearning (Average of 10 Trials)")

plt.ylabel("Average Reward over 1000 steps")
plt.xlabel("Reward index (x=1 is first 1000 steps, x=2 is next 1000)")
plt.legend()

plt.show()

print('Finished example experiment')