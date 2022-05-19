
import sys
sys.path.append("..")

import utils
import gym
from QLearningPolicy import FiniteQLearningModel as QLearning

import forecache_environment
# The typical imports
import gym
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import FiniteMCModel as MC

env = forecache_environment.environment3()
users = env.user_list
accuracies=[]
plot_list=[]

print("For user: ", users[0])
threshold = 0.8
env.process_data(users[0], threshold)

# WARNING: If you try to set eps to a very low value,
# And you attempt to get the m.score() of m.pi, there may not
# be guarranteed convergence.
eps = 3000
S = [0,1,2]
A = [1,2]
START_EPS = 0.7
m = QLearning(S, A, epsilon=START_EPS)

SAVE_FIG = False
history = []

for i in range(1, eps+1):
    ep = []
    prev_observation = env.reset()
    prev_action = m.choose_action(m.b, prev_observation)

    total_reward = 0
    while True:
        # Run simulation
        next_observation, reward, done, _ = env.step(prev_action)
        next_action = m.choose_action(m.b, next_observation)

        m.update_Q((prev_observation, prev_action, reward, next_observation, next_action))

        prev_observation = next_observation
        prev_action = next_action

        total_reward += reward
        if done:
            break

    history.append(total_reward)

cumulative_test_reward, test_accuracy = m.score(env, m.pi, n_samples=10)  # how many times to sample or repeat to get mean of rewards
env.reset(True, False)
