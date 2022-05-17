import forecache_environment
# The typical imports
import gym
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import FiniteMCModel as MC

env = forecache_environment.environment3()
users = env.user_list
threshold=0.8
env.process_data(users[0],threshold)




eps = 100 #how many episodes to train for
# S = [(x, y, z) for x in range(4,22) for y in range(1,11) for z in [True,False]]
# A = 2

S = [0,1,2]
A = [0,1]

m = MC(S, A, epsilon=1)
for i in range(1, eps+1):
    ep = []
    observation = env.reset()
    while True:
        # Choosing behavior policy
        action = m.choose_action(m.b, observation)
        # Run simulation
        next_observation, reward, done, _ = env.step(action)
        ep.append((observation, action, reward))
        observation = next_observation
        if done:
            break

    m.update_Q(ep)
    # Decaying epsilon, reach optimal policy
    m.epsilon = max((eps-i)/eps, 0.1)
    cumulative_test_reward , accuracy =m.score(env, m.pi, n_samples=1000)# how many times to sample or repeat to get mean of rewards
print("Final expected returns : {}".format(cumulative_test_reward))
plt.plot(range(len(accuracy)),accuracy)
plt.show()

